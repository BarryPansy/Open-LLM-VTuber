# agent/agents/rag_agent.py
from typing import AsyncIterator, List, Dict, Any, Optional, Union
import asyncio
from loguru import logger

from .agent_interface import AgentInterface
from ..output_types import SentenceOutput
from ..transformers import (
    sentence_divider,
    actions_extractor,
    tts_filter,
    display_processor,
)
from ...config_manager import TTSPreprocessorConfig
from ..input_types import BatchInput, TextSource
from ...rag import RAGEngine
from ...config_manager.rag import RAGConfig


class RAGAgent(AgentInterface):
    """支持RAG（检索增强生成）的Agent实现"""
    
    def __init__(
        self,
        live2d_model,
        id: str,
        tts_preprocessor_config: TTSPreprocessorConfig = None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        rag_config: RAGConfig = None,
        llm_provider: str = "openai_compatible_llm",
        llm_configs: Dict[str, Any] = None,
        system_prompt: str = "",
        character_avatar: str = "",
        system_config: Dict[str, Any] = None,
        tool_manager=None,
        tool_executor=None,
        mcp_prompt_string: str = "",
    ):
        super().__init__()
        
        self.id = id
        self._tts_preprocessor_config = tts_preprocessor_config
        self._live2d_model = live2d_model
        self._faster_first_response = faster_first_response
        self._segment_method = segment_method
        self._system_prompt = system_prompt
        self._character_avatar = character_avatar
        self._system_config = system_config or {}
        self._tool_manager = tool_manager
        self._tool_executor = tool_executor
        self._mcp_prompt_string = mcp_prompt_string
        
        # RAG相关配置
        self.rag_config = rag_config
        self.rag_engine = None
        self._rag_initialized = False
        
        # LLM配置
        self.llm_provider = llm_provider
        self.llm_configs = llm_configs or {}
        self._llm = None
        
        # 对话历史
        self._conversation_history = []
        self._interrupt_handled = False
        
        # 初始化RAG引擎
        if self.rag_config and self.rag_config.enabled:
            self.rag_engine = RAGEngine(self.rag_config)
            asyncio.create_task(self._initialize_rag())
        
        # 延迟应用装饰器
        self.chat = tts_filter(self._tts_preprocessor_config)(
            display_processor()(
                actions_extractor(self._live2d_model)(
                    sentence_divider(
                        faster_first_response=self._faster_first_response,
                        segment_method=self._segment_method,
                        valid_tags=["think"],
                    )(self._chat_function_factory)
                )
            )
        )
    
    async def _initialize_rag(self):
        """异步初始化RAG引擎"""
        try:
            if self.rag_engine:
                success = await self.rag_engine.initialize()
                self._rag_initialized = success
                if success:
                    logger.info("RAG engine initialized successfully")
                else:
                    logger.warning("RAG engine initialization failed")
        except Exception as e:
            logger.error(f"Error initializing RAG engine: {e}")
            self._rag_initialized = False
    
    def _initialize_llm(self):
        """初始化LLM"""
        if self._llm is not None:
            return
        
        try:
            from ..stateless_llm_factory import LLMFactory
            
            llm_config = self.llm_configs.get(self.llm_provider, {})
            self._llm = LLMFactory.create_llm(self.llm_provider, **llm_config)
            logger.info(f"LLM initialized: {self.llm_provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    async def _get_rag_context(self, query: str) -> str:
        """获取RAG上下文"""
        if not self._rag_initialized or not self.rag_engine:
            return ""
        
        try:
            # 检查是否需要更新知识库
            if await self.rag_engine.should_update():
                await self.rag_engine.update_knowledge_base()
            
            # 获取上下文
            context = await self.rag_engine.get_context(query)
            return context
        except Exception as e:
            logger.error(f"Error getting RAG context: {e}")
            return ""
    
    async def _chat_with_rag(self, input_data: BatchInput) -> AsyncIterator[str]:
        """带RAG的聊天实现"""
        try:
            # 提取用户输入
            user_input = ""
            if input_data.texts and len(input_data.texts) > 0:
                # 获取第一个文本输入
                user_input = input_data.texts[0].content
            
            if not user_input:
                return
            
            # 获取RAG上下文
            rag_context = ""
            if self._rag_initialized:
                rag_context = await self._get_rag_context(user_input)
                if rag_context:
                    logger.info(f"RAG context retrieved for query: '{user_input[:50]}...'")
                    logger.debug(f"RAG context length: {len(rag_context)} characters")
                else:
                    logger.info(f"No RAG context found for query: '{user_input[:50]}...'")
            
            # 构建增强的系统提示
            enhanced_system_prompt = self._build_enhanced_system_prompt(rag_context)
            
            # 初始化LLM
            self._initialize_llm()
            
            # 构建消息
            messages = self._build_messages(user_input, enhanced_system_prompt)
            
            # 调用LLM
            token_stream = self._llm.chat_completion(messages, enhanced_system_prompt)
            
            complete_response = ""
            async for event in token_stream:
                text_chunk = ""
                if isinstance(event, dict) and event.get("type") == "text_delta":
                    text_chunk = event.get("text", "")
                elif isinstance(event, str):
                    text_chunk = event
                else:
                    continue
                
                if text_chunk:
                    yield text_chunk
                    complete_response += text_chunk
            
            # 保存到对话历史
            if complete_response:
                self._add_message(complete_response, "assistant")
                self._add_message(user_input, "user")
        
        except Exception as e:
            logger.error(f"Error in RAG chat: {e}")
            yield f"抱歉，处理您的请求时出现了错误：{str(e)}"
    
    def _build_enhanced_system_prompt(self, rag_context: str) -> str:
        """构建增强的系统提示"""
        base_prompt = self._system_prompt
        
        if rag_context:
            rag_instruction = """
基于以下检索到的相关信息来回答用户的问题。如果相关信息不足以回答问题，请明确说明并基于你的知识进行回答。

相关信息：
{rag_context}

请确保你的回答：
1. 优先使用检索到的相关信息
2. 如果信息不足，明确说明并补充你的知识
3. 保持回答的准确性和相关性
4. 引用信息来源（如果适用）
""".format(rag_context=rag_context)
            
            enhanced_prompt = f"{base_prompt}\n\n{rag_instruction}"
        else:
            enhanced_prompt = base_prompt
        
        return enhanced_prompt
    
    def _build_messages(self, user_input: str, system_prompt: str) -> List[Dict[str, Any]]:
        """构建消息列表"""
        messages = [{"role": "system", "content": system_prompt}]
        
        # 添加对话历史
        for msg in self._conversation_history[-10:]:  # 只保留最近10条消息
            messages.append(msg)
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    async def _chat_function_factory(self, input_data: BatchInput) -> AsyncIterator[str]:
        """创建聊天函数工厂，用于装饰器链"""
        async for output in self._chat_with_rag(input_data):
            yield output
    
    def _add_message(self, content: str, role: str):
        """添加消息到对话历史"""
        self._conversation_history.append({
            "role": role,
            "content": content
        })
    
    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        """设置历史记忆（RAG Agent不需要特殊处理）"""
        pass
    
    def handle_interrupt(self, heard_response: str) -> None:
        """处理中断"""
        self._interrupt_handled = True
        logger.info("RAG Agent interrupt handled")
    
    def reset_interrupt(self) -> None:
        """重置中断标志"""
        self._interrupt_handled = False
    
    async def get_rag_stats(self) -> Dict[str, Any]:
        """获取RAG统计信息"""
        if not self.rag_engine:
            return {"rag_enabled": False}
        
        try:
            stats = await self.rag_engine.get_stats()
            health = await self.rag_engine.health_check()
            return {
                "rag_enabled": True,
                "rag_initialized": self._rag_initialized,
                "stats": stats,
                "health": health
            }
        except Exception as e:
            return {
                "rag_enabled": True,
                "rag_initialized": False,
                "error": str(e)
            }
    
    async def add_document_to_knowledge_base(self, file_path: str) -> bool:
        """添加文档到知识库"""
        if not self.rag_engine or not self._rag_initialized:
            logger.warning("RAG engine not initialized")
            return False
        
        try:
            return await self.rag_engine.add_document(file_path)
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    async def search_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索知识库"""
        if not self.rag_engine or not self._rag_initialized:
            return []
        
        try:
            return await self.rag_engine.search(query, top_k)
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []
    
    async def chat(self, input_data: BatchInput) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:
        """实现AgentInterface的chat方法"""
        chat_func_decorated = self._chat_function_factory()
        async for output in chat_func_decorated(input_data):
            yield output

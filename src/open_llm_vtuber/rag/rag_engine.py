# rag/rag_engine.py
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

from .document_processor import DocumentProcessor
from .vector_store import create_vector_store
from .retriever import DocumentRetriever
from ..config_manager.rag import RAGConfig


class RAGEngine:
    """RAG引擎主类，协调文档处理、向量存储和检索"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.document_processor = DocumentProcessor(config)
        self.vector_store = create_vector_store(config)
        self.retriever = DocumentRetriever(self.vector_store, config)
        self.is_initialized = False
        self._last_update_time = 0
    
    async def initialize(self) -> bool:
        """初始化RAG引擎"""
        try:
            if not self.config.enabled:
                logger.info("RAG is disabled in configuration")
                return False
            
            # 检查知识库路径
            knowledge_path = Path(self.config.knowledge_base_path)
            if not knowledge_path.exists():
                logger.warning(f"Knowledge base path {knowledge_path} does not exist, creating it")
                knowledge_path.mkdir(parents=True, exist_ok=True)
            
            # 处理文档
            documents = await self.document_processor.process_documents(self.config.knowledge_base_path)
            
            if not documents:
                logger.warning("No documents found in knowledge base, but RAG engine will still be initialized")
                # 即使没有文档，也标记为已初始化，这样后续可以添加文档
                self.is_initialized = True
                return True
            
            # 生成嵌入向量
            logger.info("Generating embeddings for documents...")
            embeddings = await self.document_processor.generate_embeddings(documents)
            
            # 添加到向量存储
            logger.info("Adding documents to vector store...")
            await self.vector_store.add_documents(documents, embeddings)
            
            self.is_initialized = True
            logger.info(f"RAG engine initialized successfully with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            return False
    
    async def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """搜索相关文档"""
        if not self.is_initialized:
            logger.warning("RAG engine not initialized")
            return []
        
        try:
            # 检索文档
            results = await self.retriever.retrieve_documents(query, top_k)
            
            # 格式化结果
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    async def get_context(self, query: str, max_documents: int = None) -> str:
        """获取用于LLM的上下文"""
        if not self.is_initialized:
            return ""
        
        try:
            return await self.retriever.format_context_for_llm(query, max_documents)
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return ""
    
    async def update_knowledge_base(self) -> bool:
        """更新知识库"""
        if not self.config.auto_update:
            return False
        
        try:
            logger.info("Updating knowledge base...")
            
            # 重新处理文档
            documents = await self.document_processor.process_documents(self.config.knowledge_base_path)
            
            if not documents:
                logger.warning("No documents found during update")
                return False
            
            # 清空现有向量存储
            # 注意：这里简化处理，实际应用中可能需要增量更新
            await self.vector_store.delete_documents([])  # 清空所有文档
            
            # 生成新的嵌入向量
            embeddings = await self.document_processor.generate_embeddings(documents)
            
            # 添加到向量存储
            await self.vector_store.add_documents(documents, embeddings)
            
            self._last_update_time = asyncio.get_event_loop().time()
            logger.info(f"Knowledge base updated with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error updating knowledge base: {e}")
            return False
    
    async def add_document(self, file_path: str) -> bool:
        """添加单个文档到知识库"""
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                logger.error(f"File {file_path} does not exist")
                return False
            
            # 处理文档
            documents = await self.document_processor._process_single_file(file_path_obj)
            if not documents:
                return False
            
            # 生成嵌入向量
            embeddings = await self.document_processor.generate_embeddings(documents)
            
            # 添加到向量存储
            await self.vector_store.add_documents(documents, embeddings)
            
            logger.info(f"Added document {file_path} to knowledge base")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            return False
    
    async def remove_document(self, file_path: str) -> bool:
        """从知识库中移除文档"""
        try:
            # 查找相关的文档ID
            # 这里简化处理，实际应用中需要更复杂的逻辑
            file_path_obj = Path(file_path)
            file_name = file_path_obj.name
            
            # 获取所有文档进行匹配（这里简化处理）
            # 实际应用中应该在向量存储中维护文件到文档ID的映射
            logger.warning("Document removal not fully implemented - requires document ID tracking")
            return False
            
        except Exception as e:
            logger.error(f"Error removing document {file_path}: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取RAG引擎统计信息"""
        if not self.is_initialized:
            return {"initialized": False}
        
        try:
            retrieval_stats = await self.retriever.get_retrieval_stats()
            return {
                "initialized": True,
                "enabled": self.config.enabled,
                "knowledge_base_path": self.config.knowledge_base_path,
                "vector_store_type": self.config.vector_store_type,
                "embedding_model": self.config.embedding_model,
                **retrieval_stats
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"initialized": False, "error": str(e)}
    
    async def should_update(self) -> bool:
        """检查是否需要更新知识库"""
        if not self.config.auto_update:
            return False
        
        current_time = asyncio.get_event_loop().time()
        return (current_time - self._last_update_time) > self.config.update_interval
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            stats = await self.get_stats()
            
            # 检查知识库路径
            knowledge_path = Path(self.config.knowledge_base_path)
            path_exists = knowledge_path.exists()
            
            # 检查向量存储
            doc_count = await self.vector_store.get_document_count()
            
            return {
                "status": "healthy" if stats.get("initialized", False) else "unhealthy",
                "initialized": stats.get("initialized", False),
                "knowledge_base_exists": path_exists,
                "document_count": doc_count,
                "vector_store_type": self.config.vector_store_type,
                "last_update": self._last_update_time
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

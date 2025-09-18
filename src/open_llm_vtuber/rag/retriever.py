# rag/retriever.py
from typing import List, Tuple, Dict, Any
from langchain.schema import Document
from loguru import logger

from .vector_store import VectorStoreManager
from ..config_manager.rag import RAGConfig


class DocumentRetriever:
    """文档检索器，负责从向量存储中检索相关文档"""
    
    def __init__(self, vector_store: VectorStoreManager, config: RAGConfig):
        self.vector_store = vector_store
        self.config = config
    
    async def retrieve_documents(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """检索相关文档"""
        if top_k is None:
            top_k = self.config.retrieval_top_k
        
        # 执行相似度搜索
        results = await self.vector_store.similarity_search(query, top_k)
        
        # 应用分数阈值过滤
        filtered_results = [
            (doc, score) for doc, score in results 
            if score >= self.config.retrieval_score_threshold
        ]
        
        logger.debug(f"Retrieved {len(filtered_results)} documents for query: {query[:50]}...")
        return filtered_results
    
    async def retrieve_with_reranking(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """检索文档并进行重排序"""
        # 首先获取更多文档
        initial_k = (top_k or self.config.retrieval_top_k) * 2
        results = await self.retrieve_documents(query, initial_k)
        
        if not results:
            return results
        
        # 简单的重排序：基于文档长度和相似度分数的组合
        def rerank_score(doc_score_tuple):
            doc, score = doc_score_tuple
            # 考虑文档长度（避免过短或过长的文档）
            content_length = len(doc.page_content)
            length_score = min(1.0, content_length / 500)  # 500字符为理想长度
            
            # 组合分数
            combined_score = score * 0.8 + length_score * 0.2
            return combined_score
        
        # 重排序
        reranked_results = sorted(results, key=rerank_score, reverse=True)
        
        # 返回top-k结果
        final_k = top_k or self.config.retrieval_top_k
        return reranked_results[:final_k]
    
    async def get_context_documents(self, query: str, max_documents: int = None) -> List[Document]:
        """获取用于上下文的文档（不包含分数）"""
        if max_documents is None:
            max_documents = self.config.max_context_documents
        
        results = await self.retrieve_documents(query, max_documents)
        return [doc for doc, score in results]
    
    async def format_context_for_llm(self, query: str, max_documents: int = None) -> str:
        """格式化检索到的文档为LLM上下文"""
        documents = await self.get_context_documents(query, max_documents)
        
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content.strip()
            
            context_part = f"[文档 {i} - 来源: {source}]\n{content}\n"
            context_parts.append(context_part)
        
        context = "\n".join(context_parts)
        
        # 检查上下文长度限制
        if len(context) > self.config.context_window_size:
            # 截断到合适长度
            context = context[:self.config.context_window_size]
            context += "\n\n[上下文已截断...]"
        
        return context
    
    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        total_docs = await self.vector_store.get_document_count()
        
        return {
            "total_documents": total_docs,
            "retrieval_top_k": self.config.retrieval_top_k,
            "score_threshold": self.config.retrieval_score_threshold,
            "vector_store_type": self.config.vector_store_type
        }

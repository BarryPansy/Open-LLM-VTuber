# rag/vector_store.py
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import asyncio
from loguru import logger

import chromadb
from chromadb.config import Settings
import faiss
import numpy as np
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from ..config_manager.rag import RAGConfig


class VectorStoreManager:
    """管理向量存储的抽象类"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': config.embedding_device}
        )
    
    async def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """添加文档到向量存储"""
        raise NotImplementedError
    
    async def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """相似度搜索"""
        raise NotImplementedError
    
    async def delete_documents(self, document_ids: List[str]) -> None:
        """删除文档"""
        raise NotImplementedError
    
    async def get_document_count(self) -> int:
        """获取文档数量"""
        raise NotImplementedError


class ChromaDBVectorStore(VectorStoreManager):
    """ChromaDB向量存储实现"""
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化ChromaDB客户端"""
        persist_directory = self.config.chromadb_config.get("persist_directory", "./chroma_db")
        collection_name = self.config.chromadb_config.get("collection_name", "knowledge_base")
        
        # 确保目录存在
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except (ValueError, Exception) as e:
            # 如果集合不存在或出现其他错误，创建新集合
            logger.info(f"Collection {collection_name} not found or error occurred: {e}")
            try:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "Knowledge base for RAG"}
                )
                logger.info(f"Created new collection: {collection_name}")
            except Exception as create_error:
                logger.error(f"Failed to create collection {collection_name}: {create_error}")
                # 如果创建失败，尝试使用默认集合
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": "Knowledge base for RAG"}
                )
                logger.info(f"Got or created collection: {collection_name}")
    
    async def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """添加文档到ChromaDB"""
        if not documents or not embeddings:
            return
        
        # 准备数据
        ids = [doc.metadata.get('chunk_id', f"doc_{i}") for i, doc in enumerate(documents)]
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # 转换为numpy数组
        embeddings_array = np.array(embeddings).astype(np.float32)
        
        # 添加到集合
        self.collection.add(
            ids=ids,
            embeddings=embeddings_array.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    async def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """在ChromaDB中进行相似度搜索"""
        if not self.collection:
            return []
        
        try:
            # 检查集合是否为空
            count = self.collection.count()
            if count == 0:
                logger.debug("Collection is empty, returning empty results")
                return []
            
            # 生成查询嵌入
            query_embedding = self.embeddings.embed_query(query)
            
            # 搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
        
        # 构建结果
        documents_with_scores = []
        if results['documents'] and results['documents'][0]:
            for i, (doc_text, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # 将距离转换为相似度分数（ChromaDB返回的是距离，越小越相似）
                similarity_score = 1.0 / (1.0 + distance)
                
                doc = Document(
                    page_content=doc_text,
                    metadata=metadata
                )
                documents_with_scores.append((doc, similarity_score))
        
        return documents_with_scores
    
    async def delete_documents(self, document_ids: List[str]) -> None:
        """从ChromaDB删除文档"""
        if not self.collection:
            return
        
        self.collection.delete(ids=document_ids)
        logger.info(f"Deleted {len(document_ids)} documents from ChromaDB")
    
    async def get_document_count(self) -> int:
        """获取ChromaDB中的文档数量"""
        if not self.collection:
            return 0
        
        return self.collection.count()


class FAISSVectorStore(VectorStoreManager):
    """FAISS向量存储实现"""
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.index = None
        self.documents = []
        self.index_path = config.faiss_config.get("index_path", "./faiss_index")
        self.index_type = config.faiss_config.get("index_type", "IndexFlatIP")
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """加载或创建FAISS索引"""
        index_file = Path(self.index_path) / "faiss_index.bin"
        metadata_file = Path(self.index_path) / "metadata.json"
        
        if index_file.exists() and metadata_file.exists():
            try:
                # 加载现有索引
                self.index = faiss.read_index(str(index_file))
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.documents = [Document(**doc_data) for doc_data in json.load(f)]
                logger.info(f"Loaded FAISS index with {len(self.documents)} documents")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """创建新的FAISS索引"""
        os.makedirs(self.index_path, exist_ok=True)
        
        # 创建空索引（维度将在添加第一个文档时确定）
        self.index = None
        self.documents = []
        logger.info("Created new FAISS index")
    
    async def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """添加文档到FAISS索引"""
        if not documents or not embeddings:
            return
        
        embeddings_array = np.array(embeddings).astype(np.float32)
        
        if self.index is None:
            # 创建新索引
            dimension = embeddings_array.shape[1]
            if self.index_type == "IndexFlatIP":
                self.index = faiss.IndexFlatIP(dimension)
            else:
                self.index = faiss.IndexFlatL2(dimension)
        
        # 添加到索引
        self.index.add(embeddings_array)
        self.documents.extend(documents)
        
        # 保存索引和元数据
        await self._save_index()
        
        logger.info(f"Added {len(documents)} documents to FAISS index")
    
    async def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """在FAISS中进行相似度搜索"""
        if not self.index or not self.documents:
            return []
        
        # 生成查询嵌入
        query_embedding = self.embeddings.embed_query(query)
        query_array = np.array([query_embedding]).astype(np.float32)
        
        # 搜索
        scores, indices = self.index.search(query_array, min(k, len(self.documents)))
        
        # 构建结果
        documents_with_scores = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                # FAISS返回的是相似度分数（对于IndexFlatIP）或距离（对于IndexFlatL2）
                if self.index_type == "IndexFlatIP":
                    similarity_score = float(score)
                else:
                    similarity_score = 1.0 / (1.0 + float(score))
                
                documents_with_scores.append((doc, similarity_score))
        
        return documents_with_scores
    
    async def delete_documents(self, document_ids: List[str]) -> None:
        """从FAISS删除文档（重建索引）"""
        if not self.documents:
            return
        
        # 过滤掉要删除的文档
        original_count = len(self.documents)
        self.documents = [
            doc for doc in self.documents 
            if doc.metadata.get('chunk_id') not in document_ids
        ]
        
        if len(self.documents) < original_count:
            # 重建索引
            if self.documents:
                # 重新生成嵌入向量
                texts = [doc.page_content for doc in self.documents]
                embeddings = self.embeddings.embed_documents(texts)
                embeddings_array = np.array(embeddings).astype(np.float32)
                
                # 创建新索引
                dimension = embeddings_array.shape[1]
                if self.index_type == "IndexFlatIP":
                    self.index = faiss.IndexFlatIP(dimension)
                else:
                    self.index = faiss.IndexFlatL2(dimension)
                
                self.index.add(embeddings_array)
                await self._save_index()
            
            logger.info(f"Deleted {original_count - len(self.documents)} documents from FAISS")
    
    async def get_document_count(self) -> int:
        """获取FAISS中的文档数量"""
        return len(self.documents) if self.documents else 0
    
    async def _save_index(self):
        """保存FAISS索引和元数据"""
        if not self.index:
            return
        
        os.makedirs(self.index_path, exist_ok=True)
        
        # 保存索引
        index_file = Path(self.index_path) / "faiss_index.bin"
        faiss.write_index(self.index, str(index_file))
        
        # 保存元数据
        metadata_file = Path(self.index_path) / "metadata.json"
        metadata_data = [doc.dict() for doc in self.documents]
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_data, f, ensure_ascii=False, indent=2)


class MemoryVectorStore(VectorStoreManager):
    """内存向量存储实现（用于测试或临时使用）"""
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.documents = []
        self.embeddings_list = []
    
    async def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """添加文档到内存存储"""
        self.documents.extend(documents)
        self.embeddings_list.extend(embeddings)
        logger.info(f"Added {len(documents)} documents to memory storage")
    
    async def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """在内存中进行相似度搜索"""
        if not self.documents:
            return []
        
        # 生成查询嵌入
        query_embedding = self.embeddings.embed_query(query)
        
        # 计算相似度
        similarities = []
        for embedding in self.embeddings_list:
            # 使用余弦相似度
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append(similarity)
        
        # 获取top-k结果
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        documents_with_scores = []
        for idx in top_indices:
            if idx < len(self.documents):
                doc = self.documents[idx]
                score = similarities[idx]
                documents_with_scores.append((doc, score))
        
        return documents_with_scores
    
    async def delete_documents(self, document_ids: List[str]) -> None:
        """从内存删除文档"""
        original_count = len(self.documents)
        indices_to_remove = []
        
        for i, doc in enumerate(self.documents):
            if doc.metadata.get('chunk_id') in document_ids:
                indices_to_remove.append(i)
        
        # 从后往前删除，避免索引变化
        for i in reversed(indices_to_remove):
            del self.documents[i]
            del self.embeddings_list[i]
        
        logger.info(f"Deleted {len(indices_to_remove)} documents from memory storage")
    
    async def get_document_count(self) -> int:
        """获取内存中的文档数量"""
        return len(self.documents)


def create_vector_store(config: RAGConfig) -> VectorStoreManager:
    """根据配置创建向量存储"""
    if config.vector_store_type == "chromadb":
        return ChromaDBVectorStore(config)
    elif config.vector_store_type == "faiss":
        return FAISSVectorStore(config)
    elif config.vector_store_type == "memory":
        return MemoryVectorStore(config)
    else:
        raise ValueError(f"Unsupported vector store type: {config.vector_store_type}")

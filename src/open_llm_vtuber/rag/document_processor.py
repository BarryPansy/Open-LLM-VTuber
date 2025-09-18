# rag/document_processor.py
import os
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from loguru import logger

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# 尝试导入可选的文档加载器
try:
    from langchain_community.document_loaders import (
        UnstructuredMarkdownLoader,
        UnstructuredPDFLoader,
        UnstructuredWordDocumentLoader,
        UnstructuredHTMLLoader,
    )
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logger.warning("Unstructured library not available, only .txt files will be processed")

from ..config_manager.rag import RAGConfig


class DocumentProcessor:
    """处理各种格式的文档，将其分割成块并生成嵌入向量"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        
        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': config.embedding_device}
        )
        
        # 文件类型映射
        self.loader_mapping = {
            '.txt': TextLoader,
        }
        
        # 如果unstructured库可用，添加更多文件类型支持
        if UNSTRUCTURED_AVAILABLE:
            self.loader_mapping.update({
                '.md': UnstructuredMarkdownLoader,
                '.pdf': UnstructuredPDFLoader,
                '.docx': UnstructuredWordDocumentLoader,
                '.html': UnstructuredHTMLLoader,
            })
        else:
            # 作为备选方案，将.md文件也当作文本文件处理
            self.loader_mapping['.md'] = TextLoader
            logger.info("Using simplified document processing (only .txt and .md as text files)")
    
    async def process_documents(self, knowledge_base_path: str) -> List[Document]:
        """处理知识库中的所有文档"""
        documents = []
        knowledge_path = Path(knowledge_base_path)
        
        if not knowledge_path.exists():
            logger.warning(f"Knowledge base path {knowledge_base_path} does not exist")
            return documents
        
        # 遍历知识库目录
        for file_path in knowledge_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.config.supported_file_types:
                try:
                    file_docs = await self._process_single_file(file_path)
                    documents.extend(file_docs)
                    logger.info(f"Processed {file_path}: {len(file_docs)} chunks")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"Total documents processed: {len(documents)}")
        return documents
    
    async def _process_single_file(self, file_path: Path) -> List[Document]:
        """处理单个文件"""
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.loader_mapping:
            logger.warning(f"Unsupported file type: {file_extension}")
            return []
        
        # 使用对应的加载器
        loader_class = self.loader_mapping[file_extension]
        
        try:
            if file_extension in ['.txt', '.md']:
                # 对于文本文件，尝试不同编码
                try:
                    loader = loader_class(str(file_path), encoding='utf-8')
                    raw_documents = loader.load()
                except UnicodeDecodeError:
                    try:
                        loader = loader_class(str(file_path), encoding='gbk')
                        raw_documents = loader.load()
                    except UnicodeDecodeError:
                        loader = loader_class(str(file_path), encoding='latin-1')
                        raw_documents = loader.load()
            else:
                loader = loader_class(str(file_path))
                raw_documents = loader.load()
                
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return []
        
        # 添加元数据
        for doc in raw_documents:
            doc.metadata.update({
                'source': str(file_path),
                'file_name': file_path.name,
                'file_type': file_extension,
                'file_hash': self._calculate_file_hash(file_path)
            })
        
        # 分割文档
        split_documents = self.text_splitter.split_documents(raw_documents)
        
        # 为每个块添加唯一ID
        for i, doc in enumerate(split_documents):
            doc.metadata['chunk_id'] = f"{file_path.stem}_{i}"
            doc.metadata['chunk_index'] = i
        
        return split_documents
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值用于检测文件变化"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    async def generate_embeddings(self, documents: List[Document]) -> List[List[float]]:
        """为文档生成嵌入向量"""
        texts = [doc.page_content for doc in documents]
        
        # 使用异步方式生成嵌入向量
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self.embeddings.embed_documents, texts
        )
        
        return embeddings
    
    def get_document_by_id(self, documents: List[Document], chunk_id: str) -> Optional[Document]:
        """根据chunk_id获取文档"""
        for doc in documents:
            if doc.metadata.get('chunk_id') == chunk_id:
                return doc
        return None
    
    def filter_documents_by_score(self, documents: List[Document], scores: List[float], 
                                 threshold: float) -> List[Document]:
        """根据相似度分数过滤文档"""
        filtered_docs = []
        for doc, score in zip(documents, scores):
            if score >= threshold:
                filtered_docs.append(doc)
        return filtered_docs

# config_manager/rag.py
from pydantic import Field, field_validator
from typing import Dict, ClassVar, List, Optional, Literal
from .i18n import I18nMixin, Description


class RAGConfig(I18nMixin):
    """RAG (Retrieval-Augmented Generation) configuration settings."""

    # 是否启用RAG功能
    enabled: bool = Field(default=False, alias="enabled")
    
    # 知识库配置
    knowledge_base_path: str = Field(default="./knowledge_base", alias="knowledge_base_path")
    
    # 向量数据库配置
    vector_store_type: Literal["chromadb", "faiss", "memory"] = Field(
        default="chromadb", alias="vector_store_type"
    )
    
    # 嵌入模型配置
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="embedding_model")
    embedding_device: str = Field(default="cpu", alias="embedding_device")
    
    # 检索配置
    retrieval_top_k: int = Field(default=5, alias="retrieval_top_k")
    retrieval_score_threshold: float = Field(default=0.7, alias="retrieval_score_threshold")
    
    # 文档处理配置
    chunk_size: int = Field(default=1000, alias="chunk_size")
    chunk_overlap: int = Field(default=200, alias="chunk_overlap")
    
    # 支持的文档类型
    supported_file_types: List[str] = Field(
        default=[".txt", ".md", ".pdf", ".docx", ".html"], 
        alias="supported_file_types"
    )
    
    # 自动文档更新
    auto_update: bool = Field(default=True, alias="auto_update")
    update_interval: int = Field(default=3600, alias="update_interval")  # 秒
    
    # 上下文增强配置
    context_window_size: int = Field(default=4000, alias="context_window_size")
    max_context_documents: int = Field(default=3, alias="max_context_documents")
    
    # ChromaDB特定配置
    chromadb_config: Dict[str, str] = Field(
        default_factory=lambda: {
            "persist_directory": "./chroma_db",
            "collection_name": "knowledge_base"
        },
        alias="chromadb_config"
    )
    
    # FAISS特定配置
    faiss_config: Dict[str, str] = Field(
        default_factory=lambda: {
            "index_path": "./faiss_index",
            "index_type": "IndexFlatIP"
        },
        alias="faiss_config"
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "enabled": Description(
            en="Enable RAG functionality", zh="启用RAG功能"
        ),
        "knowledge_base_path": Description(
            en="Path to the knowledge base directory", zh="知识库目录路径"
        ),
        "vector_store_type": Description(
            en="Type of vector store to use", zh="使用的向量存储类型"
        ),
        "embedding_model": Description(
            en="Embedding model for text vectorization", zh="文本向量化嵌入模型"
        ),
        "embedding_device": Description(
            en="Device to run embedding model on", zh="运行嵌入模型的设备"
        ),
        "retrieval_top_k": Description(
            en="Number of top documents to retrieve", zh="检索的文档数量"
        ),
        "retrieval_score_threshold": Description(
            en="Minimum similarity score for retrieval", zh="检索的最小相似度分数"
        ),
        "chunk_size": Description(
            en="Size of text chunks for processing", zh="文本块处理大小"
        ),
        "chunk_overlap": Description(
            en="Overlap between text chunks", zh="文本块之间的重叠"
        ),
        "supported_file_types": Description(
            en="Supported file types for document processing", zh="支持的文档处理文件类型"
        ),
        "auto_update": Description(
            en="Automatically update knowledge base", zh="自动更新知识库"
        ),
        "update_interval": Description(
            en="Interval for automatic updates in seconds", zh="自动更新间隔（秒）"
        ),
        "context_window_size": Description(
            en="Maximum context window size for LLM", zh="LLM的最大上下文窗口大小"
        ),
        "max_context_documents": Description(
            en="Maximum number of context documents to include", zh="包含的最大上下文文档数量"
        ),
        "chromadb_config": Description(
            en="ChromaDB specific configuration", zh="ChromaDB特定配置"
        ),
        "faiss_config": Description(
            en="FAISS specific configuration", zh="FAISS特定配置"
        ),
    }

    @field_validator("retrieval_top_k")
    def validate_retrieval_top_k(cls, v):
        if v < 1:
            raise ValueError("retrieval_top_k must be at least 1")
        return v

    @field_validator("retrieval_score_threshold")
    def validate_retrieval_score_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("retrieval_score_threshold must be between 0 and 1")
        return v

    @field_validator("chunk_size")
    def validate_chunk_size(cls, v):
        if v < 100:
            raise ValueError("chunk_size must be at least 100")
        return v

    @field_validator("chunk_overlap")
    def validate_chunk_overlap(cls, v):
        if v < 0:
            raise ValueError("chunk_overlap must be non-negative")
        return v

    @field_validator("update_interval")
    def validate_update_interval(cls, v):
        if v < 60:
            raise ValueError("update_interval must be at least 60 seconds")
        return v

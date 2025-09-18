# rag_api.py
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from loguru import logger

from .rag import RAGEngine
from .config_manager.rag import RAGConfig
from .service_context import ServiceContext

router = APIRouter(prefix="/api/rag", tags=["RAG"])


async def get_rag_engine_from_context(service_context: ServiceContext) -> Optional[RAGEngine]:
    """从服务上下文中获取RAG引擎"""
    if hasattr(service_context, 'agent_engine') and service_context.agent_engine:
        if hasattr(service_context.agent_engine, 'rag_engine'):
            return service_context.agent_engine.rag_engine
    return None


@router.get("/status")
async def get_rag_status(service_context: ServiceContext = Depends()):
    """获取RAG状态"""
    try:
        rag_engine = await get_rag_engine_from_context(service_context)
        if not rag_engine:
            return {
                "enabled": False,
                "message": "RAG engine not available"
            }
        
        stats = await rag_engine.get_stats()
        health = await rag_engine.health_check()
        
        return {
            "enabled": True,
            "stats": stats,
            "health": health
        }
    except Exception as e:
        logger.error(f"Error getting RAG status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initialize")
async def initialize_rag(service_context: ServiceContext = Depends()):
    """初始化RAG引擎"""
    try:
        rag_engine = await get_rag_engine_from_context(service_context)
        if not rag_engine:
            raise HTTPException(status_code=400, detail="RAG engine not available")
        
        success = await rag_engine.initialize()
        if success:
            return {"message": "RAG engine initialized successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to initialize RAG engine")
    except Exception as e:
        logger.error(f"Error initializing RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update")
async def update_knowledge_base(service_context: ServiceContext = Depends()):
    """更新知识库"""
    try:
        rag_engine = await get_rag_engine_from_context(service_context)
        if not rag_engine:
            raise HTTPException(status_code=400, detail="RAG engine not available")
        
        success = await rag_engine.update_knowledge_base()
        if success:
            return {"message": "Knowledge base updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update knowledge base")
    except Exception as e:
        logger.error(f"Error updating knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_knowledge_base(
    query: str,
    top_k: int = 5,
    service_context: ServiceContext = Depends()
):
    """搜索知识库"""
    try:
        rag_engine = await get_rag_engine_from_context(service_context)
        if not rag_engine:
            raise HTTPException(status_code=400, detail="RAG engine not available")
        
        results = await rag_engine.search(query, top_k)
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add-document")
async def add_document(
    file: UploadFile = File(...),
    service_context: ServiceContext = Depends()
):
    """添加文档到知识库"""
    try:
        rag_engine = await get_rag_engine_from_context(service_context)
        if not rag_engine:
            raise HTTPException(status_code=400, detail="RAG engine not available")
        
        # 保存上传的文件
        upload_dir = Path("temp_uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 添加到知识库
        success = await rag_engine.add_document(str(file_path))
        
        # 清理临时文件
        file_path.unlink()
        
        if success:
            return {"message": f"Document {file.filename} added successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to add document")
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents(service_context: ServiceContext = Depends()):
    """列出知识库中的文档"""
    try:
        rag_engine = await get_rag_engine_from_context(service_context)
        if not rag_engine:
            raise HTTPException(status_code=400, detail="RAG engine not available")
        
        # 这里简化实现，实际应用中需要从向量存储中获取文档列表
        knowledge_path = Path(rag_engine.config.knowledge_base_path)
        if not knowledge_path.exists():
            return {"documents": []}
        
        documents = []
        for file_path in knowledge_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in rag_engine.config.supported_file_types:
                documents.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime
                })
        
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_name}")
async def remove_document(
    document_name: str,
    service_context: ServiceContext = Depends()
):
    """从知识库中移除文档"""
    try:
        rag_engine = await get_rag_engine_from_context(service_context)
        if not rag_engine:
            raise HTTPException(status_code=400, detail="RAG engine not available")
        
        # 查找文档
        knowledge_path = Path(rag_engine.config.knowledge_base_path)
        document_path = knowledge_path / document_name
        
        if not document_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")
        
        # 从知识库中移除
        success = await rag_engine.remove_document(str(document_path))
        
        if success:
            return {"message": f"Document {document_name} removed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to remove document")
    except Exception as e:
        logger.error(f"Error removing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_rag_config(service_context: ServiceContext = Depends()):
    """获取RAG配置"""
    try:
        rag_config = service_context.character_config.rag_config
        return rag_config.model_dump()
    except Exception as e:
        logger.error(f"Error getting RAG config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test")
async def test_rag(
    query: str,
    service_context: ServiceContext = Depends()
):
    """测试RAG功能"""
    try:
        rag_engine = await get_rag_engine_from_context(service_context)
        if not rag_engine:
            raise HTTPException(status_code=400, detail="RAG engine not available")
        
        # 搜索相关文档
        search_results = await rag_engine.search(query, top_k=3)
        
        # 获取上下文
        context = await rag_engine.get_context(query)
        
        return {
            "query": query,
            "search_results": search_results,
            "context": context,
            "context_length": len(context)
        }
    except Exception as e:
        logger.error(f"Error testing RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))

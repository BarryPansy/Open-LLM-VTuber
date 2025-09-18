#!/usr/bin/env python3
"""
RAG功能测试脚本
用于验证RAG引擎的基本功能
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from open_llm_vtuber.config_manager.rag import RAGConfig
from open_llm_vtuber.rag import RAGEngine


async def test_rag_engine():
    """测试RAG引擎基本功能"""
    print("🚀 开始测试RAG引擎...")
    
    # 创建RAG配置
    config = RAGConfig(
        enabled=True,
        knowledge_base_path="./knowledge_base",
        vector_store_type="memory",  # 使用内存存储进行测试
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        retrieval_top_k=3,
        retrieval_score_threshold=0.5
    )
    
    # 创建RAG引擎
    rag_engine = RAGEngine(config)
    
    try:
        # 初始化RAG引擎
        print("📚 初始化RAG引擎...")
        success = await rag_engine.initialize()
        
        if not success:
            print("❌ RAG引擎初始化失败")
            return False
        
        print("✅ RAG引擎初始化成功")
        
        # 测试搜索功能
        print("\n🔍 测试搜索功能...")
        test_queries = [
            "Open-LLM-VTuber是什么？",
            "支持哪些文档格式？",
            "如何配置语音识别？",
            "RAG功能有什么特点？"
        ]
        
        for query in test_queries:
            print(f"\n查询: {query}")
            results = await rag_engine.search(query, top_k=2)
            
            if results:
                print(f"找到 {len(results)} 个相关文档:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. 相似度: {result['score']:.3f}")
                    print(f"     内容: {result['content'][:100]}...")
            else:
                print("  未找到相关文档")
        
        # 测试上下文生成
        print("\n📝 测试上下文生成...")
        context = await rag_engine.get_context("如何开始使用这个项目？")
        if context:
            print(f"生成的上下文长度: {len(context)} 字符")
            print(f"上下文预览: {context[:200]}...")
        else:
            print("未能生成上下文")
        
        # 获取统计信息
        print("\n📊 获取统计信息...")
        stats = await rag_engine.get_stats()
        print(f"统计信息: {stats}")
        
        # 健康检查
        print("\n🏥 健康检查...")
        health = await rag_engine.health_check()
        print(f"健康状态: {health}")
        
        print("\n✅ RAG引擎测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_document_processing():
    """测试文档处理功能"""
    print("\n📄 测试文档处理功能...")
    
    from open_llm_vtuber.rag.document_processor import DocumentProcessor
    
    config = RAGConfig(
        chunk_size=500,
        chunk_overlap=100,
        supported_file_types=[".txt", ".md"]
    )
    
    processor = DocumentProcessor(config)
    
    # 测试处理示例文档
    knowledge_path = Path("./knowledge_base")
    if knowledge_path.exists():
        documents = await processor.process_documents(str(knowledge_path))
        print(f"处理了 {len(documents)} 个文档块")
        
        if documents:
            print("第一个文档块示例:")
            doc = documents[0]
            print(f"  内容: {doc.page_content[:200]}...")
            print(f"  元数据: {doc.metadata}")
    else:
        print("知识库目录不存在，跳过文档处理测试")


async def main():
    """主测试函数"""
    print("🧪 RAG功能测试开始")
    print("=" * 50)
    
    # 测试文档处理
    await test_document_processing()
    
    # 测试RAG引擎
    success = await test_rag_engine()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 所有测试通过！RAG功能正常工作。")
    else:
        print("💥 测试失败，请检查配置和依赖。")
    
    return success


if __name__ == "__main__":
    # 运行测试
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

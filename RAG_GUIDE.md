# RAG功能使用指南

## 概述

RAG（Retrieval-Augmented Generation，检索增强生成）功能为Open-LLM-VTuber添加了知识库检索能力，让AI能够基于特定文档内容进行更准确和相关的回答。

## 功能特性

- **多格式文档支持**：支持TXT、MD、PDF、DOCX、HTML等格式
- **智能文档分割**：自动将长文档分割成合适的块进行处理
- **向量检索**：使用语义相似度检索最相关的文档内容
- **多种向量存储**：支持ChromaDB、FAISS、内存存储
- **自动更新**：支持知识库的自动更新和增量添加
- **RESTful API**：提供完整的API接口进行管理

## 快速开始

### 1. 启用RAG功能

在 `conf.yaml` 中配置RAG：

```yaml
character_config:
  # 选择RAG Agent
  agent_config:
    conversation_agent_choice: 'rag_agent'
    agent_settings:
      rag_agent:
        llm_provider: 'ollama_llm'  # 选择你的LLM提供商
        faster_first_response: true
        segment_method: 'pysbd'
  
  # RAG配置
  rag_config:
    enabled: true
    knowledge_base_path: './knowledge_base'
    vector_store_type: 'chromadb'
    embedding_model: 'sentence-transformers/all-MiniLM-L6-v2'
    retrieval_top_k: 5
    retrieval_score_threshold: 0.7
    chunk_size: 1000
    chunk_overlap: 200
    auto_update: true
    update_interval: 3600
```

### 2. 准备知识库

将你的文档放入 `knowledge_base` 目录：

```
knowledge_base/
├── README.md
├── project_docs.txt
├── user_manual.pdf
└── faq.md
```

### 3. 启动服务器

```bash
uv run run_server.py
```

### 4. 初始化RAG引擎

访问 `http://localhost:12393/api/rag/initialize` 或使用API：

```bash
curl -X POST http://localhost:12393/api/rag/initialize
```

## API接口

### 获取RAG状态

```bash
GET /api/rag/status
```

### 搜索知识库

```bash
POST /api/rag/search
Content-Type: application/json

{
  "query": "如何配置语音识别？",
  "top_k": 5
}
```

### 添加文档

```bash
POST /api/rag/add-document
Content-Type: multipart/form-data

file: [上传的文档文件]
```

### 列出文档

```bash
GET /api/rag/documents
```

### 更新知识库

```bash
POST /api/rag/update
```

## 配置详解

### 向量存储选项

1. **ChromaDB**（推荐）
   - 持久化存储
   - 支持元数据过滤
   - 易于管理和查询

2. **FAISS**
   - 高性能检索
   - 支持多种索引类型
   - 适合大规模数据

3. **Memory**
   - 内存存储
   - 适合测试和小规模使用

### 嵌入模型选择

- `sentence-transformers/all-MiniLM-L6-v2`：轻量级，适合快速部署
- `sentence-transformers/all-mpnet-base-v2`：更高质量，适合生产环境
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`：多语言支持

### 检索参数调优

- `retrieval_top_k`：检索文档数量，建议3-10
- `retrieval_score_threshold`：相似度阈值，建议0.6-0.8
- `chunk_size`：文档块大小，建议500-1500
- `chunk_overlap`：块重叠大小，建议100-300

## 使用技巧

### 1. 文档准备

- 确保文档内容清晰、结构化
- 使用标题和段落来组织内容
- 避免过长的单个文档

### 2. 查询优化

- 使用具体、明确的问题
- 包含关键词和上下文
- 避免过于宽泛的查询

### 3. 性能优化

- 定期更新知识库
- 监控检索质量
- 根据使用情况调整参数

## 故障排除

### 常见问题

1. **RAG引擎初始化失败**
   - 检查知识库路径是否存在
   - 确认文档格式是否支持
   - 查看日志获取详细错误信息

2. **检索结果不准确**
   - 调整相似度阈值
   - 检查文档内容质量
   - 尝试不同的嵌入模型

3. **性能问题**
   - 减少检索文档数量
   - 使用更快的嵌入模型
   - 考虑使用FAISS存储

### 日志查看

```bash
# 查看详细日志
uv run run_server.py --verbose
```

## 高级功能

### 自定义嵌入模型

```yaml
rag_config:
  embedding_model: 'your-custom-model'
  embedding_device: 'cuda'  # 使用GPU加速
```

### 元数据过滤

在文档中添加元数据：

```python
# 在文档处理时添加自定义元数据
doc.metadata.update({
    'category': 'technical',
    'priority': 'high',
    'language': 'zh'
})
```

### 增量更新

```python
# 通过API添加单个文档
await rag_engine.add_document('/path/to/new/document.pdf')
```

## 最佳实践

1. **文档组织**：按主题和类型组织文档
2. **定期更新**：设置自动更新机制
3. **质量监控**：定期检查检索质量
4. **备份数据**：定期备份向量数据库
5. **性能监控**：监控检索延迟和准确性

## 示例场景

### 技术文档问答

```yaml
# 配置用于技术文档问答
rag_config:
  knowledge_base_path: './tech_docs'
  retrieval_top_k: 3
  retrieval_score_threshold: 0.8
```

### 客服知识库

```yaml
# 配置用于客服知识库
rag_config:
  knowledge_base_path: './customer_service'
  retrieval_top_k: 5
  retrieval_score_threshold: 0.7
```

### 多语言支持

```yaml
# 使用多语言嵌入模型
rag_config:
  embedding_model: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
```

通过合理配置和使用RAG功能，可以让你的VTuber AI具备更专业、更准确的知识问答能力！

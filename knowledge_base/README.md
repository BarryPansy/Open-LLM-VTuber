# 知识库目录

这个目录用于存储RAG（检索增强生成）功能的知识库文档。

## 支持的文档格式

- `.txt` - 纯文本文件
- `.md` - Markdown文件
- `.pdf` - PDF文档
- `.docx` - Word文档
- `.html` - HTML文件

## 使用方法

1. 将你的文档放入此目录
2. 确保RAG功能在配置中已启用
3. 重启服务器或使用API更新知识库

## 文档处理

- 文档会被自动分割成小块进行处理
- 每个文档块会生成向量嵌入
- 系统会根据用户查询检索最相关的文档块

## 配置说明

在 `conf.yaml` 中配置RAG相关设置：

```yaml
character_config:
  rag_config:
    enabled: true
    knowledge_base_path: './knowledge_base'
    vector_store_type: 'chromadb'
    embedding_model: 'sentence-transformers/all-MiniLM-L6-v2'
    retrieval_top_k: 5
    retrieval_score_threshold: 0.7
```

## API接口

- `GET /api/rag/status` - 获取RAG状态
- `POST /api/rag/initialize` - 初始化RAG引擎
- `POST /api/rag/update` - 更新知识库
- `POST /api/rag/search` - 搜索知识库
- `POST /api/rag/add-document` - 添加文档
- `GET /api/rag/documents` - 列出文档
- `DELETE /api/rag/documents/{name}` - 删除文档

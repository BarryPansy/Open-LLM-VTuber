# RAG功能安装和使用指南

## 安装步骤

### 1. 安装依赖包

RAG功能需要额外的Python包，运行以下命令安装：

```bash
# 使用uv安装（推荐）
uv sync

# 或者使用pip安装
pip install langchain langchain-community chromadb sentence-transformers faiss-cpu tiktoken
```

### 2. 配置RAG功能

编辑 `conf.yaml` 文件，启用RAG功能：

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
```

### 3. 准备知识库

将你的文档放入 `knowledge_base` 目录：

```bash
mkdir -p knowledge_base
# 将你的文档复制到knowledge_base目录
cp your_documents/* knowledge_base/
```

### 4. 启动服务器

```bash
uv run run_server.py
```

### 5. 初始化RAG引擎

访问以下URL初始化RAG引擎：

```
http://localhost:12393/api/rag/initialize
```

或者使用curl：

```bash
curl -X POST http://localhost:12393/api/rag/initialize
```

## 使用方法

### 通过WebSocket对话

启动服务器后，通过WebSocket与AI对话，AI会自动使用RAG功能检索相关知识。

### 通过API管理

#### 搜索知识库

```bash
curl -X POST http://localhost:12393/api/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "如何配置语音识别？", "top_k": 5}'
```

#### 添加文档

```bash
curl -X POST http://localhost:12393/api/rag/add-document \
  -F "file=@your_document.pdf"
```

#### 查看RAG状态

```bash
curl http://localhost:12393/api/rag/status
```

## 测试RAG功能

运行测试脚本验证RAG功能：

```bash
python test_rag.py
```

## 故障排除

### 常见问题

1. **依赖安装失败**
   ```bash
   # 尝试使用conda安装
   conda install -c conda-forge chromadb sentence-transformers
   
   # 或者使用pip安装特定版本
   pip install chromadb==0.5.0 sentence-transformers==3.0.0
   ```

2. **嵌入模型下载失败**
   - 检查网络连接
   - 尝试使用代理
   - 手动下载模型到本地

3. **ChromaDB初始化失败**
   - 检查目录权限
   - 确保有足够的磁盘空间
   - 尝试使用内存存储进行测试

4. **检索结果不准确**
   - 调整相似度阈值
   - 检查文档内容质量
   - 尝试不同的嵌入模型

### 日志查看

```bash
# 查看详细日志
uv run run_server.py --verbose

# 查看特定日志
tail -f logs/debug_*.log
```

## 性能优化

### 1. 使用GPU加速

```yaml
rag_config:
  embedding_device: 'cuda'  # 使用GPU
```

### 2. 调整检索参数

```yaml
rag_config:
  retrieval_top_k: 3  # 减少检索数量
  chunk_size: 800     # 调整块大小
```

### 3. 使用更快的向量存储

```yaml
rag_config:
  vector_store_type: 'faiss'  # 使用FAISS
```

## 高级配置

### 自定义嵌入模型

```yaml
rag_config:
  embedding_model: 'sentence-transformers/all-mpnet-base-v2'
```

### 多语言支持

```yaml
rag_config:
  embedding_model: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
```

### 自定义文档处理

```yaml
rag_config:
  chunk_size: 1500
  chunk_overlap: 300
  supported_file_types: ['.txt', '.md', '.pdf', '.docx', '.html', '.json']
```

## 监控和维护

### 定期更新知识库

```bash
# 手动更新
curl -X POST http://localhost:12393/api/rag/update

# 设置自动更新（在配置中）
rag_config:
  auto_update: true
  update_interval: 3600  # 每小时更新一次
```

### 监控RAG状态

```bash
# 查看状态
curl http://localhost:12393/api/rag/status

# 查看健康状态
curl http://localhost:12393/api/rag/health
```

### 备份向量数据库

```bash
# 备份ChromaDB
cp -r chroma_db chroma_db_backup

# 备份FAISS索引
cp -r faiss_index faiss_index_backup
```

通过以上步骤，你就可以成功安装和使用RAG功能了！如果遇到问题，请查看日志文件或参考故障排除部分。

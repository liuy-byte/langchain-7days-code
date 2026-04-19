# LangChain 7 天系列教程

基于 `langchain==1.2.15` / `langchain-core==1.3.0` 的代码示例。

## 环境安装

```bash
uv sync
```

## 文件结构

```
langchain-7days/
├── main.py              # 模块导入入口
├── pyproject.toml       # 依赖配置
├── api/
│   ├── main.py              # RAG 问答机器人 CLI 入口
│   ├── day1_components.py   # Day 1: 四大组件
│   ├── day2_model_io.py      # Day 2: Model I/O
│   ├── day3_retrieval.py     # Day 3: Retrieval
│   ├── day4_rag.py           # Day 4: RAG 全链路
│   ├── day5_agent.py         # Day 5: Agent + Tools
│   ├── day6_memory_chain.py  # Day 6: Memory + LCEL
│   └── day7_review.py        # Day 7: 全景回顾
├── core/                # 核心工具
│   ├── config.py            # 配置管理
│   ├── document_loader.py   # 文档加载
│   ├── embedding.py          # Embedding 模型
│   ├── vectorstore.py        # 向量库
│   ├── retriever.py          # 检索器
│   ├── memory.py             # 聊天记忆
│   ├── search_tool.py        # 联网搜索
│   └── rag_chain.py          # RAG Chain 组装
├── tests/
├── config.json           # 配置文件
└── README.md
```

---

## 实战项目：RAG 问答机器人

一个完整的 RAG 问答机器人，串联 LangChain 7 天系列的所有核心概念。

### 功能特性

- **文档问答** — 支持 PDF / MD / TXT 文件，导入即搜
- **聊天记忆** — 多轮对话，自动记住上下文
- **联网搜索** — 配置 Tavily API Key，实时获取最新信息
- **三种模式** — local（本地）/ api（硅基流动）/ prod（OpenAI）

### 环境准备

```bash
# 配置 API Key（环境变量）
export OPENAI_API_KEY=your-api-key
```

### 快速启动

```bash
cd langchain-7days
python api/main.py
```

### 使用命令

```
!add ./docs/guide.pdf    # 添加文档到向量库
!reset                   # 清空聊天记忆
!mode local              # 切换模式 (local/api/prod)
!help                    # 显示帮助
!quit                    # 退出
```

### 模式说明

| 模式 | LLM | 向量库 | 搜索 | 适用场景 |
|------|-----|--------|------|----------|
| `local` | Ollama (本地) | ChromaDB | 无 | 学习/离线 |
| `api` | SiliconFlow | ChromaDB | Tavily | 国内开发者 |
| `prod` | OpenAI/Anthropic | Pinecone | Tavily | 正式使用 |

---

## 环境变量

```bash
export OPENAI_API_KEY=your-siliconflow-key
export OPENAI_BASE_URL=https://api.siliconflow.cn/v1
export OPENAI_MODEL=Pro/deepseek-ai/DeepSeek-V3.2
```

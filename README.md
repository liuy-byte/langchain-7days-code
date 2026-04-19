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
│   ├── day1_components.py   # Day 1: 四大组件
│   ├── day2_model_io.py    # Day 2: Model I/O
│   ├── day3_retrieval.py    # Day 3: Retrieval
│   ├── day4_rag.py          # Day 4: RAG 全链路
│   ├── day5_agent.py        # Day 5: Agent + Tools
│   ├── day6_memory_chain.py # Day 6: Memory + LCEL
│   └── day7_review.py       # Day 7: 全景回顾
├── core/                # 核心工具
└── tests/               # 测试用例
```

## 快速开始

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

llm = ChatOpenAI(
    model="Pro/deepseek-ai/DeepSeek-V3.2",
    base_url="https://api.siliconflow.cn/v1",
    api_key="your-api-key"
)
agent = create_agent(llm, tools=[], system_prompt="你是一个助手")
```

## 环境变量

```bash
export OPENAI_API_KEY=your-siliconflow-key
export OPENAI_BASE_URL=https://api.siliconflow.cn/v1
export OPENAI_MODEL=Pro/deepseek-ai/DeepSeek-V3.2
```

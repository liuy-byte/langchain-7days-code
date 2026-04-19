"""
LangChain 7 天系列教程代码示例

文件结构：
- main.py        : 各模块导入入口
- api/           : 各章节 API 示例
- core/          : 核心工具函数
"""

# ========== Day 1: 四大组件 ==========
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.agents import create_agent

# ========== Day 2: Model I/O ==========
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel

# ========== Day 3: Retrieval ==========
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ========== Day 4: RAG ==========
from langchain_core.runnables import RunnablePassthrough

# ========== Day 5: Agent ==========
# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_core.tools import tool

# ========== Day 6: Memory & LCEL ==========
from langchain_classic.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_core.runnables import RunnableParallel, RunnableLambda

# ========== Day 7: 全景回顾 ==========
# 整合以上所有模块

__all__ = [
    "ChatOpenAI",
    "ChatPromptTemplate",
    "StrOutputParser",
    "JsonOutputParser",
    "PydanticOutputParser",
    "create_agent",
    "tool",
    "RunnablePassthrough",
    "RunnableParallel",
    "RunnableLambda",
    "ConversationBufferMemory",
    "ConversationSummaryBufferMemory",
    "PyPDFLoader",
    "RecursiveCharacterTextSplitter",
    "OpenAIEmbeddings",
    "Chroma",
]

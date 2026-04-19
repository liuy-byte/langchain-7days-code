#!/usr/bin/env python3
"""LangChain 7 天实战项目 - RAG 问答机器人 CLI 入口"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    BotConfig,
    load_config,
    load_document,
    VectorStoreManager,
    RetrieverManager,
    ChatMemoryManager,
    SearchTool,
    create_rag_chain,
    chat,
)


class RAGBot:
    """RAG 问答机器人"""

    def __init__(self, config_path: str | None = None):
        self.config = load_config(config_path)
        self.vectorstore = VectorStoreManager(
            config=self.config.vectorstore,
            embedding_config=self.config.embedding,
        )
        self.retriever = RetrieverManager(self.vectorstore, top_k=4)
        self.memory = ChatMemoryManager(self.config.memory)
        self.search_tool = SearchTool(self.config.search)
        self.chain = create_rag_chain(
            config=self.config,
            retriever=self.retriever,
            memory=self.memory,
            search_tool=self.search_tool,
        )
        self._doc_loaded = False

    def add_document(self, path: str) -> str:
        """添加文档到向量库"""
        try:
            docs = load_document(path)
            self.vectorstore.add_documents(docs)
            self.vectorstore.save()
            self._doc_loaded = True
            return f"✅ 成功加载 {len(docs)} 个文档片段"
        except Exception as e:
            return f"❌ 加载失败: {e}"

    def chat(self, query: str) -> str:
        """对话"""
        return chat(self.chain, self.memory, query)

    def reset_memory(self) -> str:
        """清空记忆"""
        self.memory.clear()
        return "✅ 记忆已清空"

    def switch_mode(self, mode: str) -> str:
        """切换模式"""
        if mode not in ("local", "api", "prod"):
            return f"❌ 不支持的模式: {mode}，可用: local / api / prod"
        self.config.mode = mode
        self.chain = create_rag_chain(
            config=self.config,
            retriever=self.retriever,
            memory=self.memory,
            search_tool=self.search_tool,
        )
        return f"✅ 已切换到 {mode} 模式"


def print_intro():
    print("=" * 60)
    print("  LangChain 7 天实战 — RAG 问答机器人")
    print("=" * 60)
    print()


def print_help():
    print("\n📖 命令说明：")
    print("  !add <path>    — 添加文档到向量库")
    print("  !reset        — 清空聊天记忆")
    print("  !mode <mode>   — 切换模式 (local / api / prod)")
    print("  !help         — 显示帮助")
    print("  !quit         — 退出")
    print()


def main():
    print_intro()

    config_path = os.environ.get("RAG_BOT_CONFIG")
    bot = RAGBot(config_path)

    print(f"📋 当前模式: {bot.config.mode}")
    print(f"🔍 联网搜索: {'已启用' if bot.search_tool.is_enabled else '未启用 (未配置 TAVILY_API_KEY)'}")
    print()
    print_help()

    print("💬 开始对话（直接输入问题，按回车发送）：\n")

    while True:
        try:
            user_input = input("👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 再见！")
            break

        if not user_input:
            continue

        if user_input.startswith("!add "):
            path = user_input[5:].strip()
            print(f"\n📄 正在加载文档: {path}")
            result = bot.add_document(path)
            print(f"\n{result}\n")
        elif user_input == "!reset":
            result = bot.reset_memory()
            print(f"\n{result}\n")
        elif user_input.startswith("!mode "):
            mode = user_input[6:].strip()
            result = bot.switch_mode(mode)
            print(f"\n{result}\n")
        elif user_input == "!help":
            print_help()
        elif user_input == "!quit":
            print("\n👋 再见！")
            break
        else:
            print("\n🤖 AI: ", end="", flush=True)
            try:
                response = bot.chat(user_input)
                print(f"{response}\n")
            except Exception as e:
                print(f"\n❌ 错误: {e}\n")


if __name__ == "__main__":
    main()

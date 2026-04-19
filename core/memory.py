"""聊天记忆模块"""

from typing import Any

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory

from core.config import MemoryConfig


class ChatMemoryManager:
    """聊天记忆管理器"""

    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()
        self._history: ChatMessageHistory = ChatMessageHistory()

    def add_user_message(self, message: str) -> None:
        """添加用户消息"""
        self._history.add_user_message(message)

    def add_ai_message(self, message: str) -> None:
        """添加 AI 消息"""
        self._history.add_ai_message(message)

    def get_messages(self) -> list[BaseMessage]:
        """获取所有消息"""
        return self._history.messages

    def clear(self) -> None:
        """清空记忆"""
        self._history.clear()

    def load_memory_variables(self) -> dict[str, Any]:
        """输出与 LCEL 链兼容的 memory 变量"""
        messages = self._history.messages
        if self.config.max_history > 0 and len(messages) > self.config.max_history:
            messages = messages[-self.config.max_history :]
        return {"chat_history": messages}

    def as_runnable(self):
        """转换为 LCEL 可运行对象"""
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(lambda _: self.load_memory_variables())

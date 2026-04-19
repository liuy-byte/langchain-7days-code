"""联网搜索工具模块"""

from typing import Any

from langchain_core.tools import BaseTool, tool
from langchain_tavily import TavilySearch

from core.config import SearchConfig


@tool
def web_search(query: str) -> str:
    """根据查询关键词搜索互联网，返回最新相关信息。
    用于回答时效性问题或需要最新数据的问题。
    """
    return f"[联网搜索结果]\n查询: {query}\n注意: 请在 Tavily API 配置后获取真实搜索结果"


class SearchTool:
    """联网搜索工具封装"""

    def __init__(self, config: SearchConfig):
        self.config = config
        self._tavily: TavilySearch | None = None
        self._setup()

    def _setup(self) -> None:
        if self.config.provider == "tavily" and self.config.api_key:
            self._tavily = TavilySearch(api_key=self.config.api_key)

    def search(self, query: str) -> str:
        """执行搜索，返回格式化结果"""
        if self._tavily:
            results = self._tavily.invoke(query)
            if not results:
                return "（未找到相关搜索结果）"
            parts = []
            for r in results[:3]:
                parts.append(f"- {r.get('title', '未知标题')}: {r.get('url', '')}\n  {r.get('content', '')[:200]}")
            return "\n".join(parts)
        return web_search.invoke(query)

    def as_tool(self) -> BaseTool:
        """转换为 LangChain Tool"""
        if self._tavily:
            return self._tavily
        return web_search

    @property
    def is_enabled(self) -> bool:
        """是否启用联网搜索"""
        return self._tavily is not None

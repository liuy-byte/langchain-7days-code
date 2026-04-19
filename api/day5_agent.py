"""
Day 5: Agent + Tools - create_agent / @tool / Tool Calling

本文件展示 LangChain Agent 和 Tools 的用法。
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

# 内置工具支持
try:
    from langchain_community.tools import DuckDuckGoSearchRun
except ImportError:
    DuckDuckGoSearchRun = None

try:
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.tools.wikipedia.tool import WikipediaAPIWrapper
except ImportError:
    WikipediaQueryRun = None
    WikipediaAPIWrapper = None

_api_key = os.environ.get("OPENAI_API_KEY", "")
_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
_default_model = os.environ.get("OPENAI_MODEL", "Pro/MiniMaxAI/MiniMax-M2.5")


def _has_api_key():
    return bool(_api_key and _api_key != "your-api-key")


def _create_llm():
    return ChatOpenAI(
        model=_default_model,
        base_url=_base_url,
        api_key=_api_key,
        temperature=0
    )


def demo_custom_tool():
    """自定义 Tool：@tool 装饰器"""
    print("【自定义 Tool 演示】")

    @tool
    def get_weather(city: str) -> str:
        """查询城市天气，传入中文城市名如'北京'"""
        weather_data = {"北京": "晴，25°C", "上海": "多云，28°C", "深圳": "雷阵雨，30°C"}
        return weather_data.get(city, f"未找到 {city} 的天气数据")

    @tool
    def calculator(expression: str) -> str:
        """执行数学计算，传入表达式如'(35+17)*2'"""
        return str(eval(expression))

    print("  @tool 装饰器定义了两个工具:")
    print(f"    - get_weather: {get_weather.description}")
    print(f"    - calculator: {calculator.description}")
    return [get_weather, calculator]


def demo_create_agent():
    """创建 Agent（基于 LangGraph）"""
    print("\n【创建 Agent 演示】")

    tools = demo_custom_tool()

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        return

    llm = _create_llm()
    agent = create_agent(
        llm,
        tools=tools,
        system_prompt="你是一个智能助手，可以调用工具来回答问题。"
    )
    result = agent.invoke({
        "messages": [{"role": "user", "content": "北京今天天气怎么样？"}]
    })
    print(f"  Agent 响应: {result['messages'][-1].content}")


def demo_tool_calling():
    """Tool Calling：GPT-4 的结构化工具调用"""
    print("\n【Tool Calling 演示】")

    @tool
    def get_weather(city: str) -> str:
        """查询城市天气"""
        return {"北京": "晴25°C", "上海": "多云28°C"}.get(city, "未知")

    @tool
    def get_news(topic: str) -> str:
        """搜索最新新闻"""
        return f"{topic}的最新消息：..."

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        return

    llm = _create_llm()
    llm_with_tools = llm.bind_tools([get_weather, get_news])
    messages = [{"role": "user", "content": "北京今天热吗？有什么科技新闻？"}]
    response = llm_with_tools.invoke(messages)
    print(f"  工具调用: {response.tool_calls}")


def demo_builtin_tools():
    """内置工具：DuckDuckGoSearch / Wikipedia"""
    print("\n【内置工具演示】")

    print("  1. DuckDuckGoSearchRun (网络搜索):")
    if DuckDuckGoSearchRun:
        print("     from langchain_community.tools import DuckDuckGoSearchRun")
        print("     search = DuckDuckGoSearchRun()")
        print("     result = search.run('LangChain 教程')")
    else:
        print("     ⚠️  DuckDuckGoSearchRun 未安装 (pip install langchain-community)")

    print("  2. WikipediaQueryRun (维基百科查询):")
    if WikipediaQueryRun:
        print("     from langchain_community.tools import WikipediaQueryRun")
        print("     wiki = WikipediaQueryRun()")
        print("     result = wiki.run('Python 编程语言')")
    else:
        print("     ⚠️  WikipediaQueryRun 未安装 (pip install langchain-community)")

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际工具调用")
        return

    if DuckDuckGoSearchRun:
        try:
            search = DuckDuckGoSearchRun()
            result = search.run("LangChain AI")
            print(f"\n  搜索结果示例: {result[:100]}...")
        except ImportError as e:
            print(f"\n  ⚠️  DuckDuckGoSearchRun 需要额外依赖: {e}")
            print("     安装命令: pip install ddgs")

    if WikipediaQueryRun and WikipediaAPIWrapper:
        try:
            wiki_wrapper = WikipediaAPIWrapper()
            wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)
            result = wiki.run("Python (programming language)")
            print(f"\n  Wikipedia 结果示例: {result[:100]}...")
        except (ImportError, Exception) as e:
            print(f"\n  ⚠️  WikipediaQueryRun 不可用: {e}")


if __name__ == "__main__":
    print("=== Day 5: Agent + Tools ===\n")
    demo_create_agent()
    demo_tool_calling()
    demo_builtin_tools()
    print("\n=== Day 5 完成 ===")

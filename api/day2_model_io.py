"""
Day 2: Model I/O - ChatModel / LLM / PromptTemplate / OutputParser

本文件展示 LangChain Model I/O 组件的用法。
"""

import os
import sys
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser, StrOutputParser, CommaSeparatedListOutputParser
from pydantic import BaseModel

# 多模型支持
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_community.chat_models import ChatZhipuAI
except ImportError:
    ChatZhipuAI = None

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

_api_key = os.environ.get("OPENAI_API_KEY", "")
_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
_default_model = os.environ.get("OPENAI_MODEL", "deepseek-ai/DeepSeek-V3")


def _has_api_key():
    return bool(_api_key and _api_key != "your-api-key")


def _create_llm(model: str = None, temperature: float = 0):
    return ChatOpenAI(
        model=model or _default_model,
        base_url=_base_url,
        api_key=_api_key,
        temperature=temperature
    )


def demo_chatmodel():
    """ChatModel：消息进，消息出"""
    print("【ChatModel 演示】")
    print(f"  模型: {_default_model}")

    if _has_api_key():
        llm = _create_llm()
        response = llm.invoke([{"role": "user", "content": "用一句话解释什么是 RAG"}])
        print(f"  响应: {response.content}")
    else:
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")


def demo_llm():
    """LLM：字符串进，字符串出"""
    print("\n【LLM 演示】")
    print(f"  模型: {os.environ.get('OPENAI_MODEL', 'deepseek-ai/DeepSeek-V3')} (OpenAI 兼容)")

    if _has_api_key():
        llm = OpenAI(model=os.environ.get("OPENAI_MODEL", "deepseek-ai/DeepSeek-V3"), base_url=_base_url, api_key=_api_key)
        response = llm.invoke("解释什么是 RAG")
        print(f"  响应: {response}")
    else:
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")


def demo_prompt_template():
    """PromptTemplate：模板化"""
    print("\n【PromptTemplate 演示】")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{topic}助手"),
        ("human", "请回答以下{question_count}个问题：{questions}"),
    ])
    print(f"  Prompt 模板: input_variables={prompt.input_variables}")

    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    llm = _create_llm()
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "topic": "Python 编程",
        "question_count": 2,
        "questions": "1. 什么是装饰器？2. __init__ 方法有什么用？"
    })
    print(f"  响应: {result[:100]}...")


def demo_json_output_parser():
    """JsonOutputParser：结构化 JSON 输出"""
    print("\n【JsonOutputParser 演示】")

    class Recipe(BaseModel):
        name: str
        ingredients: list[str]
        steps: list[str]

    parser = JsonOutputParser(pydantic_object=Recipe)
    print(f"  Parser: {type(parser).__name__}")

    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    prompt = PromptTemplate.from_template(
        "生成一个{cuisine}风味的简单甜点食谱"
    )
    chain = prompt | _create_llm(temperature=0) | parser
    result = chain.invoke({"cuisine": "法式"})
    print(f"  解析结果: {result}")
    print(f"  类型: {type(result)}")


def demo_stream():
    """流式输出：实时显示生成过程"""
    print("\n【流式输出演示】")
    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    llm = _create_llm(temperature=0)
    print("  开始流式响应: ", end="", flush=True)
    for chunk in llm.stream("用3句话解释什么是LangChain"):
        print(chunk.content, end="", flush=True)
    print()


def demo_pydantic_output_parser():
    """PydanticOutputParser：带类型校验的结构化输出"""
    print("\n【PydanticOutputParser 演示】")

    class MeetingAction(BaseModel):
        action: str = Field(description="需要执行的动作")
        owner: str = Field(description="负责人姓名")
        deadline: str = Field(description="截止日期，YYYY-MM-DD 格式")

    parser = PydanticOutputParser(pydantic_object=MeetingAction)
    print(f"  Parser: {type(parser).__name__}")
    print(f"  格式说明:\n{parser.get_format_instructions()}")

    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    prompt = PromptTemplate.from_template(
        "从以下文本中提取会议信息：{text}"
    )
    chain = prompt | _create_llm(temperature=0) | parser
    result = chain.invoke({"text": "周三下午3点开会讨论项目进度，张三负责，李四 deadline 是下周五"})
    print(f"  解析结果: action={result.action}, owner={result.owner}, deadline={result.deadline}")
    print(f"  类型: {type(result).__name__} (MeetingAction)")


def demo_error_handling():
    """错误处理：Rate Limit / API Key / 超时"""
    print("\n【错误处理演示】")
    print("  实际配置: llm = ChatOpenAI(..., max_retries=3, request_timeout=30)")
    llm_with_retry = ChatOpenAI(
        model=_default_model,
        base_url=_base_url,
        api_key=_api_key,
        max_retries=3,
        request_timeout=30
    )
    print("  ✅ 错误处理参数配置正确: max_retries=3, request_timeout=30")


def demo_multi_model_support():
    """多模型支持：Anthropic / 阿里云 / Ollama"""
    print("\n【多模型支持演示】")

    # Anthropic Claude
    if ChatAnthropic:
        print("  ✅ ChatAnthropic 可用: from langchain_anthropic import ChatAnthropic")
        print("    使用: ChatAnthropic(model='claude-3-5-sonnet-20241022', anthropic_api_key=...)")
    else:
        print("  ⚠️  ChatAnthropic 未安装 (pip install langchain-anthropic)")

    # 阿里云通义千问
    if ChatZhipuAI:
        print("  ✅ ChatZhipuAI 可用: from langchain_community.chat_models import ChatZhipuAI")
        print("    使用: ChatZhipuAI(zhipuai_api_key=...)")
    else:
        print("  ⚠️  ChatZhipuAI 未安装 (pip install langchain-community)")

    # Ollama 本地模型
    if ChatOllama:
        print("  ✅ ChatOllama 可用: from langchain_ollama import ChatOllama")
        print("    使用: ChatOllama(model='llama3', base_url='http://localhost:11434')")
    else:
        print("  ⚠️  ChatOllama 未安装 (pip install langchain-ollama)")


def demo_prompt_from_template():
    """PromptTemplate.from_template：字符串模板"""
    print("\n【PromptTemplate.from_template 演示】")

    prompt = PromptTemplate.from_template("请把以下句子翻译成{language}：{sentence}")
    print(f"  模板: {prompt.template}")
    print(f"  变量: {prompt.input_variables}")

    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    chain = prompt | _create_llm(temperature=0) | StrOutputParser()
    result = chain.invoke({"language": "英语", "sentence": "LangChain 让 LLM 应用开发更简单"})
    print(f"  翻译结果: {result}")


def demo_comma_separated_list_parser():
    """CommaSeparatedListOutputParser：逗号分隔列表解析"""
    print("\n【CommaSeparatedListOutputParser 演示】")

    parser = CommaSeparatedListOutputParser()
    print(f"  Parser: {type(parser).__name__}")

    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    prompt = PromptTemplate.from_template(
        "列出 5 个 {topic} 相关的术语，用逗号分隔"
    )
    chain = prompt | _create_llm(temperature=0) | parser
    result = chain.invoke({"topic": "Python 编程"})
    print(f"  解析结果: {result}")
    print(f"  类型: {type(result)} (list)")


if __name__ == "__main__":
    print("=== Day 2: Model I/O ===\n")
    demo_chatmodel()
    demo_llm()
    demo_prompt_template()
    demo_json_output_parser()
    demo_pydantic_output_parser()
    demo_stream()
    demo_error_handling()
    demo_multi_model_support()
    demo_prompt_from_template()
    demo_comma_separated_list_parser()
    print("\n=== Day 2 完成 ===")

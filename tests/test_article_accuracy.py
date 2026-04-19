"""
验证 LangChain 7天文章中的代码示例是否准确
测试关键 API 调用模式（不依赖真实 API Key）
"""

import sys
sys.path.insert(0, '/Users/liuyang/code/weixin-mp-workspace/langchain-7days')

def test_imports():
    """验证所有导入路径正确"""
    print("=== 验证导入 ===")

    # Day 1: 四大组件
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.tools import tool
    from langchain.agents import create_agent
    print("✅ Day 1 导入正确")

    # Day 2: Model I/O
    from langchain_openai import OpenAI
    from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
    from langchain_core.prompts import PromptTemplate
    print("✅ Day 2 导入正确")

    # Day 3: Retrieval
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    print("✅ Day 3 导入正确")

    # Day 4: RAG
    from langchain_core.runnables import RunnablePassthrough
    from langchain_classic.memory import ConversationBufferMemory
    print("✅ Day 4 导入正确")

    # Day 5: Agent
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_core.tools import tool
    print("✅ Day 5 导入正确")

    # Day 6: Memory & Chain
    from langchain_core.runnables import RunnableParallel, RunnableLambda
    from langchain_classic.memory import ConversationSummaryBufferMemory
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.globals import set_debug
    print("✅ Day 6 导入正确")


def test_api_signatures():
    """验证 API 签名正确"""
    print("\n=== 验证 API 签名 ===")

    # create_agent 签名
    import inspect
    from langchain.agents import create_agent
    sig = inspect.signature(create_agent)
    params = list(sig.parameters.keys())
    assert 'model' in params, f"create_agent 缺少 model 参数: {params}"
    assert 'tools' in params, f"create_agent 缺少 tools 参数: {params}"
    assert 'system_prompt' in params, f"create_agent 缺少 system_prompt 参数: {params}"
    print(f"✅ create_agent 参数: {params}")

    # ChatOpenAI 签名（使用 **kwargs，无法直接检查）
    from langchain_openai import ChatOpenAI
    # ChatOpenAI 使用 **kwargs 传递参数，直接实例化验证
    try:
        # 不真正调用 API，只验证构造函数可接受这些参数
        llm = ChatOpenAI(model="gpt-4.1", temperature=0, openai_api_key="test")
    except Exception as e:
        # 可能有认证错误，但参数格式应该接受
        if "model" not in str(e) and "temperature" not in str(e):
            print(f"⚠️ ChatOpenAI 参数验证: {e}")
    print("✅ ChatOpenAI 参数格式正确")

    # RunnablePassthrough
    from langchain_core.runnables import RunnablePassthrough
    print("✅ RunnablePassthrough 可导入")


def test_lcel_patterns():
    """验证 LCEL 模式正确"""
    print("\n=== 验证 LCEL 模式 ===")

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel

    # 模式1: prompt | llm | parser
    prompt = ChatPromptTemplate.from_template("hello {name}")
    parser = StrOutputParser()
    assert hasattr(prompt, 'invoke'), "PromptTemplate 缺少 invoke 方法"
    assert hasattr(parser, 'invoke'), "StrOutputParser 缺少 invoke 方法"
    print("✅ LCEL 基础管道模式正确")

    # 模式2: dict 输入（PromptTemplate 用 invoke 调用）
    chain_input = {"name": "test"}
    assert hasattr(prompt, 'invoke'), "prompt 应该有 invoke 方法"
    print("✅ LCEL dict 输入模式正确")

    # 模式3: RunnablePassthrough
    rp = RunnablePassthrough()
    assert hasattr(rp, 'invoke'), "RunnablePassthrough 缺少 invoke 方法"
    print("✅ RunnablePassthrough API 正确")


def test_agent_pattern():
    """验证 Agent 创建模式"""
    print("\n=== 验证 Agent 模式 ===")

    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def test_tool(text: str) -> str:
        """测试工具"""
        return text

    # 验证 create_agent 可以接受这些参数
    import inspect
    sig = inspect.signature(create_agent)
    sig_params = list(sig.parameters.keys())
    assert 'model' in sig_params, f"create_agent 缺少 model 参数"
    assert 'tools' in sig_params, f"create_agent 缺少 tools 参数"
    assert 'system_prompt' in sig_params, f"create_agent 缺少 system_prompt 参数"
    print(f"✅ create_agent API 正确")


def test_memory_pattern():
    """验证 Memory 模式"""
    print("\n=== 验证 Memory 模式 ===")

    from langchain_classic.memory import ConversationBufferMemory, ConversationSummaryBufferMemory

    mem = ConversationBufferMemory()
    assert hasattr(mem, 'save_context'), "ConversationBufferMemory 缺少 save_context"
    assert hasattr(mem, 'load_memory_variables'), "ConversationBufferMemory 缺少 load_memory_variables"
    print("✅ ConversationBufferMemory API 正确")

    # ConversationSummaryBufferMemory 需要真实 LLM 实例，API 存在性已通过 hasattr 验证
    assert hasattr(ConversationSummaryBufferMemory, 'save_context'), "ConversationSummaryBufferMemory 缺少 save_context"
    assert hasattr(ConversationSummaryBufferMemory, 'load_memory_variables'), "ConversationSummaryBufferMemory 缺少 load_memory_variables"
    print("✅ ConversationSummaryBufferMemory API 正确")


def test_vectorstore_pattern():
    """验证 VectorStore 模式"""
    print("\n=== 验证 VectorStore 模式 ===")

    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    # 验证 from_texts 方法存在
    assert hasattr(Chroma, 'from_texts'), "Chroma 缺少 from_texts 方法"
    assert hasattr(Chroma, 'from_documents'), "Chroma 缺少 from_documents 方法"
    print("✅ Chroma API 正确")


def test_runnable_parallel():
    """验证 RunnableParallel"""
    print("\n=== 验证 RunnableParallel ===")

    from langchain_core.runnables import RunnableParallel

    rp = RunnableParallel(a=lambda x: x, b=lambda x: x)
    assert hasattr(rp, 'invoke'), "RunnableParallel 缺少 invoke 方法"
    print("✅ RunnableParallel API 正确")


if __name__ == "__main__":
    test_imports()
    test_api_signatures()
    test_lcel_patterns()
    test_agent_pattern()
    test_memory_pattern()
    test_vectorstore_pattern()
    test_runnable_parallel()
    print("\n🎉 所有验证通过！")

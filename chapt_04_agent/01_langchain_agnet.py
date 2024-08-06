import os

from langchain_community.chat_models import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

# langSmith，服务的 API key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'

# 千帆的 API key 和 Secret key
os.environ["QIANFAN_AK"] = "Z1Os3TDFDM7zPDOjspytwEjD"
os.environ["QIANFAN_SK"] = "nkMXxCV0ulG0hVWu7YOCMkHURP9d7tGM"

# langgraph
os.environ['ANTHROPIC_API_KEY'] = ''


def test01():
    # Create the agent
    memory = SqliteSaver.from_conn_string(":memory:")
    model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
    search = TavilySearchResults(max_results=2)
    tools = [search]
    agent_executor = create_react_agent(model, tools, checkpointer=memory)

    # Use the agent
    config = {"configurable": {"thread_id": "abc123"}}
    for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content="hi im bob! and i live in sf")]}, config
    ):
        print(chunk)
        print("----")

    for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content="whats the weather where I live?")]}, config
    ):
        print(chunk)
        print("----")


if __name__ == '__main__':
    test01()
    pass

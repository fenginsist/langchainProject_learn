import os

from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# langSmith，服务的 API key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'

# 千帆的 API key 和 Secret key
os.environ["QIANFAN_AK"] = "Z1Os3TDFDM7zPDOjspytwEjD"
os.environ["QIANFAN_SK"] = "nkMXxCV0ulG0hVWu7YOCMkHURP9d7tGM"


def test1_HumanMessage():
    '''
    正常
    :return:
    '''
    model = QianfanChatEndpoint(
        streaming=True,
        model="ERNIE-4.0-8K"  # ERNIE-Bot
    )
    result = model.invoke([HumanMessage(content="Hi! I'm Bob")])
    print(result)
    result2 = model.invoke([HumanMessage(content="What's my name?")])
    print(result2)


def test2():
    ''' 消息列表 '''
    model = QianfanChatEndpoint(
        streaming=True,
        model="ERNIE-4.0-8K"  # ERNIE-Bot
    )
    result = model.invoke(
        [
            HumanMessage(content="Hi! I'm Bob"),
            AIMessage(content="Hello Bob! How can I assist you today?"),
            HumanMessage(content="What's my name?"),
        ]
    )
    print(result)


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def test3_history_memory():
    '''
    使用 session_id 记住每一段对话。
    :return:
    '''
    model = QianfanChatEndpoint(
        streaming=True,
        model="ERNIE-4.0-8K"  # ERNIE-Bot
    )

    with_message_history = RunnableWithMessageHistory(model, get_session_history)

    '''
    第一次: session_id = abc2
    '''
    config1 = {"configurable": {"session_id": "abc2"}}
    response1 = with_message_history.invoke(
        [HumanMessage(content="Hi! 我是中创")],
        config=config1,
    )
    print('response1: ', response1)
    print('response1.content: ', response1.content)  # 您好，中创！很高兴与您交流。。。。

    '''
    第二次: 更改 session_id = abc3
    '''
    config2 = {"configurable": {"session_id": "abc3"}}
    response2 = with_message_history.invoke(
        [HumanMessage(content="What's my name?")],  # 更改问题，
        config=config2,
    )
    print('response2: ', response2)
    print('response2.content: ', response2.content)  # I don't know your name。。。。

    '''
    第三次: 更改回 session_id = abc2，和第一次的一样了
    '''
    config3 = {"configurable": {"session_id": "abc2"}}
    response3 = with_message_history.invoke(
        [HumanMessage(content="What's my name?")],  # 问题同上
        config=config3,
    )
    print('response3: ', response3) # content='您的名字是中创。如果您有其他疑问。。。
    print('response3.content: ', response3.content)  # 您的名字是中创。如果您有其他疑问或者需要帮助，请随时告诉我。


if __name__ == '__main__':
    # test1_HumanMessage()
    # test2()
    test3_history_memory()

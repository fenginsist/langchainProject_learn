import os
from operator import itemgetter

from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import trim_messages, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory

# langSmith，服务的 API key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'

# 千帆的 API key 和 Secret key
os.environ["QIANFAN_AK"] = "Z1Os3TDFDM7zPDOjspytwEjD"
os.environ["QIANFAN_SK"] = "nkMXxCV0ulG0hVWu7YOCMkHURP9d7tGM"

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def test1_manager_history():
    model = QianfanChatEndpoint(
        streaming=True,
        model="ERNIE-4.0-8K"  # ERNIE-Bot
    )
    trimmer = trim_messages(
        max_tokens=65,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )
    messages = [    # 一共5段对话
        SystemMessage(content="you're a good assistant"),
        HumanMessage(content="hi! I'm bob"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
        HumanMessage(content="whats 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!"),
    ]
    res = trimmer.invoke(messages)
    print('res: ', res)  # [SystemMessage(content="you're a good assistant"), HumanMessage(content="hi! I'm bob")。。。


def test2_manager_history():
    model = QianfanChatEndpoint(
        streaming=True,
        model="ERNIE-4.0-8K"  # ERNIE-Bot
    )

    trimmer = trim_messages(    # 是一个用于消息修剪或裁剪的函数或方法，主要用于确保输入的消息列表或对话不会超过模型处理能力的最大 token 数量。
        max_tokens=65,          # 指定裁剪后的消息总长度上限，以 token 为单位。
        strategy="last",        # 指定修剪消息时采用的策略。"last"：保留最后的消息，丢弃最早的部分。"first"：保留最早的消息，丢弃后续多余的部分。"middle"：从中间裁剪，可能保留开头和结尾，丢弃中间部分。
        token_counter=model,    # 指定用于计算 token 数量的计数器，通常是一个语言模型实例。通过 token_counter，可以精确计算每条消息占用的 token 数量，从而实现精确的裁剪。
        include_system=True,    # 决定是否在修剪过程中包括系统消息。通过设置为 True，修剪过程会考虑包括系统消息的总 token 计数；如果设置为 False，系统消息将被忽略，仅裁剪用户和助手消息
        allow_partial=False,    # 指定是否允许部分消息被裁剪。如果设置为 True，则一条消息超过 token 限制时，允许截取一部分；如果为 False，则整条消息要么保留，要么被完全删除
        start_on="human",       # 指定修剪操作开始时的角色。"human"：从人类消息开始。"system"：从系统消息开始。"assistant"：从助手消息开始。
    )

    messages = [
        SystemMessage(content="you're a good assistant"),
        HumanMessage(content="hi! I'm bob"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
        HumanMessage(content="whats 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!"),
    ]

    '''
    添加了 提示模板 *****
    '''
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = (
            RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
            | prompt
            | model
    )

    response1 = chain.invoke(
        {
            "messages": messages + [HumanMessage(content="what's my name?")],  #
            "language": "English",
        }
    )
    print('response1.content: ', response1.content)  # your name is bob.

    response2 = chain.invoke(
        {
            "messages": messages + [HumanMessage(content="what math problem did i ask")],
            "language": "English",
        }
    )
    print('response2.content: ', response2.content)  # you asked me what 2 + 2 equals.

    '''
    加入到 消息历史中
    '''
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages",
    )

    config = {"configurable": {"session_id": "abc20"}}

    response3 = with_message_history.invoke(
        {
            "messages": messages + [HumanMessage(content="whats my name?")],
            "language": "English",
        },
        config=config,
    )
    print('response3.content: ', response3.content)  # bob, right?

    response4 = with_message_history.invoke(
        {
            "messages": [HumanMessage(content="what math problem did i ask?")],
            "language": "English",
        },
        config=config,
    )
    print('response4.content: ', response4.content)  # you asked "whats 2 + 2"

    '''
    stream back each token as it is generated. This allows the user to see progress.
    '''
    # config = {"configurable": {"session_id": "abc15"}}
    # for r in with_message_history.stream(
    #         {
    #             "messages": [HumanMessage(content="hi! I'm todd. tell me a joke")],
    #             "language": "English",
    #         },
    #         config=config,
    # ):
    #     print(r.content, end="|")


if __name__ == '__main__':
    # test1_manager_history()
    test2_manager_history()
    pass

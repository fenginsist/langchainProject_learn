import os

from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

# langSmith，服务的 API key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'

# 千帆的 API key 和 Secret key
os.environ["QIANFAN_AK"] = "Z1Os3TDFDM7zPDOjspytwEjD"
os.environ["QIANFAN_SK"] = "nkMXxCV0ulG0hVWu7YOCMkHURP9d7tGM"


def test1_prompt_template():
    model = QianfanChatEndpoint(
        streaming=True,
        model="ERNIE-4.0-8K"  # ERNIE-Bot
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | model

    '''请注意，这略微改变了输入类型 - 我们现在不是传入消息列表，而是传入一个带有键的字典，其中包含消息列表'''
    response = chain.invoke({"messages": [HumanMessage(content="hi! I'm bob")]})
    print(response.content)  # Hello Bob, nice to meet you! What can I help you with?


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def test2_prompt_template_with_message_history():
    '''
    使用模板，带上消息历史
    :return:
    '''
    model = QianfanChatEndpoint(
        streaming=True,
        model="ERNIE-4.0-8K"  # ERNIE-Bot
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | model

    '''存入历史'''
    with_message_history = RunnableWithMessageHistory(chain, get_session_history)

    config = {"configurable": {"session_id": "abc5"}}
    response = with_message_history.invoke(
        [HumanMessage(content="Hi! I'm Jim")],
        config=config,
    )

    print('response.content: ',
          response.content)  # Hello Jim! Nice to meet you. Is there anything I can assist you with?


def test3_prompt_template_with_message_history_add_completed():
    '''
    使用提示模板，带上消息历史,
    再提示模板中，加上传参
    :return:
    '''
    model = QianfanChatEndpoint(
        streaming=True,
        model="ERNIE-4.0-8K"  # ERNIE-Bot
    )

    '''
    提示模板 传参
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
    chain = prompt | model

    response1 = chain.invoke(
        {"messages": [HumanMessage(content="hi! I'm bob")], "language": "Spanish"}
    )
    print('response1.content: ', response1.content)  # ¡Hola Bob! ¿Cómo estás? ¿Necesitas ayuda en algo?

    '''
    存入 消息历史
    '''
    with_message_history = RunnableWithMessageHistory(chain,
                                                      get_session_history,
                                                      input_messages_key="messages", )

    config = {"configurable": {"session_id": "abc11"}}

    response2 = with_message_history.invoke(
        {"messages": [HumanMessage(content="hi! I'm todd")],
         "language": "Spanish"},
        config=config,
    )
    print('response2.content: ', response2.content)  # ¡Hola! Bienvenido, Todd. ¿Cómo puedo ayudar?

    # 第二次传入Q，和对应参数
    response3 = with_message_history.invoke(
        {"messages": [HumanMessage(content="whats my name?")], "language": "Spanish"},
        config=config,
    )
    print('response3.content: ', response3.content)  # Tu nombre es Todd. ¿Necesitas ayuda con algo?


if __name__ == '__main__':
    # test1_prompt_template()
    # test2_prompt_template_with_message_history()
    test3_prompt_template_with_message_history_add_completed()

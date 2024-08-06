"""For basic init and call"""
import asyncio
import os

from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.language_models.chat_models import HumanMessage

# 千帆的 API key 和 Secret key
os.environ["QIANFAN_AK"] = "Z1Os3TDFDM7zPDOjspytwEjD"
os.environ["QIANFAN_SK"] = "nkMXxCV0ulG0hVWu7YOCMkHURP9d7tGM"


def test1_invoke():
    '''
    正常版本
    :return:
    '''
    # 模型
    chat = QianfanChatEndpoint(streaming=True)
    # 问题
    messages = [HumanMessage(content="你是谁")]
    # 生成回答
    response = chat.invoke(messages)
    print(response)
    """
    content='您好，我是百度研发的文心一言，是新一代的人工智能语言模型。我能做许多事，例如提供有用的信息，解答疑难问题，与用户互动对话等等。我可以协助处理多种类型的任务和需求，并致力于为您提供最准确、
    最有价值的信息和建议。如果您有任何问题需要帮助，可以随时向我提问。' response_metadata={'token_usage': {'prompt_tokens': 1, 'completion_tokens': 70, 'total_tokens': 71}, 
    'model_name': 'ERNIE-Lite-8K', 'finish_reason': 'stop'} id='run-d31036a1-aba5-4567-896c-acd715f8e23a-0'
    """


async def test2_ainvoke_batch():
    '''
    测试 await 异步
    要注意。这里执行 异步函数，有坑，仔细看执行的方法。
    :return:
    '''
    chat = QianfanChatEndpoint(streaming=True)
    messages = [HumanMessage(content="你是谁")]

    response = chat.invoke(messages)
    print('response: ', response)  # content='您好，我...' response_metadata='{}' id=''

    res = await chat.ainvoke(messages)
    print('res: ', res)  # 返回的就是content， content='您。' response_metadata='' id=''

    res2 = chat.batch([messages])
    print('res2: ', res2)  # 返回的是list，AIMessage对象；[AIMessage(content='', response_metadata='', id='')]


async def test3_stream():
    '''
    新方法：chat.stream(messages)

    逐步获取生成的文本内容。这种方法特别适用于生成长文本或需要实时输出的场景，比如聊天机器人、内容生成工具等。
    方法作用：
        1、逐步生成输出：stream 方法用于逐步输出语言模型生成的内容，而不是一次性返回整个结果。这样可以实现更快的响应时间和更好的用户体验。
        2、 适用于流式输出场景： 在生成长文本或连续对话时，stream 可以实时返回生成的部分内容，使得用户无需等待完整生成，便可以开始阅读或进行交互。
        3、节省资源： 通过逐步输出，stream 方法可以减少内存占用和处理时间，尤其在生成长篇文本时，逐步处理更为高效。
    使用场景：
        1、聊天机器人：流式输出可以使用户在输入问题后立即看到部分响应，而不是等待模型生成完整的答案。
        2、长文本生成：如小说或文章生成，用户可以实时查看内容的进展。
        3、逐步反馈系统：在需要不断获取反馈的系统中，stream 方法可以及时返回当前状态或部分结果。
    :return:
    '''
    chat = QianfanChatEndpoint(streaming=True)
    messages = [HumanMessage(content="你是谁")]

    try:
        print('type(chat.stream(messages)): ', type(chat.stream(messages))) # <class 'generator'>
        for chunk in chat.stream(messages):
            print('type(chunk): ', type(chunk)) #  <class 'langchain_core.messages.ai.AIMessageChunk'>
            print(chunk.content, end="", flush=True)
            print()
    except TypeError as e:
        print("")


def test4_use_other_model():
    '''
    使用千帆的 其他模型
    :return:
    '''
    chatBot = QianfanChatEndpoint(
        streaming=True,
        model="ERNIE-4.0-8K"  # ERNIE-Bot
    )
    messages = [HumanMessage(content="你是谁，你知道北京是啥么")]
    res = chatBot.invoke(messages)
    print('res: ', res)


if __name__ == '__main__':
    '''
    langchain 调用 百度千帆的 LLM。
    '''
    # test1()
    # asyncio.run(test2())
    asyncio.run(test3_stream())
    # test4_use_other_model()

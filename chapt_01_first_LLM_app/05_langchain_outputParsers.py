import os

from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

# 千帆的 API key 和 Secret key
os.environ["QIANFAN_AK"] = "Z1Os3TDFDM7zPDOjspytwEjD"
os.environ["QIANFAN_SK"] = "nkMXxCV0ulG0hVWu7YOCMkHURP9d7tGM"


def test01_outputParsers():
    # 1 创建模型
    chatBot = QianfanChatEndpoint(
        streaming=True,
        model="ERNIE-4.0-8K"  # ERNIE-Bot
    )
    # 2 创新消息列表
    messages = [
        SystemMessage(content="Translate the following from English into Italian"),
        HumanMessage(content="hi!"),
    ]
    '''输出第一种方式： 模型.invoke(message)'''
    result = chatBot.invoke(messages)
    print('result: ',
          result)  # content='Ciao!' response_metadata={'token_usage': {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}, 'model_name': 'ERNIE-4.0-8K', 'finish_reason': 'stop'} id='run-684880e5-367f-423b-ad83-97e406fbc483-0'
    print('result.content: ', result.content)  # Ciao!

    # 4 输出解析
    parser = StrOutputParser()
    ''' 4.1 使用 解析 输出第一种方式 '''
    res = parser.invoke(result)
    print('res: ', res)  # Ciao!

    ''' 4.2 使用解析 链式输出的第二种方式 '''
    chain = chatBot | parser
    print('chain: ', chain)
    # first=QianfanChatEndpoint(client=<qianfan.resources.llm.chat_completion.ChatCompletion object at 0x000001FD9C01BF40>, qianfan_ak=SecretStr('**********'), qianfan_sk=SecretStr('**********'), streaming=True, model='ERNIE-4.0-8K') last=StrOutputParser()
    res2 = chain.invoke(messages)
    print('res2: ', res2)  # Ciao!


if __name__ == '__main__':
    test01_outputParsers()

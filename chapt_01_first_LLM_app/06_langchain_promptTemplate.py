import os

from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 千帆的 API key 和 Secret key
os.environ["QIANFAN_AK"] = "Z1Os3TDFDM7zPDOjspytwEjD"
os.environ["QIANFAN_SK"] = "nkMXxCV0ulG0hVWu7YOCMkHURP9d7tGM"


def test01_promptTemplate():
    # 1. 创建模型
    chatBot = QianfanChatEndpoint(
        streaming=True,
        model="ERNIE-4.0-8K"  # ERNIE-Bot
    )

    # 创建一个字符串，将其格式化为系统消息
    system_template = "Translate the following into {language}:"
    # 2. 创建 提示模板
    # 创建 PromptTemplate。这是一个更简单的模板组合，用于将要翻译的文本放在 system_template
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    # 对上面的 {} 中的两个数据进行传值。
    result = prompt_template.invoke({"language": "italian", "text": "hi"})
    print('result: ',
          result)  # 这里的 result 就是 补充完参数后的数据: messages=[SystemMessage(content='Translate the following into italian:'), HumanMessage(content='hi')]
    res = result.to_messages()
    print('res: ', res)  # [SystemMessage(content='Translate the following into italian:'), HumanMessage(content='hi')]

    # 模型 invoke 提示模板
    r = chatBot.invoke(result)
    print('r: ',
          r)  # content='ciao' response_metadata={'token_usage': {'prompt_tokens': 8, 'completion_tokens': 1, 'total_tokens': 9}, 'model_name': 'ERNIE-4.0-8K', 'finish_reason': 'stop'} id='run-a743ea4c-467d-4e31-b8e5-11c1d16b52b6-0'

    # 输出解析
    parser = StrOutputParser()
    res = parser.invoke(r)
    print('res: ', res)  # ciao


if __name__ == '__main__':
    test01_promptTemplate()

import os

from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# langSmith，服务的 API key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'

# 千帆的 API key 和 Secret key
os.environ["QIANFAN_AK"] = "Z1Os3TDFDM7zPDOjspytwEjD"
os.environ["QIANFAN_SK"] = "nkMXxCV0ulG0hVWu7YOCMkHURP9d7tGM"


def test01_promptTemplate_LCEL():
    # 1. 创建模型
    chatBot = QianfanChatEndpoint(
        streaming=True,
        model="ERNIE-4.0-8K"  # ERNIE-Bot
    )

    # 创建一个字符串模板，将其格式化为系统消息
    system_template = "Translate the following into {language}:"
    # 2. 创建 提示模板
    # 创建 PromptTemplate。这是一个更简单的模板组合，用于将要翻译的文本放在 system_template
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    # 输出解析
    parser = StrOutputParser()
    chain = prompt_template | chatBot | parser      # 消息模板|模型|解析 输出
    response = chain.invoke({"language": "italian", "text": "hi"})
    print(response)  # ciao


if __name__ == '__main__':
    '''
    不在使用invoke 进行一步步的链接
    使用 pipe() 管道符 | 进行连接。然后在进行填充参数，直接返回 回答。
    '''
    test01_promptTemplate_LCEL()

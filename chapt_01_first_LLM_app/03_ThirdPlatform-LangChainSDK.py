import os
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

os.environ['CHATGLM_API_KEY'] = '09accc9c442473f6029f0062d1cd9411.cAbNNUUldblEFrAo'


def test1():
    llm = ChatOpenAI(
        temperature=0.95,
        model="glm-4",
        openai_api_key=os.environ.get('CHATGLM_API_KEY'),
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a nice chatbot having a conversation with a human."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    conversation.invoke({"question": "tell me a joke"})


if __name__ == '__main__':
    '''
    chatglm4 集成的第三方平台 Langchain。本质上还是调用 chatglm4
    文档：https://open.bigmodel.cn/dev/api#langchain_sdk
    '''
    test1()

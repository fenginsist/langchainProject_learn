import os

from langchain_chroma import Chroma
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# langSmith，服务的 API key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'

# 千帆的 API key 和 Secret key
os.environ["QIANFAN_AK"] = "Z1Os3TDFDM7zPDOjspytwEjD"
os.environ["QIANFAN_SK"] = "nkMXxCV0ulG0hVWu7YOCMkHURP9d7tGM"


def create_documents():
    documents = [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
            metadata={"source": "fish-pets-doc"},
        ),
        Document(
            page_content="Parrots are intelligent birds capable of mimicking human speech.",
            metadata={"source": "bird-pets-doc"},
        ),
        Document(
            page_content="Rabbits are social animals that need plenty of space to hop around.",
            metadata={"source": "mammal-pets-doc"},
        ),
    ]
    return documents


def test01_retrieval_RunnableLambda():
    documents = create_documents()

    # 生成  向量存储
    vectorstore = Chroma.from_documents(
        documents,
        embedding=QianfanEmbeddingsEndpoint(),
    )

    '''
    第一种方法：使用 RunnableLambda 生成检索器
    '''
    retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result
    # 检索并返回
    response = retriever.batch(["cat", "shark"])
    print('response: ', response)
    '''
    输出
    response:  [[Document(metadata={'source': 'mammal-pets-doc'}, page_content='Cats are independent pets that often enjoy their own space.')], 
    [Document(metadata={'source': 'fish-pets-doc'}, page_content='Goldfish are popular pets for beginners, requiring relatively simple care.')]]
    '''


def test02_retrieval_as_retriever():
    documents = create_documents()

    # 生成  向量存储
    vectorstore = Chroma.from_documents(
        documents,
        embedding=QianfanEmbeddingsEndpoint(),
    )
    '''
    第二种方法：vectorstore.as_retriever() 生成 检索器
    '''
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    response = retriever.batch(["cat", "shark"])
    print('response: ', response)
    '''
    输出
    response:  [[Document(metadata={'source': 'mammal-pets-doc'}, page_content='Cats are independent pets that often enjoy their own space.')], 
    [Document(metadata={'source': 'fish-pets-doc'}, page_content='Goldfish are popular pets for beginners, requiring relatively simple care.')]]
    '''


def test03_retrieval_QianfanLLM_ChatPromptTemplate():
    # 1. 生成模型
    llm = QianfanChatEndpoint(
        streaming=True,
        model="ERNIE-4.0-8K"  # ERNIE-Bot
    )
    '''
    2. 向量存储
    '''
    documents = create_documents()

    # 生成  向量存储
    vectorstore = Chroma.from_documents(
        documents,
        embedding=QianfanEmbeddingsEndpoint(),
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    '''
    3. 
    '''
    message = """
    Answer this question using the provided context only.

    {question}

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages([("human", message)])

    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

    response = rag_chain.invoke("tell me about cats")

    print(response.content) # Cats are independent pets that often enjoy their own space.
    pass


if __name__ == '__main__':
    test01_retrieval_RunnableLambda()
    # test02_retrieval_as_retriever()
    # test03_retrieval_QianfanLLM_ChatPromptTemplate()

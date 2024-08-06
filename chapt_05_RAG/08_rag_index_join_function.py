import os

import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 千帆的 API key 和 Secret key
os.environ["QIANFAN_AK"] = "Z1Os3TDFDM7zPDOjspytwEjD"
os.environ["QIANFAN_SK"] = "nkMXxCV0ulG0hVWu7YOCMkHURP9d7tGM"


def create_retrieval():
    # 1。构建知识库
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    # 2. 划分知识库
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    # 3. 生成 向量存储
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=QianfanEmbeddingsEndpoint())

    # 4. 生成检索器
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return retriever


def test01():
    retriever = create_retrieval()
    llm = QianfanChatEndpoint(
        stream=True,
        model='ERNIE-4.0-8K'
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": "What is Task Decomposition?"})
    print(response["answer"])
    '''
    Task Decomposition（任务分解）是一种将复杂任务或问题细化为更小、更具体子任务的方法。这些子任务相对独立，可以并行执行，从而简化复杂问题的处理，提高整体任务的执行效率。具体来说：
    1. 任务分解的核心思想在于将一个大的任务或问题，分割成若干个小任务，每个小任务可以单独处理，这样做能够降低问题的复杂度，使得每个小任务更易于管理和解决。
    2. 通过任务分解，可以更好地分配资源，使得不同的执行单元（如处理器核心、线程等）能够并行处理这些子任务，从而充分利用系统性能，加快整体任务的完成速度。
    3. 任务分解的关键点在于保证子任务之间的独立性，以及确保任务能够均匀分布到所有可执行单元上。这要求在进行任务分解时，需要仔细分析任务之间的依赖关系，以及任务的计算量和资源需求。
    '''
    print('-----------------------------------------')
    for document in response["context"]:
        print(document)
        print()

if __name__ == '__main__':
    test01()

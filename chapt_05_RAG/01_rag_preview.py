import os

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 千帆的 API key 和 Secret key
os.environ["QIANFAN_AK"] = "Z1Os3TDFDM7zPDOjspytwEjD"
os.environ["QIANFAN_SK"] = "nkMXxCV0ulG0hVWu7YOCMkHURP9d7tGM"

def test01_preview():
    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    print('len(docs[0].page_content): ', len(docs[0].page_content))     # 43131

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=QianfanEmbeddingsEndpoint(),
                                        persist_directory='./vectorStore')

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = QianfanChatEndpoint(
        streaming=True,
        model="ERNIE-4.0-8K"  # ERNIE-Bot
    )

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    res = rag_chain.invoke(" Task Decomposition 是什么? ")
    print('res: ', res)

    # cleanup
    vectorstore.delete_collection()


if __name__ == '__main__':
    test01_preview()
import os

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 千帆的 API key 和 Secret key
os.environ["QIANFAN_AK"] = "Z1Os3TDFDM7zPDOjspytwEjD"
os.environ["QIANFAN_SK"] = "nkMXxCV0ulG0hVWu7YOCMkHURP9d7tGM"


def test01_index_load():
    print('------------------------------------------------------test01_index_load start')
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    print('------------------------------------------------------test01_index_load end')
    return docs


def test02_index_split():
    docs = test01_index_load()
    print('------------------------------------------------------test02_index_split start')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    print('len(all_splits): ', len(all_splits))  # 66
    print('type(all_splits): ', type(all_splits))  # <class 'list'>
    print('type(all_splits[0]): ', type(all_splits[0]))  # <class 'langchain_core.documents.base.Document'>
    print('len(all_splits[0].page_content): ', len(all_splits[0].page_content))  # 969
    print('type(all_splits[0].page_content): ', type(all_splits[0].page_content))  # <class 'str'>
    print('all_splits[10].metadata: ', all_splits[10].metadata)
    # {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'start_index': 7056}
    print('all_splits[0].metadata: ', all_splits[0].metadata)
    # {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'start_index': 7056}
    print('------------------------------------------------------test02_index_split end')
    return all_splits


def test03_index_store():
    all_splits = test02_index_split()
    print('------------------------------------------------------test03_index_store start')
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=QianfanEmbeddingsEndpoint())
    print('------------------------------------------------------test03_index_store end')
    return vectorstore


def test04_index_retrieve():
    vectorstore = test03_index_store()
    print('------------------------------------------------------test04_index_retrieve start')
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition using chinese?")
    print('len(retrieved_docs): ', len(retrieved_docs))  # 6
    print('retrieved_docs: ', retrieved_docs)  # [Document(metadata={'source': 'https://lilian。。。。
    print('type(retrieved_docs): ', type(retrieved_docs))  # <class 'list'>

    print('type(retrieved_docs[0]): ', type(retrieved_docs[0]))  # <class 'langchain_core.documents.base.Document'>
    print('retrieved_docs[0]: ', retrieved_docs[0])  # 输出的是 page_content 和 metadata内容

    print('retrieved_docs[0].metadata: ', retrieved_docs[0].metadata)
    # {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'start_index': 34741}
    print('retrieved_docs[0].page_content: ', retrieved_docs[0].page_content)
    # You always add a comment briefly de。。。
    print('------------------------------------------------------test04_index_retrieve end')
    return retriever


def test05_index_generate():
    print('------------------------------------------------------test05_index_generate start')
    prompt = hub.pull("rlm/rag-prompt")

    example_messages = prompt.invoke(
        {"context": "filler context", "question": "filler question"}
    ).to_messages()

    print('example_messages: ', example_messages)  # [HumanMessage(content="You are an assistant for 。。。
    print('type(example_messages): ', type(example_messages))  # <class 'list'>
    print('type(example_messages[0]): ', type(example_messages[0]))
    # <class 'langchain_core.messages.human.HumanMessage'>
    print('example_messages[0].content: ', example_messages[0].content)
    '''
    example_messages[0].content:  You are an assistant for question-answering tasks. Use the following pieces of。。。
    Question: filler question 
    Context: filler context 
    Answer:
    '''
    print('------------------------------------------------------test05_index_generate end')
    return prompt


if __name__ == '__main__':
    test05_index_generate()

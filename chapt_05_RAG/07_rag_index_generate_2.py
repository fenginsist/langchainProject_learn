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


def test01_index_load():
    print('------------------------------------------------------test01_index_load start')
    # Only keep post title, headers, and content from the full HTML.
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
    print('retriever: ', retriever)
    print('type(retriever): ', type(retriever))
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
    return retriever  # retriever retrieved_docs


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


def test06_index_generate_2():
    prompt = test05_index_generate()
    retriever = test04_index_retrieve()

    print('------------------------------------------------------test06_index_generate_2 start')
    llm = QianfanChatEndpoint(
        streaming=True,
        model='ERNIE-4.0-8K'
    )
    print('------------------LLM 创建成功')

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    print('------------------ chain 成功')
    print('rag_chain: ', rag_chain)
    '''
    rag_chain: first={
      context: VectorStoreRetriever(tags=['Chroma', 'QianfanEmbeddingsEndpoint'], vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x000001FE592B1F60>, search_kwargs={'k': 6})
               | RunnableLambda(format_docs),
      question: RunnablePassthrough()
    } middle=[ChatPromptTemplate(input_variables=['context', '.....
    '''
    print('type(rag_chain): ', type(rag_chain))  # <class 'langchain_core.runnables.base.RunnableSequence'>

    '''
    这里报错了：****************************
    TypeError: Additional kwargs key completion_tokens already exists in left dict and value has unsupported type <class 'int'>.
    '''
    for chunk in rag_chain.stream("What is Task Decomposition?"):
        print(chunk, end="", flush=True)
    print('------------------------------------------------------test06_index_generate_2 end')


if __name__ == '__main__':
    test06_index_generate_2()

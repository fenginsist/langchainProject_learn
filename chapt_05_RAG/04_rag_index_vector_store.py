import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter


def test01_index_load():
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    print('------------------------------------------------------')
    return docs


def test02_index_split():
    docs = test01_index_load()

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
    print('------------------------------------------------------')
    return all_splits


def test03_index_store():
    all_splits = test02_index_split()
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=QianfanEmbeddingsEndpoint())


if __name__ == '__main__':
    test03_index_store()

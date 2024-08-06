# 将文本分割成更小的部分以进行处理
import os

from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 读入txt文件
from langchain.document_loaders import TextLoader

# 千帆的 API key 和 Secret key
os.environ["QIANFAN_AK"] = "Z1Os3TDFDM7zPDOjspytwEjD"
os.environ["QIANFAN_SK"] = "nkMXxCV0ulG0hVWu7YOCMkHURP9d7tGM"

# Emdedding模型
from langchain.embeddings import HuggingFaceBgeEmbeddings

# 向量数据库Chroma
from langchain.vectorstores import Chroma


def create():
    '''导入本地文本'''
    loader = TextLoader("test.txt", encoding='utf-8')
    data = loader.load()

    # 将文本分割成长度为200个字符的小块，并且每个小块之间有20个字符的重叠
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    all_splits = text_splitter.split_documents(data)

    '''Embedding模型，暂时使用 QianfanEmbeddingsEndpoint '''
    # embedding_function = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")
    embedding_function = QianfanEmbeddingsEndpoint()

    '''导入向量数据库 Chroma'''
    vectorstore_torist = Chroma.from_documents(all_splits, embedding_function, persist_directory="./vector_store")

    '''----------------使用Ollama:llama3------------------'''
    OLLAMA_MODEL = 'llama3'

    # 在操作系统级别设置该环境
    os.environ['OLLAMA_MODEL'] = OLLAMA_MODEL
    # !echo $OLLAMA_MODEL

    # 启动Ollama服务
    # !ollama serve



if __name__ == '__main__':
    create()
    print('create success!!!')
    import ollama

    ollama.serve()
    # ollama.serve(port=8000, host='0.0.0.0', model_name='llama-7b', max_tokens=1024, temp=0.7, top_p=0.9)

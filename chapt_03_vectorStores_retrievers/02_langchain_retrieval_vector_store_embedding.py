import asyncio
import os

from langchain_chroma import Chroma
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_core.documents import Document

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


def test01_vector_store_embedding_similarity_search():
    '''

    :return:
    '''
    documents = create_documents()
    '''
    to provide an embedding model to specify how text should be converted into a numeric vector. 
    Here we will use Qianfan embeddings.
    '''
    vectorstore = Chroma.from_documents(
        documents,
        embedding=QianfanEmbeddingsEndpoint(),
    )

    response = vectorstore.similarity_search("cat")
    print('response: ', response)
    '''
    [Document(metadata={'source': 'mammal-pets-doc'}, page_content='Cats are independent pets that often enjoy their own space.'), 
    Document(metadata={'source': 'mammal-pets-doc'}, page_content='Rabbits are social animals that need plenty of space to hop around.'), 
    Document(metadata={'source': 'mammal-pets-doc'}, page_content='Dogs are great companions, known for their loyalty and friendliness.'), 
    Document(metadata={'source': 'fish-pets-doc'}, page_content='Goldfish are popular pets for beginners, requiring relatively simple care.')]
    '''


async def test02_vector_store_embedding_asimilarity_search():
    '''

    :return:
    '''
    documents = create_documents()
    '''
    to provide an embedding model to specify how text should be converted into a numeric vector. 
    Here we will use Qianfan embeddings.
    '''
    vectorstore = Chroma.from_documents(
        documents,
        embedding=QianfanEmbeddingsEndpoint(),
    )

    response = await vectorstore.asimilarity_search("cat")
    print('response: ', response)
    '''
    这里的await 有点麻烦
    输出 同 test1
    response:  [Document(metadata={'source': 'mammal-pets-doc'}, page_content='Cats are independent pets that often enjoy their own space.'), 
    Document(metadata={'source': 'mammal-pets-doc'}, page_content='Rabbits are social animals that need plenty of space to hop around.'), 
    Document(metadata={'source': 'mammal-pets-doc'}, page_content='Dogs are great companions, known for their loyalty and friendliness.'), 
    Document(metadata={'source': 'fish-pets-doc'}, page_content='Goldfish are popular pets for beginners, requiring relatively simple care.')]
    '''


def test03_vector_store_embedding_similarity_search_with_score():
    '''

    :return:
    '''
    documents = create_documents()
    '''
    to provide an embedding model to specify how text should be converted into a numeric vector. 
    Here we will use Qianfan embeddings.
    '''
    vectorstore = Chroma.from_documents(
        documents,
        embedding=QianfanEmbeddingsEndpoint(),
    )

    response = vectorstore.similarity_search_with_score("cat")
    print('response: ', response)
    '''
    输出 response:  [(Document(metadata={'source': 'mammal-pets-doc'}, page_content='Cats are independent pets that often enjoy their own space.'), 1.1077227592468262), 
    (Document(metadata={'source': 'mammal-pets-doc'}, page_content='Rabbits are social animals that need plenty of space to hop around.'), 1.454264760017395), 
    (Document(metadata={'source': 'mammal-pets-doc'}, page_content='Dogs are great companions, known for their loyalty and friendliness.'), 1.4571776390075684), 
    (Document(metadata={'source': 'fish-pets-doc'}, page_content='Goldfish are popular pets for beginners, requiring relatively simple care.'), 1.5933877229690552)]
    '''


def test04_vector_store_embedding_similarity_search_by_vector():
    '''

    :return:
    '''
    documents = create_documents()
    '''
    to provide an embedding model to specify how text should be converted into a numeric vector. 
    Here we will use Qianfan embeddings.
    '''
    vectorstore = Chroma.from_documents(
        documents,
        embedding=QianfanEmbeddingsEndpoint(),
    )
    ''' 修改了这儿'''
    embedding = QianfanEmbeddingsEndpoint().embed_query("cat")
    response = vectorstore.similarity_search_by_vector(embedding)
    print('response: ', response)
    '''
    输出 response:  [Document(metadata={'source': 'mammal-pets-doc'}, page_content='Cats are independent pets that often enjoy their own space.'), 
    Document(metadata={'source': 'mammal-pets-doc'}, page_content='Rabbits are social animals that need plenty of space to hop around.'), 
    Document(metadata={'source': 'mammal-pets-doc'}, page_content='Dogs are great companions, known for their loyalty and friendliness.'), 
    Document(metadata={'source': 'fish-pets-doc'}, page_content='Goldfish are popular pets for beginners, requiring relatively simple care.')]
    '''


if __name__ == '__main__':
    # test01_vector_store_embedding_similarity_search()
    # asyncio.run(test02_vector_store_embedding_asimilarity_search())
    # test03_vector_store_embedding_similarity_search_with_score()
    test04_vector_store_embedding_similarity_search_by_vector()

import os

from langchain_core.documents import Document

# langSmith，服务的 API key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'


def test_document():
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
    print(documents[0])  # page_content='Dogs are great companions,。。。
    print(type(documents[0]))  # <class 'langchain_core.documents.base.Document'>
    print(documents[0].page_content)  # Dogs are great companions, known for their loyalty and friendliness.
    print(type(documents[0].page_content))  # <class 'str'>
    print(documents)  # [Document(metadata={'source': 'mammal-pets-doc'}, page_con。。。。。
    print(type(documents))  # <class 'list'>


if __name__ == '__main__':
    '''
    实例化 langchain 中 构建向量存储中 概念 document(文档)
    '''
    test_document()
    pass

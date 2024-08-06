import os

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.llms.openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate

# 千帆的 API key 和 Secret key
os.environ["QIANFAN_AK"] = "Z1Os3TDFDM7zPDOjspytwEjD"
os.environ["QIANFAN_SK"] = "nkMXxCV0ulG0hVWu7YOCMkHURP9d7tGM"

from utils import docs_split_retrieval


def test01():
    retriever = docs_split_retrieval()

    # llm = QianfanChatEndpoint(
    #     stream=True,
    #     model='ERNIE-4.0-8K'
    # )

    llm = OpenAI(
        base_url='https://open.bigmodel.cn/api/paas/v4/',
        api_key=os.environ.get('CHATGLM_API_KEY'))

    '''
    2. Incorporate the retriever into a question-answering chain.
    '''
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
    print('response["answer"]: ', response["answer"])

    # response["answer"]:  Task Decomposition（任务分解）是将一个复杂的问题或任务细化为更小、更具体的子任务的过程。
    # 这些子任务通常更易于管理、分配和执行。通过任务分解，可以降低整体任务的复杂性，提高执行效率，并使得并行处理或团队协作成为可能。在实施任务分解时，
    # 关键是要确保子任务之间的独立性，以便它们可以并行执行，同时还要保证任务分解的均匀性，以充分利用系统资源。


if __name__ == '__main__':
    test01()

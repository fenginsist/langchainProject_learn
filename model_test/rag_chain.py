import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama, QianfanChatEndpoint
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 千帆的 API key 和 Secret key
os.environ["QIANFAN_AK"] = "Z1Os3TDFDM7zPDOjspytwEjD"
os.environ["QIANFAN_SK"] = "nkMXxCV0ulG0hVWu7YOCMkHURP9d7tGM"

# Prompt设定输出为中文，并且将其上下文设置为向量数据库中的内容
template = """
Answer the question based only on the following context, and output in Chinese:
{context}
Question: {question}
"""
'''导入本地文本'''
loader = TextLoader("test.txt", encoding='utf-8')
data = loader.load()

# 将文本分割成长度为200个字符的小块，并且每个小块之间有20个字符的重叠
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
all_splits = text_splitter.split_documents(data)


vectorstore_torist = Chroma.from_documents(all_splits, QianfanEmbeddingsEndpoint(), persist_directory="./vector_store")
retriever = vectorstore_torist.as_retriever()
prompt = ChatPromptTemplate.from_template(template)

'''

采用本地的大语言模型llama3对话
疑问：
为什么这样调用就是本地的？
执行这个文件需要启动llama吗？   答：需要启动llama大模型。因为，本地不跑，调用找不到；启动 llama 命令：ollama run ollama3 或者 ollama serve
如何调用外部 云的LLM？
如果调用部署在Linux上的 LLM？

以上都是需要解决的点。
'''
ollama_llm = "llama3"
# 本地 LLM
model_local = ChatOllama(model=ollama_llm)
# 百度千帆的
# model_local = QianfanChatEndpoint(
#     streaming=True,
#     model="ERNIE-4.0-8K"  # ERNIE-Bot
# )

# Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model_local
    | StrOutputParser()
)

response = chain.invoke("预付购房款证明是什么")   # test.txt文件中内容
print('response:', response)

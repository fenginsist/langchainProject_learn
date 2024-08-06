#!/usr/bin/env python
import os
from typing import List

from fastapi import FastAPI
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

# 千帆的 API key 和 Secret key
os.environ["QIANFAN_AK"] = "Z1Os3TDFDM7zPDOjspytwEjD"
os.environ["QIANFAN_SK"] = "nkMXxCV0ulG0hVWu7YOCMkHURP9d7tGM"

# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 2. Create model
model = QianfanChatEndpoint(
    streaming=True,
    model="ERNIE-4.0-8K"  # ERNIE-Bot
)

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser


# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route

add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
    # 进入：http://localhost:8000/chain/playground/ ，可以进行对话
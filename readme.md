# 1. 版本
通过命令
```bash
pipreqs ./
```
pip install pipreqs

如果没有 `pipreqs` 命令安装需要先安装
```bash
pip install pipreqs
```
生成 requirements.txt 版本依赖文件
# 2. 第一部分 base
## 1.1 Build a Simple LLM Application with LCEL
**langchain 开发文档 code demo链接：https://python.langchain.com/v0.2/docs/tutorials/llm_chain/ ，
这个章节的主题是:Build a Simple LLM Application with LCEL,使用LCEL构建一个简单的LLM应用**

1. `01_langchain_bug.py`： 根据langchain 集成 chatglm文档：https://python.langchain.com/v0.2/docs/integrations/llms/chatglm/
    出问题：根据问题显示说需要本地跑大模型。所以次方法不痛
    这个文档描述的是 langchain 集合本地的 chatglm 大模型，所以目前不适用
2. `02_chatGLM3.py`: 根据智谱清言的对外开发文档的测试，文档：https://open.bigmodel.cn/dev/api#overview
    Python SDK 创建 Client
    同步调用: 调用后即可一次性获得最终结果，
    异步调用: 调用后会立即返回一个任务 ID，然后用任务ID查询调用结果（根据模型和参数的不同，通常需要等待10-30秒才能得到最终结果），
    SSE 调用: 调用后可以流式的实时获取到结果直到结束，Python
3. `03_ThirdPlatform-opanAISDK.py`: 第三方的平台
    OpenAI SDK
    文档：https://open.bigmodel.cn/dev/api#openai_sdk
   `03_ThirdPlatform-LangChainSDK.py`
    LangChain SDK
    文档：https://open.bigmodel.cn/dev/api#langchain_sdk

上面使用的智谱清言的chatglm模型，需要注册API_KEY，注册平台：https://open.bigmodel.cn/dev/howuse/introduction ，点击右上角登录注册即可。

---------------------------------------------------------------从这里才开始 使用 langchain 调用 LLM（外部百度千帆）
**langchain 开发文档 code demo链接：https://python.langchain.com/v0.2/docs/tutorials/llm_chain/ ，
这个章节的主题是:Build a Simple LLM Application with LCEL,使用LCEL构建一个简单的LLM应用**

4. `04_langchain_Qianfan_baidu.py`
    langchan 调用 百度千帆的LLM
    test1_invoke()： invoke()同步函数; 最简单的调用LLM，生成问题
    test2_ainvoke_batch()：ainvoke()执行异步函数;，有小坑，仔细看执行方法 batch()函数，返回:[AIMessage对象]
    test3_stream()：逐步获取生成的文本内容。这种方法特别适用于生成长文本或需要实时输出的场景，比如聊天机器人、内容生成工具等。
        方法作用：
            1、逐步生成输出：stream 方法用于逐步输出语言模型生成的内容，而不是一次性返回整个结果。这样可以实现更快的响应时间和更好的用户体验。
            2、 适用于流式输出场景： 在生成长文本或连续对话时，stream 可以实时返回生成的部分内容，使得用户无需等待完整生成，便可以开始阅读或进行交互。
            3、节省资源： 通过逐步输出，stream 方法可以减少内存占用和处理时间，尤其在生成长篇文本时，逐步处理更为高效。
        使用场景：
            1、聊天机器人：流式输出可以使用户在输入问题后立即看到部分响应，而不是等待模型生成完整的答案。
            2、长文本生成：如小说或文章生成，用户可以实时查看内容的进展。
            3、逐步反馈系统：在需要不断获取反馈的系统中，stream 方法可以及时返回当前状态或部分结果。
    test4_use_other_model()：百度千帆有很多模型，换一个模型。
5. `05_langchain_outputParsers.py`
    test01_outputParsers(): 对结果进行输出解析
        使用了三方方法输出
        1、直接invoke（）
        2、使用outputparsers
        3、链式输出，成为一个chain对象，最后invoke 消息（message）。
6. `06_langchain_promptTemplate.py`
    test01_promptTemplate()：对消息列表 message 使用提示模板进行重新构造
    之前把问题传进入直接invoke(message)，
    现在 使用 promptTemplate进行改造，形成一个模板
7. `07_langchain_LCEL_chain.py`
    使用LCEL的管道符 | 对整个流程 进行简化
    链式输出：chain = 消息模板|模型|解析 输出
    chain.invoke({参数})
8. `08_langchain_serve.py`
    整合以上流程，搭建本地服务，使用 langserve
    08_langchain_client.py
    为serve的客户端。


## 1.2 构建对话机器人
**langchain 开发文档 code demo链接：https://python.langchain.com/v0.2/docs/tutorials/chatbot/ ，
这个章节的主题是:Build a Chatbot**
1. `01_langchain_history.py`
    test1_HumanMessage()：连续问两次，连续invoke 两次
    test2(): 直接在invoke中，加入列表，里面放多个 HumanMessage，并且可以制定机器人说的话，使用AIMessage，代表机器人说的话
    get_session_history(): 官网开发文档提供的，获取历史信息的函数，很简单，就是将信息存在一个 store={} 字典中。
    test3_history_memory()：重点***，使用了 RunnableWithMessageHistory函数，可以对每一个对话设置一个session_id
        这个方法对对话进行了测试，很成功。
2. `02_langchain_prompt_template.py`
    test1_prompt_template()：第一次在消息模板中使用 参数占位符。
        使用了模型、提示模板、还对模型版使用了参数占位符，然后使用链式调用，最后通过chain.invoke()对模板参数设置值，这里参数是 HumanMessage消息。
    test2_prompt_template_with_message_history():
        在第一个方法的基础上，增加了记录历史消息的功能
        流程如下：
        1、创建model
        2、创建模板，带参
        3、链式调用： chain = prompt|model
        4、with_message_history = 调用历史消息类（参数就是chain等）
        5、使用 with_message_history 对模板参数传参
    test3_prompt_template_with_message_history_add_completed():
        在第二个的基础上，多对话了两次。
3. `03_manager_history_split.py`
    test1_manager_history():
        第一次使用：trim_messages()函数，里面放了模型，设置了对话的属性，然后 trimmer.invoke(message)。
        trim_messages():是一个用于消息修剪或裁剪的函数或方法，主要用于确保输入的消息列表或对话不会超过模型处理能力的最大 token 数量。
        很简单，就是 message定义了五段对话，使用了SystemMessage、HumanMessage、AIMessage，指定了回的话
    test2_manager_history():
        第一次使用了

## 1.3 Vector stores and retrievers
**langchain 开发文档 code demo链接：https://python.langchain.com/v0.2/docs/tutorials/retrievers/ ，
这个章节的主题是:Vector stores and retrievers**

**这里主要是讲 向量存储和检索，前两个py里是没有说到LLM，也就是单纯的把文件放到向量存储里，然后进行检索。
最后py 使用到了LLM，也就是先检索出来，然后在经过LLM过滤一下。**

1. `01_langchain_retrieval_document.py`
    test_document(): 实例化了一个包含5个document的对象
           第一次使用 Document
2. `02_langchain_retrieval_vector_store_embedding.py`
    test01_vector_store_embedding_similarity_search():
        第一次使用 vectorstore = Chroma.from_documents() 和 嵌入向量，我这里使用的是 QianfanEmbeddingsEndpoint，百度千帆的
        第一次使用 vectorstore.similarity_search("cat")，去搜索相似的。
    test02_vector_store_embedding_asimilarity_search():
        await vectorstore.asimilarity_search("cat")
        和第一个方法一样，只不过换成了异步的方法：
    test03_vector_store_embedding_similarity_search_with_score():
        response = vectorstore.similarity_search_with_score("cat")
        仅仅是换了一个带分数的检索函数
    test04_vector_store_embedding_similarity_search_by_vector():
        embedding = QianfanEmbeddingsEndpoint().embed_query("cat")
        response = vectorstore.similarity_search_by_vector(embedding)
        把检索的字符 转化为向量，在使用 vectorstore 去检索，得到输出。
3. `03_langchain_retrieval.py`
    test01_retrieval_RunnableLambda():
        retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)
        response = retriever.batch(["cat", "shark"])
        第一次使用 RunnableLambda() 函数，接受 vectorstore.similarity_search 参数，
    test02_retrieval_as_retriever():
        retriever = vectorstore.as_retriever(): 
        使用 向量存储 直接生成检索器
    test03_retrieval_QianfanLLM_ChatPromptTemplate(): 
        前两个方法是没有用到大模型的
        第三个方法使用大模型，
        加上LLM、提示模板
        chain = 
## 1.4 agent代理，因为技术封锁，拿不到API_KEY



# 2. Working with external knowledge

## 2.1 Build a Retrieval Augmented Generation (RAG) App
**langchain 开发文档 code demo链接：https://python.langchain.com/v0.2/docs/tutorials/rag/
这个章节的主题是:Build a Retrieval Augmented Generation (RAG) App，构建检索增强生成（RAG）APP**

1. `01_rag_preview.py`
    是一个整体结构的预览
    1、从一个网址趴下来数据 作为 docs，通过 RecursiveCharacterTextSplitter 进行分割，再通过 Chroma 生成 vectorstore，进而 retriever
    2、LLM
    3、rag_chain = 参数{***} | prompt | LLM | parser ；注意：prompt = hub.pull("rlm/rag-prompt") 这是啥我不懂。
    hub.pull("rlm/rag-prompt")
        解释：
        是用来从一个集中式的模型仓库中拉取（下载）特定的提示模板或模型，hub 是用于从一个模型或提示仓库中下载资源的方法。
        hub 可以是 LangChain 自身提供的某种工具，也可能是一个自定义或第三方库中的工具。
        "rlm/rag-prompt"：是一个字符串标识符，用于指定要拉取的资源
下面的其实都是对上面分步详细解说

2. `02_rag_index_load.py`
    test01_index_load(): 索引，加载文件，这里还是从网址上爬上来的
3. `03_rag_index_split.py`
    针对上一个py，新增 test02_index_split()函数
    text_splitter = RecursiveCharacterTextSplitter()
    all_splits = text_splitter.split_documents(docs)
    对加载的数据进行分割
4. `04_rag_index_vector_store.py`
    针对上一个py，新增 test03_index_store()函数
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=QianfanEmbeddingsEndpoint())
    生成向量存储
5. `05_rag_index_retrieve.py`
    针对上一个py，新增 test04_index_retrieve()函数
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    根据向量存储 生成 检索器
6. `06_rag_index_generate.py`
    针对上一个py，新增 test04_index_retrieve()函数
    单独的，和上面代码无关系
    生成 prompt和 message
7. `07_rag_index_generate_2.py`
    针对上一个py，新增 test06_index_generate_2()函数
    使用链式法则，chain = 参数{检索器，问题} | prompt | llm | parser
    然后通过 chain.stream(问题) 遍历回答即可啦

但是这里报错了，不知道为什么，等后面再回过头来研究
**报错信息：TypeError: Additional kwargs key completion_tokens already exists in left dict and value has unsupported type <class 'int'>.**


















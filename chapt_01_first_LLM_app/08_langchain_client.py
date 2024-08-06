from langserve import RemoteRunnable


if __name__ == '__main__':
    '''
    首先要启动 08_langchain_serve.py，再执行这个py文件。
    '''
    remote_chain = RemoteRunnable("http://localhost:8000/chain/")
    res = remote_chain.invoke({"language": "italian", "text": "hi"})
    print(res)      # 'Ciao'
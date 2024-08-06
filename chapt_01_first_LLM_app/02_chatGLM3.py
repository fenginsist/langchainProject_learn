from zhipuai import ZhipuAI
import os

os.environ['CHATGLM_API_KEY'] = '09accc9c442473f6029f0062d1cd9411.cAbNNUUldblEFrAo'


class ChatGLM3:
    def __init__(self):
        pass

    def pythonSdkCreateClient(self, chatglm_api_key):
        '''
        Python SDK 创建 Client
        :param chatglm_api_key:
        :return:
        '''
        client = ZhipuAI(api_key=chatglm_api_key)  # 请填写您自己的APIKey
        response = client.chat.completions.create(
            model="glm-4",  # 填写需要调用的模型名称
            messages=[
                {"role": "user", "content": "你好！你叫什么名字"},
            ],
            stream=True,                                                    # 比下面多了个这个
        )
        for chunk in response:
            print(chunk.choices[0].delta)

    def synchronousCall(self, chatglm_api_key):
        '''
        同步调用: 调用后即可一次性获得最终结果
        :param chatglm_api_key:
        :return:
        '''
        client = ZhipuAI(api_key=chatglm_api_key)  # 填写您自己的APIKey
        response = client.chat.completions.create(
            model="glm-4",  # 填写需要调用的模型名称
            messages=[
                {"role": "user", "content": "作为一名营销专家，请为我的产品创作一个吸引人的slogan"},
                {"role": "assistant", "content": "当然，为了创作一个吸引人的slogan，请告诉我一些关于您产品的信息"},
                {"role": "user", "content": "智谱AI开放平台"},
                {"role": "assistant", "content": "智启未来，谱绘无限一智谱AI，让创新触手可及!"},
                {"role": "user", "content": "创造一个更精准、吸引人的slogan"}
            ],
        )
        print(response.choices[0].message)

    def asynchronousCall(self, chatglm_api_key):
        '''
        异步调用: 调用后会立即返回一个任务 ID，然后用任务ID查询调用结果（根据模型和参数的不同，通常需要等待10-30秒才能得到最终结果）
        :param chatglm_api_key:
        :return:
        '''
        client = ZhipuAI(api_key=chatglm_api_key)  # 请填写您自己的APIKey
        response = client.chat.asyncCompletions.create(                                     # asyncCompletions，调用了异步方法
            model="glm-4",  # 填写需要调用的模型名称
            messages=[
                {
                    "role": "user",
                    "content": "请你作为童话故事大王，写一篇短篇童话故事，故事的主题是要永远保持一颗善良的心，要能够激发儿童的学习兴趣和想象力，同时也能够帮助儿童更好地理解和接受故事中所蕴含的道理和价值观。"
                }
            ],
        )
        print(response)

    def SSECall(self, chatglm_api_key):
        '''
        SSE 调用: 调用后可以流式的实时获取到结果直到结束
        :param chatglm_api_key:
        :return:
        '''
        client = ZhipuAI(api_key=chatglm_api_key)  # 请填写您自己的APIKey
        response = client.chat.completions.create(
            model="glm-4",  # 填写需要调用的模型名称
            messages=[
                {"role": "system",
                 "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
                {"role": "user",
                 "content": "我对太阳系的行星非常感兴趣，特别是土星。请提供关于土星的基本信息，包括其大小、组成、环系统和任何独特的天文现象。"},
            ],
            stream=True,                                                                     # stream=True
        )
        for chunk in response:
            print(chunk.choices[0].delta)


if __name__ == '__main__':
    '''
    调用智谱清言的接口，与langchain无关。
    '''
    chatglm_api_key = os.environ.get('CHATGLM_API_KEY')
    chat = ChatGLM3()
    # chat.pythonSdkCreateClient(chatglm_api_key)   # Python SDK 创建 Client
    # chat.synchronousCall(chatglm_api_key)         # 同步调用: 调用后即可一次性获得最终结果
    # chat.asynchronousCall(chatglm_api_key)        # 异步调用: 调用后会立即返回一个任务 ID，然后用任务ID查询调用结果
    chat.SSECall(chatglm_api_key)                   # SSE 调用: 调用后可以流式的实时获取到结果直到结束,一直执行

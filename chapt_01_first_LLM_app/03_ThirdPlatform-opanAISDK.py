from openai import OpenAI
import os

os.environ['CHATGLM_API_KEY'] = '09accc9c442473f6029f0062d1cd9411.cAbNNUUldblEFrAo'


if __name__ == '__main__':
    '''
    第三方框架-OpenAI SDK 使用
    '''
    client = OpenAI(
        api_key=os.environ.get('CHATGLM_API_KEY'),
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )

    completion = client.chat.completions.create(
        model="glm-4",
        messages=[
            {"role": "system", "content": "你是一个聪明且富有创造力的小说作家"},
            {"role": "user",
             "content": "请你作为童话故事大王，写一篇短篇童话故事，故事的主题是要永远保持一颗善良的心，要能够激发儿童的学习兴趣和想象力，同时也能够帮助儿童更好地理解和接受故事中所蕴含的道理和价值观。"}
        ],
        top_p=0.7,
        temperature=0.9
    )

    print(completion.choices[0].message)

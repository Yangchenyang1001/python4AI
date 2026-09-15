"""
  @Author:桌角是小黑
  @Time:2026/9/14
  @Desc:
"""
# pip install openai
from openai import OpenAI
import os
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()
client = OpenAI(
	base_url=os.getenv("OPENAI_BASE_URL"), # 平台提供的 URL,默认可省略
	api_key=os.getenv("OPENAI_API_KEY"), # 平台提供的 API-Key,默认可省略
)
# 创建LLM模型
completion = client.chat.completions.create(
	model="gpt-4o-mini", # 模型名称
	messages=[{"role": "user", "content": "将'你好'翻译成意大利语"}], # 用户输入
)
# 输出模型返回结果
print(completion)
print(completion.choices[0].message.content)
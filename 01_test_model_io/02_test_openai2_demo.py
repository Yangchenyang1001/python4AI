"""
  @Author:桌角是小黑
  @Time:2026/9/14
  @Desc:
"""
import os
from dotenv import load_dotenv

# pip install openai
# 加载环境变量
load_dotenv()
from openai import OpenAI
client = OpenAI()
response = client.responses.create(
    model="gpt-4o-mini",
    input="中国国内今天发生了哪些大事儿？",
    tools=[{"type": "web_search"}] # 可以自动调用内置工具
)
print(response.output_text)

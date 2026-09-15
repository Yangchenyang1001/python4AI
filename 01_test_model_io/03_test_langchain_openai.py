"""
  @Author:桌角是小黑
  @Time:2026/9/14
  @Desc: 输入提示词的各种格式
"""
import os

from dotenv import load_dotenv


load_dotenv()

from langchain_openai import ChatOpenAI
client = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
# 方式一 字符串传参
resp = client.invoke("你好")
# 方式二 对象传参
from langchain_core.messages import SystemMessage, HumanMessage
resp = client.invoke([SystemMessage(content="你是一个专业的数学助手"), HumanMessage(content="你好，你是谁")])
# 方式三 列表传参
messages_list = [("system", "你是一个专业的数学助手"), ("user", "你好，你是谁")]
resp = client.invoke(messages_list)
# 方式四  字典传参
message_list = [{"role": "system", "content": "你是一个专业的数学助手"}, {"role": "user", "content": "你好，你是谁"}]
resp = client.invoke(messages_list)


print(resp.content)

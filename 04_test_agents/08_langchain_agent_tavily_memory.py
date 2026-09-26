"""
  @Author:桌角是小黑
  @Time:2026/9/20
  @Desc:
"""
import os
import time

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# pip install langgraph
import os
import datetime
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
api_key=os.getenv("DEEPSEEK_API_KEY")
print(api_key)
# 定义 Tavily 搜索工具
search = TavilySearch(max_results=5)
tools = [search]

llm = ChatOpenAI(
    model="deepseek-chat",  # 指定 DeepSeek 的模型名称
    api_key=api_key,
    base_url="https://api.deepseek.com", # 核心：将 base_url 指向 DeepSeek 的接口
    temperature=0.7
)
# 关键点 1：定义 checkpointer 实例
checkpointer = InMemorySaver()  # 注意是 Saver 不是 Server 内存被打爆、不安全、token数量爆了

# 创建 Agent
agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,  # 关键点 2：将 checkpointer 实例传递给 Agent
)

# 调用
print("=== 第一次调用 ===")
for chunk in agent.stream(
        input={
            "messages": [
                {
                    "role": "system",
                    "content": f"当前时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                },
                {"role": "user", "content": "今天北京天气怎么样？"},
            ]
        },
        config={"configurable": {"thread_id": "abc123"}},  # 关键点 3：为每个调用指定一个唯一的 thread_id
):
    print(chunk, end="\n\n")

print("=== 第二次调用 ===")
for chunk in agent.stream(
        input={
            "messages": [
                {"role": "user", "content": "我刚才问你什么了"},
            ]
        },
        # 关键点 4：在多次调用中使用相同的 thread_id，模型会记住之前的对话
        config={"configurable": {"thread_id": "abc123"}},
):
    print(chunk, end="\n\n")

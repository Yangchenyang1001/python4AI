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

# 1. 创建智能体
# from langchain_ollama import ChatOllama
# llm = ChatOllama(
#     model="phi4:latest", base_url="http://localhost:11434",
# )
# llm = ChatOpenAI(
#     model="phi4:latest",  # 这里填你本地拉取的模型名
#     base_url="http://localhost:11434/v1", # 核心：必须加上 /v1
#     api_key="ollama",  # Ollama 兼容接口不需要真实密钥，随便填一个字符串即可
#     temperature=0
# )

load_dotenv()
api_key=os.getenv("DEEPSEEK_API_KEY")
print(api_key)
llm = ChatOpenAI(
    model="deepseek-chat",  # 指定 DeepSeek 的模型名称
    api_key=api_key,
    base_url="https://api.deepseek.com", # 核心：将 base_url 指向 DeepSeek 的接口
    temperature=0.7
)

print(os.getenv("TAVILY_API_KEY"))  # 打印出来看看是不是 None
# 2. 定义工具
search = TavilySearch(max_results=5)
tools = [search]  # 工具列表

# 3. 创建代理 智能体
agent = create_agent(model=llm, tools=tools, system_prompt="你是一个会调用工具的助手，会回答问题")

# 4. 调用信息
# res = agent.invoke(
#     {"messages": [{"role": "user", "content": "今天北京的天气怎么样？"}]}
# )
# print(res)
# print(res["messages"][-1].content)

# 4. 调用 Agent 采用流式调用
for chunk in agent.stream(
    {
    	"messages": [
            {"role": "system", "content": "你是位助手，需要调用工具来帮助用户。"},
            {"role": "user", "content": "今天北京的天气怎么样？"},
        ]
    },
    stream_mode="messages"
):
    print(chunk[0].content, end="", flush=True)
    time.sleep(0.1)


"""
  @Author:桌角是小黑
  @Time:2026/9/23
  @Desc:
"""
import os

from dotenv import load_dotenv
# pip install langchain_mcp_adapters
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
import asyncio

client = MultiServerMCPClient(
    {
        "mcp_tool_12306": {
            "transport": "streamable_http",
            "url": "https://mcp.api-inference.modelscope.net/0fa3def2aa224f/mcp"
        }
        # "amap-maps": {
        # "transport": "streamable_http",
        # "url": "https://mcp.api-inference.modelscope.net/14db9c03451e47/mcp"
        # }
    }
)


def print_message(message):
    from langchain.messages import AIMessage, HumanMessage, ToolMessage
    if isinstance(message, AIMessage):
        print("AI 回复：", message.content)
        print("AI 决定调用工具", message.tool_calls)
    elif isinstance(message, HumanMessage):
        print("用户输入：", message.content)
    elif isinstance(message, ToolMessage):
        print("工具调用：", message.content)
    else:
        print("未知消息类型")


async def main():
    tools = await client.get_tools()
    print(tools)

    def handle_tool_error(error) -> str:
        return f"Tool execution failed: {str(error)}"

    # 配置工具错误所对应的处理方式，避免工具调用错误导致整个 agent执行过程退出
    for tool in tools:
        tool.handle_tool_error = handle_tool_error

    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    print(api_key)
    llm = ChatOpenAI(
        model="deepseek-chat",  # 指定 DeepSeek 的模型名称
        api_key=api_key,
        base_url="https://api.deepseek.com",  # 核心：将 base_url 指向 DeepSeek 的接口
        temperature=0.7
    )

    agent = create_agent(
        llm,
        tools,
    )
    while True:
        user_input = input(">")
        if user_input == "exit":
            break
        res = await agent.ainvoke({"messages": [{"role": "user", "content": user_input}]},
                                  config={"configurable": {"thread_id": "1"}})
        print(res['messages'][-1].content, end="\n\n")
        for message in res["messages"]:
            print_message(message)
            print("\n")
        print("\n")


asyncio.run(main())

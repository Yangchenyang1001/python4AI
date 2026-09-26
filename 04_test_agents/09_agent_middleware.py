import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

load_dotenv()
# 1. 定义工具
@tool
def get_weather(city: str) -> str:
    """查询天气"""
    return f"{city}的天气晴朗，气温 25 度。"


@tool
def transfer_money(amount: int, to_account: str) -> str:
    """转账工具 (敏感操作)"""
    print(f"!!! 正在执行转账: {amount} -> {to_account} !!!")
    return f"成功转账 {amount} 元给 {to_account}。"


# 2. 初始化模型

llm = ChatOpenAI(
    model="deepseek-chat",  # 指定 DeepSeek 的模型名称
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com", # 核心：将 base_url 指向 DeepSeek 的接口
    temperature=0.7
)
# llm = init_chat_model(
#     model="deepseek-chat",  # 指定 DeepSeek 的模型名称
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com", # 核心：将 base_url 指向 DeepSeek 的接口
#     temperature=0.7
# )
# 3. 配置 HumanInTheLoopMiddleware
# 我们希望在调用 transfer_money 时暂停，让用户审核
# True 表示允许所有操作 (approve, edit, reject)
hitl_middleware = HumanInTheLoopMiddleware(interrupt_on={
    "transfer_money": True,
    "get_weather": False  # False 表示自动批准，不中断
})
# 4. 创建 Agent
# 注意：使用中断功能必须配置 checkpointer，因为中断需要保存状态
checkpointer = InMemorySaver()
agent = create_agent(
    model=llm,
    tools=[get_weather, transfer_money],
    middleware=[hitl_middleware],
    checkpointer=checkpointer,
)


async def run_demo():
    print("=== HumanInTheLoopMiddleware 演示 ===")
    thread_id = "thread-1"
    config = {"configurable": {"thread_id": thread_id}}
    current_input = {"messages": [{"role": "user", "content": "帮我转账给张三，100元"}]}
    current_command = None

    while True:
        try:
            # 如果有恢复指令就用恢复指令，否则用用户输入
            if current_command:
                result = await agent.ainvoke(current_command, config=config)
                current_command = None
            else:
                if not current_input:
                    break
                result = await agent.ainvoke(current_input, config=config)
                current_input = None

            # 正常打印 Agent 返回的消息
            if "messages" in result:
                for msg in result["messages"]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        print(f"[Agent]: 我想调用工具: {msg.tool_calls}")
                    if msg.type == "tool":
                        print(f"[Tool Output]: {msg.content}")
                    if msg.type == "ai" and not msg.tool_calls:
                        print(f"[Agent]: {msg.content}")

            # 任务正常结束，退出循环
            break

        except GraphInterrupt as e:
            # 2. 单独捕获中断异常，不再被 try 吞掉报错
            print(f"\n!!! 检测到中断 (Middleware 拦截) !!!")

            # 3. 从异常对象 e 中提取中断详情
            interrupt_payload = e.value
            print(f"中断详情: {interrupt_payload}")

            # 模拟用户决策 (实际中这里可以用 input())
            print("\n[System]: 请审核上述操作 (approve/reject/edit):")
            decision_type = "approve"
            print(f"[User]: {decision_type}")

            # 4. 构造恢复执行的 Command
            # 根据 Middleware 要求构造 decisions
            decisions = [{"type": decision_type}]
            resume_payload = {"decisions": decisions}

            current_command = Command(resume=resume_payload)
            print("[System]: 已提交审核结果，准备恢复执行...")

        except Exception as e:
            # 5. 兜底：打印其他真正的错误
            print(f"发生未知错误: {e}")
            break
if __name__ == "__main__":
    asyncio.run(run_demo())

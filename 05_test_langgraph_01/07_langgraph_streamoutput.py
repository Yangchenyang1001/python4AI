import operator
import os
import time
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
from jedi.inference.gradual.typing import TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

# 定义大模型
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
print(api_key)
llm = ChatOpenAI(
    model="deepseek-chat",  # 指定 DeepSeek 的模型名称
    api_key=api_key,
    base_url="https://api.deepseek.com",  # 核心：将 base_url 指向 DeepSeek 的接口
    temperature=0.7
)


# 定义状态图
class State(TypedDict):
    input: str
    messages: Annotated[List[BaseMessage], operator.add]
    current_step: str


# 定义节点

def node_input(state: State):
    """接收用户输入"""
    input = state["input"]
    return {
        "messages": [HumanMessage(content=input)],
        "current_step": "接收用户输入内容"
    }


# 定义节点
def node_processing(state: State, runtime: Runtime):
    """,

    模拟中间处理过程,并使用writer输出自定义流式数据
    对应stream_mode="custom

    """
    steps = ["正在分析意图...", "正在检索知识库...", "正在构建Prompt..."]
    writer = runtime.stream_writer

    for i, step in enumerate(steps):
        time.sleep(0.5)  # 模拟耗时操作

        # 使用writter发送自定义数据 不影响图的状态
        # 这些数据只通过stream_mode="custom" 接收到

        writer({
            "step_index": i + 1,
            "description": step,
            "timestamp": time.time()
        })
    return {"current_step": "处理完成"}


def node_gen(state: State):
    """
    llm生成答案 stream_mode="messages" 自动捕获llm的流式输出
    """
    response = llm.invoke(state["messages"])

    return {
        "messages": [response],
        "current_step": "生成成功"
    }


def build_graph():
    graph = StateGraph(State)

    graph.add_node("input", node_input)
    graph.add_node("process", node_processing)
    graph.add_node("generate", node_gen)

    graph.add_edge(START, "input")
    graph.add_edge("input", "process")
    graph.add_edge("process", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def demo_langgraph():
    initial_state = {"input": "你是谁", "message": [], "current_step": "start"}
    compile_graph = build_graph()
    print(f"\n{'=' * 20} 1. Mode: values {'=' * 20}")
    # 1. mode:values 输出完整状态 每执行一个节点,输出当前完整的state
    # for event in compile_graph.stream(initial_state, stream_mode="values"):
    #     print(f"State {event}")

    # 2. Mode: updates (输出增量更新) 输出该节点返回的增量数据
    print(f"\n{'=' * 20} 2. Mode: updates {'=' * 20}")
    # print("描述: 每执行完一个节点，输出该节点返回的增量数据")
    # for event in compile_graph.stream(initial_state, stream_mode="updates"):
    #     print(f"Update: {event}")


    # 3. Mode: custom (输出自定义数据)  仅输出节点内部通过 writer() 发送的数据
    print(f"\n{'=' * 20} 3. Mode: custom {'=' * 20}")
    # print("描述: 仅输出节点内部通过 writer() 发送的数据")
    # for event in compile_graph.stream(initial_state, stream_mode="custom"):
    #     print(f"Custom Data: {event}")

    # 4. Mode: messages (输出 LLM Token)
    print(f"\n{'=' * 20} 4. Mode: messages {'=' * 20}")
    # print("描述: 输出 LLM 生成的消息片段 (Token)")
    # for chunk, metadata in compile_graph.stream(initial_state, stream_mode="messages"):
    #     node_name = metadata.get('langgraph_node', 'unknown')
    #     # 打印 Token 内容，模拟打字机效果
    #     print(f"[{node_name}] Token: {chunk.content!r}")
    #     time.sleep(0.1)  # 仅用于演示视觉效果
    # 5. Mode: debug (调试模式)
    # print(f"\n{'=' * 20} 5. Mode: debug {'=' * 20}")
    # print("描述: 输出所有详细的执行信息")
    # count = 0
    # for event in compile_graph.stream(initial_state, stream_mode="debug"):
    #     if count < 3:  # 仅演示前几条
    #         print(f"Debug Event: {event['type']} - {event.get('payload', {}).get('name')}")
    #     count += 1
    # print("... (省略后续 debug 信息)")

    # 6. Mixed Mode (混合模式)
    print(f"\n{'=' * 20} 6. Mixed Mode {'=' * 20}")
    print("描述: 同时获取 updates 和 custom 数据")
    for mode, data in compile_graph.stream(initial_state, stream_mode=["updates", "custom"]):
        if mode == "updates":
            print(f"[Updates] 来自节点 {list(data.keys())[0]}")
        elif mode == "custom":
            print(f"[Custom] {data['description']}")
if __name__ == '__main__':
    demo_langgraph()

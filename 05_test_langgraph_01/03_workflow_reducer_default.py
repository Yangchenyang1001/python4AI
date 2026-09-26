"""
  @Author:桌角是小黑
  @Time:2026/9/25
  @Desc:
"""

"""
LangGraph Reducer函数演示 - 默认Reducer（覆盖更新）
"""

from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

# 1. 定义图状态 - 所有字段均未指定reducer，故默认为覆盖
class AgentState(TypedDict):
    query: str
    messages: List[BaseMessage]  # 历史消息列表
    search_results: List[str]    # 搜索结果列表

# 2. 节点函数：输入处理
def input_node(state: AgentState):
    query = state["query"]
    new_message = HumanMessage(content=f"用户问题: {query}")

    # 返回新列表，将完全替换掉state["messages"]的旧列表
    return {"messages": [new_message]}

# 3. 节点函数：搜索处理
def search_node(state: AgentState):
    query = state["query"]
    query_result = f"模拟搜索结果：关于'{query}'的相关信息..."
    search_msg = AIMessage(content=f"搜索完成: {query_result}")

    # 返回新列表，将完全替换掉state["search_msg"]和state["query_result"]的旧列表
    return {
        "messages": [search_msg],
        "search_results": [query_result]
    }

# 4. 节点函数：生成最终回复
def response_node(state: AgentState):
    context = state["search_results"] # 此时search_results只包含上一个节点的单条结果
    query = state["query"]
    final_response = f"基于搜索结果，对问题'{query}'的回答是：{context} "
    ai_response = AIMessage(content=final_response)
    return {
        "messages": [ai_response], # 再次覆盖整个messages列表
        "search_results": [f"最终回复已生成"] # 覆盖整个search_results列表
    }

# 5. 构建图
def build_graph():
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("input", input_node)
    workflow.add_node("search", search_node)
    workflow.add_node("response", response_node)

    # 设置边
    workflow.add_edge("input", "search")
    workflow.add_edge("search", "response")

    # 设置入口和出口
    workflow.set_entry_point("input")
    workflow.set_finish_point("response")

    return workflow.compile()

# 运行示例
if __name__ == "__main__":
    # 创建图实例
    app = build_graph()

    # 初始状态
    initial_state = {
        "query": "Python编程最佳实践",
        "messages": [SystemMessage(content="你是一个AI助手，正在进行信息检索任务。")],
        "search_results": []
    }

    # 执行
    result = app.invoke(initial_state)

    print("=== 最终状态 ===")
    print(f"查询: {result['query']}")

    print("\n=== 消息历史 ===")
    for i, msg in enumerate(result['messages']):
        print(f"[{i}] {msg.type}: {msg.content}")

    print("\n=== 搜索结果 ===")
    for i, res in enumerate(result['search_results']):
        print(f"[{i}] {res}")
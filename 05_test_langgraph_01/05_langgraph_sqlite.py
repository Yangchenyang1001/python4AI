"""
  @Author:桌角是小黑
  @Time:2026/9/25
  @Desc:
"""
"""
LangGraph 状态存储示例：从checkpointer中恢复状态
uv add langgraph-checkpoint-sqlite
"""
import os
import sqlite3
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import END
from langgraph.graph import START
from langgraph.graph import StateGraph


# 1. 构建图状态
class MyState(TypedDict):
    key_1: str
    key_2: str
    key_3: str


# 2. 定义节点函数
def node_1(state: MyState) -> MyState:
    print('node_1状态为', state)
    return {"key_1": "value_1"}


def node_2(state: MyState) -> MyState:
    print('node_2状态为', state)
    # raise Exception("模拟node_2节点报错")
    return {"key_2": "value_2"}


def node_3(state: MyState) -> MyState:
    print('node_3状态为', state)
    return {"key_3": "value_3"}


# 3. 构建图
def build_graph():
    graph = StateGraph(MyState)
    graph.add_node(node_1)
    graph.add_node(node_2)
    graph.add_node(node_3)

    graph.add_edge(START, "node_1")

    graph.add_edge("node_1", "node_2")
    graph.add_edge("node_1", "node_3")

    graph.add_edge("node_2", END)
    graph.add_edge("node_3", END)
    return graph


# 4. 主函数
def demo_langgraph():
    # 1. 构建Connection对象
    # database：指定数据库保存的位置

    # 创建sqlite数据库目录，存在就不创建
    os.makedirs("./sqlite_data", exist_ok=True)
    # check_same_thread=False 默认True
    # SQLite很"谨慎"，只允许创建它的那个线程使用它
    # 改成 False：允许其他线程也使用这个数据库连接
    # 因为LangGraph可能会在后台用不同的线程操作数据库，如果不改会报错
    conn = sqlite3.connect(database="./sqlite_data/langgraph_sqlite.db", check_same_thread=False)
    # 2. 通过connection对象构建checkpointer实例
    # uv add langgraph-checkpoint-sqlite
    # 长期记忆数据库
    checkpointer = SqliteSaver(conn)

    # 3. 构建图
    graph = build_graph()

    # 4. 编译图时传入 Checkpointer
    compiled_graph = graph.compile(checkpointer=checkpointer)

    # 5. 定义config
    config = {"configurable": {"thread_id": "a1"}}

    # 6. 调用时传入config
    # result = compiled_graph.invoke({}, config=config)

    # # 6. 从状态中恢复：传入None
    # input在断点续传的情况下入参是None，如果还包含实际信息则会重新走流程。
    result = compiled_graph.invoke(None, config=config)

    # 7. 打印结果
    print(result)
    # compiled_graph.get_graph().print_ascii()


if __name__ == "__main__":
    demo_langgraph()

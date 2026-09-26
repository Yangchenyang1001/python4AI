"""
  @Author:桌角是小黑
  @Time:2026/9/24
  @Desc:
"""
import time
from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph

"""
目标：通过问题搜索答案
1. 定义图状态
2. 定义节点函数
3. 通过状态创建图实例
4. 添加节点
5. 添加边
6. 编译图
7. 启动工作流
8. 输出结果

问题：大模型中的“幻觉”是什么意思？
"""


# 1. 定义状态图 实体类 给各个节点使用
class MyState(TypedDict):
    query: str  # 用户输入问题
    rag_result: str  # RAG检索结果
    web_search_result: str  # web联网查询结果
    final_result: str  # 两路检索合并最终 结果 rag_result + web_search_result


class InputState(TypedDict):
    query: str  # 用户输入问题


class OutputState(TypedDict):
    final_result: str  # 两路检索合并最终 结果 rag_result + web_search_result


# 2. 定义节点函数
# rag处理节点     ->MyState|dict 允许返回对象或字典
def rag_node(state: InputState) -> MyState | dict:
    query = state["query"]
    rag_result = f"RAG检索结果 {query}，从Milvus中检索一些数据...."
    # state["rag_result"] = rag_result
    # return state  # 返回一个全量的输入法   并行环境不允许
    return {"rag_result": rag_result}  # 返回一个增量数据  建议使用该方式


# web处理节点
def web_search_node(state: InputState) -> MyState | dict:
    query = state["query"]
    web_search_result = f"WEB联网结果 {query}，从tavily中获取一些数据...."
    return {"web_search_result": web_search_result}  # 返回一个增量数据  建议使用该方式


# 汇总节点
def final_answer_node(state: MyState) -> OutputState | dict:
    rag_result = state["rag_result"]
    web_search_result = state["web_search_result"]
    final_result = f"结果汇总 {rag_result}{web_search_result}，总结最终数据...."
    return {"final_result": final_result}  # 返回一个增量数据  建议使用该方式


# 3. 通过状态创建图实例
graph = StateGraph(state_schema=MyState, input_schema=InputState, output_schema=OutputState)
# 4. 添加节点
graph.add_node(rag_node)
graph.add_node(web_search_node)
graph.add_node(final_answer_node)

# 5. 添加边
graph.add_edge(START, "rag_node")
graph.add_edge(START, "web_search_node")
graph.add_edge("rag_node", "final_answer_node")
graph.add_edge("web_search_node", "final_answer_node")
graph.add_edge("final_answer_node", END)
# 6. 编译图
compile_graph = graph.compile()
# 7. 启动工作流
state = {"query": "万用表如何使用？"}
final_state: OutputState = compile_graph.invoke(state)
# 8. 输出结果
# 格式化json
# print(json.dumps(state, indent=4, ensure_ascii=False,cls=json.JSONEncoder))
print(final_state["final_result"])

# 9. 打印图(uv add grandalf)
# pip install grandalf
compile_graph.get_graph().print_ascii()
# structure = graph_structure.draw_ascii()
# print(structure)

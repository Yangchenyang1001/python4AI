"""
  @Author:桌角是小黑
  @Time:2026/9/25
  @Desc:
"""

import operator
from typing import Annotated

from langgraph.checkpoint.memory import InMemorySaver
from pip._internal.commands import configuration
from torch.distributed.checkpoint._experimental import checkpointer
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. 定义图状态
class State(TypedDict):
    aggregate: Annotated[list, operator.add]


# 2. 定义节点函数
def a(state: State, config):
    print(f'Adding "A" to {state["aggregate"]}')
    return {"aggregate": ["A"]}


def b(state: State, config):
    print(f'Adding "B" to {state["aggregate"]}')
    return {"aggregate": ["B"]}


def c(state: State, config):
    print(f'Adding "C" to {state["aggregate"]}')
    return {"aggregate": ["C"]}


def b_2(state: State, config):
    print(f'Adding "B_2" to {state["aggregate"]}')
    return {"aggregate": ["B_2"]}


def d(state: State, config):
    print(f'Adding "D" to {state["aggregate"]}')
    return {"aggregate": ["D"]}


# 3. 通过状态创建图实例
graph = StateGraph(State)
graph.add_node("a", a)
graph.add_node("b", b)
graph.add_node("b_2", b_2)
graph.add_node("c", c)
graph.add_node("d", d)

# 4. 添加边
graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_edge("a", "c")
graph.add_edge("b", "b_2")
# 该方式执行 d节点会执行多次，因为有多个超级步
# graph.add_edge("b_2", "d")
# graph.add_edge("c", "d")
# 该方式d仅执行一次
graph.add_edge(["b_2", "c"], "d")

graph.add_edge("d", END)
# 指定会话
configuration = {'configurable': {'thread_id': '1'}}
# 指定持久化方式
saver = InMemorySaver()

# 5. 编译图
compiled_graph = graph.compile(checkpointer=saver)

# 6. 执行图，查看执行结果
output_state = compiled_graph.invoke({"aggregate": []}, config=configuration)
print('执行图后的状态为', output_state, end="\n\n")
compiled_graph.get_graph().print_ascii()

# 7. 查看当前图的节点
print('当前图的节点为', compiled_graph.nodes, end="\n\n")

# 8. 查看当前图的 channels
print('当前图的channels为', compiled_graph.channels, end="\n\n")

# 9. 查看 a 节点的 trigger（当前节点的订阅）和 writers
print('节点a的triggers为', compiled_graph.nodes['a'].triggers, end="\n\n")
print('节点a的writers为', compiled_graph.nodes['a'].writers, end="\n\n")

# 10. 查看 d 节点的 trigger 和 writers
print('当前图的节点d的triggers为', compiled_graph.nodes['d'].triggers, end="\n\n")
print('当前图的节点d的writers为', compiled_graph.nodes['d'].writers, end="\n\n")

# 查看历史所有状态
all_states = compiled_graph.get_state_history(config=configuration)
all_states_list = list(all_states)

print("历史所有状态如下：\n")
for state in all_states_list:
    print(state, end="\n" + "=" * 30 + "\n")

print("\n\n最近一次状态如下：\n")
# 3、获取最近一次状态
last_state = compiled_graph.get_state(config=configuration)
print(last_state)

"""

StateSnapshot(values=
    {'aggregate': ['A', 'B', 'C', 'B_2', 'D']}, 
    next=(), 
    config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f1b8f23-d7c3-662b-8004-35f2067ca709'}}, 
    metadata={'source': 'loop', 'step': 4, 'parents': {}}, created_at='2026-09-25T15:03:17.834295+00:00', 
    parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f1b8f23-d7c3-662a-8003-6e4db483d412'}}, 
    tasks=(), 
    interrupts=())

"""

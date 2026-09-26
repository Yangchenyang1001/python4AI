"""
  @Author:桌角是小黑
  @Time:2026/9/25
  @Desc:
"""

import operator
from typing import Annotated
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

# 5. 编译图
compiled_graph = graph.compile()

# 6. 执行图，查看执行结果
output_state = compiled_graph.invoke({"aggregate": []})
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

"""
    获取所有节点nodes
    {
        '__start__': <langgraph.pregel._read.PregelNode object at 0x000002AC00FA4490>,
         'a': <langgraph.pregel._read.PregelNode object at 0x000002AC00FA44C0>, 
         'b': <langgraph.pregel._read.PregelNode object at 0x000002AC00FA6740>, 
         'b_2': <langgraph.pregel._read.PregelNode object at 0x000002AC00FA4460>,
         'c': <langgraph.pregel._read.PregelNode object at 0x000002AC00FA4EB0>, 
         'd': <langgraph.pregel._read.PregelNode object at 0x000002AC00FA4DF0>
     }
    查看当前图的 channels   Topic主题模式 通过发布与订阅实现 消息通知
    {
        'aggregate': <langgraph.channels.binop.BinaryOperatorAggregate object at 0x000002AC00FA9580>, 
        '__start__': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x000002AC00FAA340>, 
        '__pregel_tasks': <langgraph.channels.topic.Topic object at 0x000002AC00FAA500>, 
        'branch:to:a': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x000002AC00FAA840>, 
        'branch:to:b': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x000002AC00FAAB80>, 
        'branch:to:b_2': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x000002AC00FAAE00>, 
        'branch:to:c': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x000002AC00FAB100>, 
        'branch:to:d': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x000002AC00FAB400>, 
        'join:b_2+c:d': <langgraph.channels.named_barrier_value.NamedBarrierValue object at 0x000002AC00FABFC0>
    }
    
    节点a的triggers为 ['branch:to:a']

    节点a的writers为 [
        ChannelWrite<...,...>(tags=None, recurse=True, explode_args=False, func_accepts={'config': ('N/A', <class 'inspect._empty'>)}, writes=(ChannelWriteTupleEntry(mapper=<function CompiledStateGraph.attach_node.<locals>._get_updates at 0x000002AC00FA3130>, value=<object object at 0x000002AC00764A30>, static=None), ChannelWriteTupleEntry(mapper=<function _control_branch at 0x000002AC00FA25F0>, value=<object object at 0x000002AC00764A30>, static=[]))), 
        ChannelWrite<branch:to:c>(tags=None, recurse=True, explode_args=False, func_accepts={'config': ('N/A', <class 'inspect._empty'>)}, writes=(ChannelWriteEntry(channel='branch:to:c', value=None, skip_none=False, mapper=None),)), ChannelWrite<branch:to:b>(tags=None, recurse=True, explode_args=False, func_accepts={'config': ('N/A', <class 'inspect._empty'>)}, writes=(ChannelWriteEntry(channel='branch:to:b', value=None, skip_none=False, mapper=None),))
    ]
    
    当前图的节点d的triggers为 ['branch:to:d', 'join:b_2+c:d']
    
    当前图的节点d的writers为 [
        ChannelWrite<...,...>(tags=None, recurse=True, explode_args=False, func_accepts={'config': ('N/A', <class 'inspect._empty'>)}, writes=(ChannelWriteTupleEntry(mapper=<function CompiledStateGraph.attach_node.<locals>._get_updates at 0x000002AC00FA3520>, value=<object object at 0x000002AC00764A30>, static=None), ChannelWriteTupleEntry(mapper=<function _control_branch at 0x000002AC00FA25F0>, value=<object object at 0x000002AC00764A30>, static=[])))
    ]

    
    
"""
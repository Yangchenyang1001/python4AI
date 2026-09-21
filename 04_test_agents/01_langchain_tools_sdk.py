"""
  @Author:桌角是小黑
  @Time:2026/9/20
  @Desc:
"""
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
import logging

logging.basicConfig(level=logging.DEBUG)

# 可选：通过BaseModel详细定义工具的参数
from pydantic import BaseModel, Field


class GetWeatherArgs(BaseModel):
    city: str = Field(description="城市名称")
    date: str = Field(description="日期，格式为 YYYY-MM-DD")

"""
@tool
将普通 Python 函数包装成 StructuredTool 对象 StructuredTool 继承自 LangChain 的 Runnable 基类 
因此具备了 .invoke()、.ainvoke() 等标准调用方法
"""
@tool
def get_weather(city, date) -> str:
    """获取指定城市在指定日期的天气"""
    return f"{city} 在 {date} 天气多云，有下雨的可能性"


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1")
llm = llm.bind_tools([get_weather])

message_list = [
    HumanMessage(content="北京 2024-12-25 的天气")
]
res = llm.invoke(message_list)
message_list.append(res)

# 当前结果当中包含调用工具的出参
print('LLM 的首次回复：', res)

# 解析调用工具的出参，手动调用工具
tool_call = res.tool_calls[0]
args = tool_call['args']

# pip install langchain-tavily
import os
call_tool_res = get_weather.invoke(args)
id = tool_call['id']

# 构造 Message 对象
tool_message = ToolMessage(tool_call_id=id, content=call_tool_res)
message_list.append(tool_message)
res = llm.invoke(message_list)
print('基于工具调用返回信息后，LLM 的回复：', res)
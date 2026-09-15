"""
  @Author:桌角是小黑
  @Time:2026/9/14
  @Desc:
"""
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
ollama_llm = ChatOllama(
	model="qwen2.5:7b", base_url="http://localhost:11434",
)
# 通过 Pydantic 定义 Json schema；
class Prime(BaseModel):
    prime: list[int] = Field(description="素数")
    count: list[int] = Field(description="小于该素数的素数个数")
# 使用构造好的 JSON Schem 构造 JsonOutputParser 实例 json_parser；
json_parser = JsonOutputParser(pydantic_object=Prime)
# 调用 json_parser 的 get_format_instructions()方法，将 json 结构输出约束，放到SystemMessage 当中
res = ollama_llm.invoke([("system", json_parser.get_format_instructions()),
                  ("user", "任意生成5个1000-100000之间素数，并标出小于该素数的素数个数")])
print(res.content)
#  结 果 使 用json_parser.parse()方法，解析成 PythonDict 对象。
parsed_res = json_parser.invoke(res)
print(type(parsed_res))
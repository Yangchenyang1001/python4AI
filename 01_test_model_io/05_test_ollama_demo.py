"""
  @Author:桌角是小黑
  @Time:2026/9/14
  @Desc:
"""
# pip install langchain-ollama
# 若 Ollama 不在本地默认端口运行，需指定 base_url，即：
from langchain_ollama import ChatOllama
ollama_llm = ChatOllama(
	model="deepseek-r1:14b", base_url="http://localhost:11434",
)
messages = [("user", "你好,请介绍下你自己")]
resp = ollama_llm.invoke(messages)
print(resp.content)
print(type(resp))
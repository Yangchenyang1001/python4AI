"""
  @Author:桌角是小黑
  @Time:2026/9/20
  @Desc:
"""
import os

from dotenv import load_dotenv
# --- Tavily 独立测试代码 ---
from langchain_tavily import TavilySearch

try:
    load_dotenv()
    print(os.getenv("TAVILY_API_KEY"))  # 打印出来看看是不是 None
    # 1. 初始化 Tavily 工具
    tavily_tool = TavilySearch(max_results=2)  # 限制结果数，防止请求超时

    # 2. 直接调用 Tavily 进行搜索测试
    print("正在测试 Tavily 搜索工具，请稍候...")
    result = tavily_tool.invoke("今天杭州天气如何？")

    # 3. 打印结果
    print("✅ Tavily 搜索成功！返回结果：")
    print(result)

except Exception as e:
    print("❌ Tavily 搜索失败，报错信息如下：")
    print(str(e))
# -------------------------
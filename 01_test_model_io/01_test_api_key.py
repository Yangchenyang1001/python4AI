"""
  @Author:桌角是小黑
  @Time:2026/9/14
  @Desc:
"""
# pip install python-dotenv
# 1、从 dotenv 导入 load_dotenv 方法
from dotenv import load_dotenv
# 2、调用 load_dotenv 方法加载.env 文件
load_dotenv()
# 3、通过os模块读取环境变量
import os
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)
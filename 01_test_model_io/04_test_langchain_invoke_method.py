"""
  @Author:桌角是小黑
  @Time:2026/9/14
  @Desc: 模型调用的各种方式 同步 异步 流式 批量
"""
import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

"""
普通同步调用
"""
def test_invoke_await():
    client = ChatOpenAI(
        model="gpt-4o-mini"
    )
    resp = client.invoke("你好")
    print(resp.content)

"""
异步调用
"""
async def test_invoke_sync():
    client = ChatOpenAI(
        model="gpt-4o-mini"
    )
    resp = await client.invoke("你好")
    print(resp.content)

"""
流式调用
"""
def test_stream_sync():
    client = ChatOpenAI(
        model="gpt-4o-mini"
    )
    resp = client.stream("你好")
    for chunk in resp:
        print(chunk.content)

from langchain_core.messages import HumanMessage
"""
批量调用
"""
def test_batch_await():
    client = ChatOpenAI(
        model="gpt-4o-mini"
    )
    response = client.batch(
        inputs=[
            [HumanMessage("什么是 Langchain")],
            [HumanMessage("Langchain 的核心模块")]
        ]
    )
    for question_chunk in response:
        print(question_chunk.content)


if __name__ == "__main__":
    # test_invoke_await()
    # asyncio.run(test_invoke_sync())
    # test_stream_sync()
    test_batch_await()

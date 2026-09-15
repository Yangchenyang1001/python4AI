"""
  @Author:桌角是小黑
  @Time:2026/9/14
  @Desc:
"""
from langchain_ollama import ChatOllama

def json_output_use_langchain():
    from pydantic import BaseModel, Field

    llm = ChatOllama(
        model="qwen2.5:7b", base_url="http://localhost:11434",
    )
    # 2、定义一个Pydantic模型，用于表示日历事件
    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    # 3、使用 with_structured_output，得到一个新的 llm，用于生成结构化输出
    new_llm = llm.with_structured_output(schema=CalendarEvent)
    # 4、调用新的llm，生成结构化输出
    res = new_llm.invoke("2026 年 5 月 13 日,张三和李四要参与一场会议，会议名称为'和平之旅' ")
    print(res)
    print(type(res))


if __name__ =="__main__":
    json_output_use_langchain();
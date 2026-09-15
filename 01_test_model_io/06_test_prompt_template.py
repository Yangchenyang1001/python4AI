"""
  @Author:桌角是小黑
  @Time:2026/9/14
  @Desc:
"""
from langchain_ollama import ChatOllama


def prompt_template_demo():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chat_models import init_chat_model
    # 使用构造方法实例化提示词模板
    chat_prompt_template = ChatPromptTemplate.from_messages(
        messages=[
            ("system", "你是一个专业的评论员"),
            ("human", "请评价{product}的优缺点，包括{aspect1}和{aspect2}。"),
        ],
    )
    chat_message_list = chat_prompt_template.invoke({"product": "iPhone15", "aspect1": "性能", "aspect2": "外观"})
    llm = ChatOllama(
        model="qwen2.5:7b", base_url="http://localhost:11434",
    )
    resp = llm.invoke(chat_message_list)
    print(resp.content)
if __name__ =="__main__":
    prompt_template_demo();
"""
  @Author:桌角是小黑
  @Time:2026/9/20
  @Desc:
"""
import json

from langchain_openai import OpenAI

client = OpenAI()
# 1. 通过JSON结构定义工具，包括工具名称，描述，参数等
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取地区特定日期的天气预报.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如 北京市",
                    },
                    "date": {
                        "type": "string",
                        "description": "想要查询的天气日期 YYYY-MM-dd,例如 2023-12-25"
                    }
                },
                "required": ["city", "date"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


def get_weather(city, date):
    return f"{city} 今天 {date}天气是: 气象局播报【晴天，温度是 25 度，风向是南风。】"


messages = [{"role": "user", "content": "北京 2025-12-25 日天气如何?"}]

# 2. Prompt the model with tools defined
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
)

for tool_call in response.choices[0].message.tool_calls or []:
    if tool_call.function.name == "get_weather":
        # 3. 执行工具函数的逻辑
        args = json.loads(tool_call.function.arguments)
        weather = get_weather(args["city"], args["date"])

        # 3.5 添加 assistant 的 tool_calls 消息（关键步骤）
        messages.append(response.choices[0].message)

        # 4. 将工具函数的执行结果添加到消息列表中
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({"weather": weather}),
            }
        )

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
)

# 5. 模型会根据工具函数的执行结果，生成最终的回复
print(response.choices[0].message.content)

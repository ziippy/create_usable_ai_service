from openai import OpenAI
from typing import Any, Callable, Dict, List, Tuple

client = OpenAI(api_key="...")
model = "gpt-4o-mini"

def respond(prompt: str, chat_history: List[str]) -> Tuple[str, List[str]]:
    messages = [{"role": "user", "content": prompt}]

    res = client.chat.completions.create(model=model, messages=messages)
    answer = res.choices[0].message.content

    chat_history.append((prompt, answer))

    return "", chat_history

prompt = "대한민국의 주요 뉴스"
response = []
_, response = respond(prompt, response)

print(response)
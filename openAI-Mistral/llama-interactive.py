# Test script only.
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:12434/engines/llama.cpp/v1/",  # llama.cpp server
    api_key="llama"
)

response = client.chat.completions.create(
    model="ai/mistral",  # model name from docker
    messages=[
        {"role": "user", "content": "how can I make you to execute on GPU inside docker"}
    ],
    temperature=0.7,
    max_tokens=256
)

print(response.choices[0].message.content)

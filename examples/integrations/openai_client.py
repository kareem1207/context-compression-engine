"""Point the official OpenAI client at cce-proxy; nothing else in your app changes.

    cce-proxy --upstream https://api.openai.com/v1        (or your local llama.cpp / Ollama / vLLM server)
    pip install openai
"""
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:9200/v1", api_key="YOUR_UPSTREAM_KEY")   # the key goes on to the upstream server
history = [{"role": "user", "content": "Hello!"}]
reply = client.chat.completions.create(
    model="gpt-4o-mini", messages=history,
    extra_headers={"X-CCE-Session": "my-chat"},        # optional; else the `user` field or a hash of the first message
)
print(reply.choices[0].message.content)

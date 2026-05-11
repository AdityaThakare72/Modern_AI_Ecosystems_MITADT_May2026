import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
client = InferenceClient(token=os.getenv("HF_TOKEN"))

messages = [
    {"role": "system", "content": "Be a concise AI expert."},
    {"role": "user", "content": "What is an LLM?"}
]

response = client.chat_completion(
    model="HuggingFaceH4/zephyr-7b-beta",
    messages=messages,
    max_tokens=50
)

print(response.choices[0].message.content)
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


model = ChatGroq(
    model="llama-3.1-8b-instant",
    max_tokens=50
)

# Direct string invocation
response = model.invoke("What is the main benefit of AI?")

print(response.content)
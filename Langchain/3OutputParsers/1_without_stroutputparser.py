from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not gemini_key:
    raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not set in .env")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=gemini_key,
    temperature=0.7,
    max_output_tokens=512,
)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"],
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text:\n{text}",
    input_variables=["text"],
)

# Generate detailed report
prompt1 = template1.invoke({"topic": "black hole"})

result = llm.invoke(prompt1)

# Generate summary based on detailed report
prompt2 = template2.invoke({"text": result.content})

result1 = llm.invoke(prompt2)

print(result1.content)

from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()


"""llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    max_output_tokens=50,
)"""

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    max_tokens=50
)

# Define templates
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text:\n{text}",
    input_variables=["text"],
)

parser = StrOutputParser()

# Compose the chain: template1 -> llm -> parse text output -> template2 -> llm -> parse
chain = template1 | llm | parser | template2 | llm | parser

result = chain.invoke({"topic": "black hole"})

print(result)

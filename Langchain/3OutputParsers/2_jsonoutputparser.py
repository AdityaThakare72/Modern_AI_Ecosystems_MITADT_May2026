from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

load_dotenv()

"""# Gemini model setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.7,
    max_output_tokens=50,
)"""
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    max_tokens=50
)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n{format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()},
)

chain = template | llm | parser

result = chain.invoke({'topic': 'black hole'})

print(result)

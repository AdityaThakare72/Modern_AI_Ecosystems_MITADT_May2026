from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()


# Initialize the Gemini language model with desired parameters
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",           
    temperature=0.7,           
    max_output_tokens=150,          
)

# Define the prompt template which takes a topic input and generates facts
prompt = ChatPromptTemplate(
    [
        {"role": "system", "content": "You are a helpful assistant that provides interesting facts."},
        {"role": "user", "content": "Give me 5 interesting facts about {topic}."}
    ]
)

# Initialize a string output parser to cleanly retrieve plain text results
parser = StrOutputParser()

# Create a chain that pipes the prompt output into model, then passes model output through the parser
chain = prompt | llm | parser

# Invoke the chain with a specific topic
result = chain.invoke({"topic": "Nikala tesla"})

# Print the final parsed result (string of 5 facts)
print(result)

# print the chain graph to visualize flow (helpful for debugging or docs)
print("\n--- Chain Graph ---")
chain.get_graph().print_ascii()

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
load_dotenv()

prompt1 = ChatPromptTemplate(
    [
        {"role": "system", "content": "You are a helpful assistant that provides interesting facts."},
        {"role": "user", "content": "Give me 5 interesting facts about {topic}."}
    ]
)

prompt2 = ChatPromptTemplate(
    [
        {"role": "system", "content": "You are a helpful assistant that summarizes text."},
        {"role": "user", "content": "Generate a 5 pointer summary from the following text \n {text}"}
    ]
)

model = ChatGroq(
    model="llama-3.1-8b-instant",
    max_tokens=50
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'Unemployment in India'})

print(result)

chain.get_graph().print_ascii()
# from langchain_core.chains import Chain
from langchain_community.chat_models import ChatOllama # Import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


load_dotenv()


# --- Ollama Model Setup ---
# No API token needed as Ollama runs locally.
# Ensure Ollama is running in your terminal: 'ollama serve'
# And that you have 'mistral' downloaded: 'ollama pull mistral' (if not already done)
model = ChatOllama(model="deepseek-r1:1.5b", temperature=0.7) # Specify the model name as 'mistral'


# --- Prompt, Model, Parser Chain ---
prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)


# StrOutputParser works universally
parser = StrOutputParser()


# Define the chain: Prompt -> Model -> Output Parser
chain = prompt | model | parser


# --- Invoke the chain ---
result = chain.invoke({'topic': 'cricket'})


print(result)


# You can still visualize the graph
print("\n--- Chain Graph ---")
chain.get_graph().print_ascii()
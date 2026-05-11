""" In the early days of LangChain, we primarily used LLMs. Now, we almost exclusively use ChatModels. Think of it like this: an LLM is a solitary scholar who finishes your sentences, while a ChatModel is a strategist you can actually converse with."""

"""Component	Input Type	    Output Type	        Analogy
LLMs =>	        Pure String	    Pure String	        A typewriter that predicts the next word.
ChatModels=>	List of Messages	Message Object	    A group chat where everyone has a specific role."""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# 1. Load the environment variables
load_dotenv()

# 2. Initialize the Gemini Model

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  # You can also use 'gemini-pro'
    temperature=0.7,
    max_output_tokens=50
)

# 3. Ask the question
question = "What is the future of AI? Answer in one sentence."

result = llm.invoke(question)
    
print(result.content) 
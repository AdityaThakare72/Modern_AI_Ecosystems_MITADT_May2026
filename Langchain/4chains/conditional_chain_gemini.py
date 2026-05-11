from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
import os

# Load .env file for environment variables like API keys
load_dotenv()

# Initialize Gemini model instead of OpenAI
model = ChatGroq(
    model="llama-3.1-8b-instant",
    max_tokens=50
)


# Simple string output parser to retrieve plain text output from the model
parser = StrOutputParser()

# Define a Pydantic data model to parse sentiment classification results from the model
class Feedback(BaseModel):
    # The sentiment can only be 'positive' or 'negative'
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

# Create a PydanticOutputParser based on the Feedback schema to automatically parse model output
parser2 = PydanticOutputParser(pydantic_object=Feedback)
# By using a PydanticOutputParser, you are forcing Gemini to return exactly {"sentiment": "positive"} or {"sentiment": "negative"}

# Prompt template for sentiment classification with formatting instructions injected via parser2
prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

# Create a classifier chain: runs prompt1 through the model and parses the output into Feedback object
classifier_chain = prompt1 | model | parser2

# Prompt template for positive feedback responses
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

# Prompt template for negative feedback responses
prompt3 = PromptTemplate(
    template='Write a sarcastic and rude response to this negative feedback because problem is not us its them \n {feedback}',
    input_variables=['feedback']
)

# RunnableBranch allows branching execution based on condition (sentiment value)
branch_chain = RunnableBranch(
    # If sentiment is positive, run prompt2 -> model -> plain text parser
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),

    # If sentiment is negative, run prompt3 -> model -> plain text parser
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),

    # Fallback case if sentiment is not recognized
    RunnableLambda(lambda x: "could not find sentiment")
)

# Compose full chain: first classify sentiment, then branch to response generation accordingly
chain = classifier_chain | branch_chain

# Example invocation with a positive feedback string
print(chain.invoke({'feedback': 'This is a really bad operating system. It crashes all the time and is very slow.'}))

# Print the chain graph as ASCII for visualization/debugging
chain.get_graph().print_ascii()

""" 
This code runs a model locally on your machine, but it uses the Hugging Face transformers library to do the heavy lifting
"""


from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)
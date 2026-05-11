import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables from the .env file
dotenv_path = '/home/aditya/GEN_AI_GOOGLE_NEW/.env'
load_dotenv(dotenv_path=dotenv_path)

"""# Get the Hugging Face API token from environment variables
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")""";



# Initialize the Hugging Face Inference Client
client = InferenceClient()



"""
Role       Purpose                          Example 

system      Sets the behavior and rules.    "You are a concise Python tutor."
user        The human's specific question.  "Explain lists."
assistant   The model's previous response.  "A list is an ordered collection..."
developer   A newer OpenAI role (replaces system)."Prioritize safety and brevity."

"""

# Define the messages in a chat format
messages = [
    {"role": "user", "content": "What is the future of AI?"}, {"role": "system", "content": "Answer in a sarcastic way."}
]

# Send the question to the model using chat_completion

response = client.chat_completion(
    model="HuggingFaceH4/zephyr-7b-beta",
    messages=messages,
    max_tokens=50,
    temperature=0.7
)

# Display the model's response
print("\n--- Model Response ---")
print(response)
print("----------------------")

print(response.choices[0].message.content)
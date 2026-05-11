import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# --- Env and API key ---
load_dotenv()


# --- Gemini model setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    max_output_tokens=512,
)

# --- Streamlit UI ---
st.set_page_config(page_title="Research Paper Explanation Tool", layout="wide")
st.title("Research Paper Explanation Tool")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
        "Cross Validation Holdout Method",
    ],
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"],
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"],
)

# --- ChatPromptTemplate ---
chat_template = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You are an expert Research Scientist and Teacher. Your goal is to explain complex AI papers "
        "clearly based on the user's requirements. If information is missing, say 'Insufficient information available'."
    ),
    (
        "human", 
        "Summarize the paper: {paper_name}\n"
        "Style: {style}\n"
        "Length: {length}\n\n"
        "Requirements:\n"
        "1. Include mathematical details/equations if relevant.\n"
        "2. Provide simple code snippets for math concepts where applicable.\n"
        "3. Use relatable analogies."
    )
])

# --- Chain and execution ---
if st.button("Summarize"):
    # The Chain: Prompt -> LLM
    chain = chat_template | llm
    
    with st.spinner("Analyzing the scrolls..."):
        result = chain.invoke(
            {
                
                "style": style_input,
                "length": length_input,
            }
        )
        st.markdown(result.content)
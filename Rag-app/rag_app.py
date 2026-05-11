"""
Minimal RAG App: Talk with Your Data (PDF only)
Upload PDF → Index with ChromaDB → Ask questions → Get answers from Gemini
"""

import streamlit as st
import os
from dotenv import load_dotenv
import PyPDF2
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ← FIXED IMPORT
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ============================================
# Config (Inline)
# ============================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PATH = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
K_RETRIEVAL = 5

# ============================================
# Streamlit Setup
# ============================================
st.set_page_config(page_title="Talk with Your Data", layout="wide")
st.title("Talk with Your Data (PDF)")

# Initialize session state
if "rag_pipeline" not in st.session_state:
    os.makedirs(CHROMA_PATH, exist_ok=True)
    st.session_state.rag_pipeline = chromadb.PersistentClient(path=CHROMA_PATH)
    st.session_state.collection = st.session_state.rag_pipeline.get_or_create_collection("docs")
    st.session_state.llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        api_key=GEMINI_API_KEY,
        temperature=0.7
    )
    st.session_state.chat_history = []

# ============================================
# Sidebar: Upload & Manage
# ============================================
with st.sidebar:
    st.header("Manage Documents")
    
    # Show stats
    count = st.session_state.collection.count()
    st.metric("Chunks Indexed", count)
    
    st.divider()
    
    # Upload PDF
    pdf_file = st.file_uploader("Upload PDF", type="pdf")
    
    if pdf_file and st.button("Process PDF"):
        # Extract text from PDF
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Chunk text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_text(text)
        
        # Add to ChromaDB
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{pdf_file.name}_chunk_{idx}"
            st.session_state.collection.add(
                ids=[chunk_id],
                documents=[chunk],
                metadatas=[{"source": pdf_file.name, "chunk_id": idx}]
            )
        
        st.success(f"{len(chunks)} chunks indexed from {pdf_file.name}")
    
    st.divider()
    
    if st.button("Clear All", use_container_width=True):
        st.session_state.rag_pipeline.delete_collection("docs")
        st.session_state.collection = st.session_state.rag_pipeline.get_or_create_collection("docs")
        st.session_state.chat_history = []
        st.success("Cleared!")

# ============================================
# Main: Chat Interface
# ============================================
if st.session_state.collection.count() == 0:
    st.warning("Upload a PDF first!")
else:
    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])
    
    st.divider()
    
    # Query input
    user_query = st.chat_input("Ask your question...")
    
    if user_query:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)
        
        # Retrieve relevant chunks
        results = st.session_state.collection.query(
            query_texts=[user_query],
            n_results=K_RETRIEVAL
        )
        
        # Format context
        context = "\n\n".join(results["documents"][0]) if results["documents"] else "No relevant info found"
        
        # Generate answer with Gemini
        prompt = PromptTemplate(
            template="Context: {context}\n\nQuestion: {query}\n\nAnswer using only the context:",
            input_variables=["context", "query"]
        )
        
        response = st.session_state.llm.invoke(
            prompt.format(context=context, query=user_query)
        )
        
        answer = response.content
        
        # Add assistant message
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

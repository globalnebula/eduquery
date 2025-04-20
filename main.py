import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import streamlit as st
import os
import tiktoken  # Add this to estimate token counts

# Initialize Sentence Transformer model and Groq client
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
client = Groq(api_key='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')

# Function to estimate tokens in a string (approximate)
def estimate_tokens(text):
    # Using cl100k_base encoder which is close to what many LLMs use
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# Function to extract text from PDFs (Knowledge Base)
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Load the knowledge base (pre-loaded PDFs from data folder)
def load_pdf_files_from_folder(folder_path):
    pdf_texts = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        pdf_texts.append(extract_text_from_pdf(pdf_path))
    return pdf_texts

# Create FAISS index from pre-loaded PDFs
def create_faiss_index(texts):
    global index, embeddings
    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

# Path to the data folder containing PDFs
data_folder = 'data'
pdf_texts = load_pdf_files_from_folder(data_folder)

# Create FAISS index
index = None
create_faiss_index(pdf_texts)

# Function to retrieve top-k results from FAISS
def retrieve_top_k(query, index, texts, k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [texts[i] for i in indices[0]]

# Function to truncate text to fit within token limits
def truncate_to_token_limit(text, max_tokens=4000):
    if estimate_tokens(text) <= max_tokens:
        return text
    
    # Simple truncation strategy - cut by characters and check token count
    tokens = tiktoken.get_encoding("cl100k_base").encode(text)
    return tiktoken.get_encoding("cl100k_base").decode(tokens[:max_tokens])

# Enhanced function with token limit handling
def enhance_with_llm(retrieved_texts, query):
    # Limit context to roughly 4000 tokens to leave room for other message parts
    combined_text = "\n\n---\n\n".join(retrieved_texts)
    limited_context = truncate_to_token_limit(combined_text, 4000)
    
    system_message = {
        "role": "system",
        "content": "You are an AI teacher. Answer based on the context provided. If the context doesn't contain relevant information, use your knowledge to provide a helpful response."
    }
    
    user_message = {
        "role": "user", 
        "content": f"Question: {query}\n\nContext:\n{limited_context}"
    }

    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[system_message, user_message],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""
        return response
    
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return "I encountered an error while processing your request. This might be due to the size of the documents or rate limits. Please try a more specific question or try again later."

# Function to get fusion answer (retrieval + LLM enhancement)
def get_fusion_answer(query, index, texts):
    retrieved_texts = retrieve_top_k(query, index, texts)
    enhanced_response = enhance_with_llm(retrieved_texts, query)
    return enhanced_response

# Streamlit Application with Enhanced UI
def main():
    st.markdown("""
        <style>
        .main {
            background-color: #f0f4f8;
            font-family: 'Roboto', sans-serif;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 0.5rem 1rem;
            border-radius: 10px;
            margin-top: 1rem;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stTextInput input {
            font-size: 16px;
            padding: 0.5rem;
            border-radius: 8px;
        }
        .stAlert {
            border-radius: 15px;
            background-color: #e0f7fa;
            color: #00796b;
            font-size: 1.25rem;
            padding: 1rem;
        }
        .stMarkdown p {
            font-size: 1.15rem;
            text-align: justify;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("✨ Interactive AI Teacher")
    st.subheader("Get AI-enhanced explanations from a knowledge base!")

    # User query input
    query = st.text_input("🔍 Enter your question here:")
    
    if st.button("Ask AI Teacher"):
        if query:
            with st.spinner("Fetching the AI response..."):
                answer = get_fusion_answer(query, index, pdf_texts)
            st.success("Here is the AI Teacher's response:")
            st.write(answer)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()

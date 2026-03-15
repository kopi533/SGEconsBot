import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA

# 1. Page Config
st.set_page_config(page_title="SG A-Level Econ Tutor", page_icon="🇸🇬")
st.title("🇸🇬 H2 Economics Assistant")
st.caption("Based on the SEAB 9570 Syllabus")

# 2. Setup (Hardcode your key or use a text input for safety)
# Check if we are in the cloud (Streamlit Cloud)
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    # This is for your local testing only
    api_key = "AIzaSyCaWopHUi2yTZd5gF5M9xIYJrqTwnfd4Ak"

# 3. Load the Knowledge (Connects to the database you already built)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", google_api_key=api_key)
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 4. Define the Tutor Brain
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=api_key, temperature=0.3)
tutor_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever())

# 5. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new input
if prompt := st.chat_input("Ask about Scarcity, Market Failure, or Macro..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # We add the syllabus context to the query
        enriched_prompt = f"Role: H2 Econ Tutor. Answer using syllabus: {prompt}"
        response = tutor_chain.run(enriched_prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

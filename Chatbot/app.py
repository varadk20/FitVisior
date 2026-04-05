import streamlit as st
import os
import shutil
from rag import RAGBot

st.set_page_config(page_title="Local RAG Chatbot", page_icon="🤖", layout="wide")

@st.cache_resource
def load_ragbot():
    # DON'T init vectorstore - just create RAGBot without loading
    bot = RAGBot()
    bot.vectorstore = None  # Prevent any DB loading
    return bot

def initialize_session():
    if "ragbot" not in st.session_state:
        st.session_state.ragbot = load_ragbot()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_ready" not in st.session_state:
        st.session_state.rag_ready = False
    if "db_exists" not in st.session_state:
        st.session_state.db_exists = os.path.exists("./chroma_db") and os.path.exists("./chroma_db/chroma.sqlite3")

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📄 Sources (File + Page)"):
                    for i, source in enumerate(message["sources"], 1):
                        st.success(f"{i}. {source}")

def add_message(role, content, sources=None):
    message = {"role": role, "content": content}
    if sources:
        message["sources"] = sources
    st.session_state.messages.append(message)

def main():
    initialize_session()
    
    # ✅ NO AUTO-INGEST - JUST CHECK EXISTING DB
    if not st.session_state.rag_ready:
        if st.session_state.db_exists:
            # Load existing DB (no folder creation)
            st.session_state.ragbot._init_vectorstore()
            if st.session_state.ragbot.vectorstore:
                st.session_state.rag_ready = True
                st.success("✅ Loaded existing vectorstore")
            else:
                st.error("❌ Corrupted vectorstore. Run ingest.py")
                st.stop()
        else:
            st.warning("⚠️ No vectorstore found")
            st.info("**Run first:** `python ingest.py`")
            st.stop()
    
    st.title("🤖 Local Fitness RAG Chatbot")
    st.markdown("**Powered by Ollama + ChromaDB**. Drop PDFs in `documents/` folder.")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("**Model**: mistral:7b-instruct")
        st.info("**Embedding**: nomic-embed-text:latest")
        st.info("**DB Path**: ./chroma_db")
        
        st.header("📊 Status")
        if st.session_state.rag_ready:
            try:
                count = st.session_state.ragbot.vectorstore._collection.count()
                st.metric("Vector Count", count)
            except:
                st.warning("Vectorstore not ready")
        
        if st.button("🗑️ Clear & Re-ingest", type="secondary"):
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db", ignore_errors=True)
            st.session_state.rag_ready = False
            st.session_state.messages = []
            st.session_state.db_exists = False
            st.rerun()
    
    display_chat_history()
    
    if st.session_state.rag_ready:
        if prompt := st.chat_input("Ask your fitness queries.."):
            # ✅ ADD USER MESSAGE FIRST
            with st.chat_message("user"):
                st.markdown(prompt)
            add_message("user", prompt)
            
            # ✅ SHOW ASSISTANT RESPONSE + SOURCES IMMEDIATELY
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.ragbot.query(prompt)
                    
                    # Answer first
                    st.markdown(result["answer"])
                    
                    # Sources immediately after answer
                    if result["sources"]:
                        with st.expander("📄 Sources (File + Page)", expanded=True):
                            for i, source in enumerate(result["sources"], 1):
                                st.success(f"{i}. {source}")
                    else:
                        st.info("❌ No sources found")
            
            # Add to history LAST
            add_message("assistant", result["answer"], result["sources"])

if __name__ == "__main__":
    main()

import streamlit as st
from rag_engine import SherBotRAG

st.set_page_config(page_title="SherBot", page_icon="🤖")

st.title("🤖 SherBot - Dataset Q&A Assistant")
st.caption("Answers strictly based on sher_khan_knowledge.jsonl")

# Initialize RAG Engine in Session State
if "rag_engine" not in st.session_state:
    try:
        with st.spinner("Loading dataset into vector store..."):
            st.session_state.rag_engine = SherBotRAG()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Query Input
if user_input := st.chat_input("Ask a question about the dataset..."):
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate bot response
    if "rag_engine" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Searching dataset..."):
                response = st.session_state.rag_engine.query(user_input)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

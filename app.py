import streamlit as st
import uuid
from tools import chat
from memory import load_history, save_history

st.title("UMN CS Graduate Advisor")
st.caption("Ask me anything about the CS graduate program at University of Minnesota.")

# Persist session ID across refreshes using query params
params = st.query_params
if "session_id" not in params:
    new_id = str(uuid.uuid4())
    st.query_params["session_id"] = new_id

session_id = st.query_params["session_id"]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = load_history(session_id)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, st.session_state.conversation_history = chat(
                prompt,
                st.session_state.conversation_history
            )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    save_history(session_id, st.session_state.conversation_history)
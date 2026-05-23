import streamlit as st
from tools import chat

st.title("UMN CS Graduate Advisor")
st.caption("Ask me anything about the CS graduate program at University of Minnesota.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
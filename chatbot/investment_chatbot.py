# ============================================================
# FILE: chatbot/web_chatbot.py
# ============================================================

import streamlit as st
import ollama


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Investment Chatbot",
    page_icon="📈",
    layout="centered"
)

st.title("📈 AI Investment Chatbot")


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# USER INPUT
# ============================================================

prompt = st.chat_input("Ask about stocks or investments...")


if prompt:

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an AI Investment Assistant. "
                            "Give beginner-friendly financial explanations."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            reply = response["message"]["content"]

            st.markdown(reply)

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply
        }
    )
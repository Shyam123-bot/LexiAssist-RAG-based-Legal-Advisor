import time
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.generation import generate_response

# Initialize UI
st.set_page_config(page_title="RAG-Based Legal Assistant")
col1, col2, col3 = st.columns([1, 25, 1])
with col2:
    st.title("RAG-Based Legal Assistant")

# Setup session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Reset conversation
def reset_conversation():
    st.session_state["messages"] = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.write(message.content)

# User input
user_query = st.chat_input("Ask me anything ..")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.status("Generating 💡...", expanded=True):
            # Invoke generation function
            result = generate_response(user_query, st.session_state["messages"])

            message_placeholder = st.empty()
            full_response = (
                "⚠️ **_This information is not intended as a substitute for legal advice. "
                "We recommend consulting with an attorney for a more comprehensive and"
                " tailored response._** \n\n\n"
            )

            # Simulate streaming effect
            for chunk in result:
                full_response += chunk
                time.sleep(0.02)
                message_placeholder.markdown(full_response + " ▌")

        st.button('Reset Conversation 🗑️', on_click=reset_conversation)

    # Append messages to session state
    st.session_state.messages.extend(
        [
            HumanMessage(content=user_query),
            AIMessage(content=result)
        ]
    )

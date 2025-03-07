import streamlit as st
from ctransformers import AutoModelForCausalLM

# Load the local Mistral model
model_path = "./models/Mistral-7B-Instruct-v0.1.Q2_K.gguf"
llm = AutoModelForCausalLM.from_pretrained(model_path)

# Streamlit UI Settings
st.set_page_config(page_title="Chatbot", page_icon="💬", layout="wide")

st.markdown(
    """
    <style>
        body {
            background-color: #f0f2f5;
        }
        .chat-container {
            max-width: 450px;
            margin: auto;
            padding: 15px;
        }
        .chat-bubble {
            padding: 10px;
            border-radius: 15px;
            margin: 5px;
            max-width: 75%;
            display: inline-block;
            font-size: 16px;
            word-wrap: break-word;
        }
        .user-bubble {
            background-color: #25D366;
            color: white;
            align-self: flex-end;
            text-align: right;
        }
        .bot-bubble {
            background-color: #E4E6EB;
            color: black;
            align-self: flex-start;
        }
        .message-container {
            display: flex;
            flex-direction: column;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💬🤖 Sereni - Your Personal Mental Health Chatbot")
st.markdown("You can share anything with me. This is a safe space ❤️")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Function to generate chatbot responses
def generate_response(user_input, chat_history):
    # Format chat history properly
    conversation = "\n".join([f"{role}: {text}" for role, text in chat_history if role in ["User", "AI"]])

    # Create a structured prompt
    prompt = f"{conversation}\nUser: {user_input}\nAI:"

    output = llm(
        prompt, 
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    ).strip()

    # Ensure bot doesn't hallucinate previous messages
    if "User:" in output:  
        output = output.split("User:")[0].strip()  # Cut off unintended history

    return output

# Display chat history in WhatsApp-style UI
st.write('<div class="chat-container">', unsafe_allow_html=True)
for role, text in st.session_state["chat_history"]:
    bubble_class = "user-bubble" if role == "User" else "bot-bubble"
    st.markdown(
        f'<div class="chat-bubble {bubble_class}">{text}</div>',
        unsafe_allow_html=True
    )
st.write("</div>", unsafe_allow_html=True)

# User input field
user_input = st.text_input("Type your message...", "", key="user_input")

if st.button("Send"):
    if user_input:
        # Store user input
        st.session_state["chat_history"].append(("User", user_input))

        # Generate response
        response = generate_response(user_input, st.session_state["chat_history"])
        st.session_state["chat_history"].append(("AI", response))

        # Refresh UI
        st.rerun()

# Clear chat button
if st.button("Clear Chat"):
    st.session_state["chat_history"] = []
    st.rerun()

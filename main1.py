from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ctransformers import AutoModelForCausalLM
import datetime

app = Flask(__name__)
CORS(app)

# Load the Mistral model locally
model_path = "./models/Mistral-7B-Instruct-v0.1.Q2_K.gguf"
llm = AutoModelForCausalLM.from_pretrained(model_path)

# Store chat history and appointments
chat_history = []
appointments = []
user_booking_data = {}

def generate_response(user_input):
    """Generates a response from the chatbot using stored history."""
    global chat_history, user_booking_data
    user_input = user_input.lower()

    # Appointment Booking Flow
    if "book an appointment" in user_input:
        user_booking_data.clear()
        user_booking_data["stage"] = "asking_details"
        return "Sure! What is your name and preferred date & time for the appointment?"
    
    if user_booking_data.get("stage") == "asking_details":
        try:
            parts = user_input.split()
            name = parts[0]
            date = parts[1]
            time = " ".join(parts[2:])
            datetime.datetime.strptime(date, "%Y-%m-%d")  # Validate date
            datetime.datetime.strptime(time, "%I:%M %p")  # Validate time
            user_booking_data["name"] = name
            user_booking_data["date"] = date
            user_booking_data["time"] = time
            user_booking_data["stage"] = "completed"
            calendly_link = "https://calendly.com/example-therapist/appointment"
            return f"Thank you, {name}. Your appointment is set for {date} at {time}. You can confirm it here: {calendly_link}"
        except:
            return "Please provide your name, date (YYYY-MM-DD), and time (HH:MM AM/PM)."

    conversation = "\n".join([f"{role}: {text}" for role, text in chat_history])
    prompt = f"{conversation}\nUser: {user_input}\nAI:"

    output = llm(prompt, max_new_tokens=150, temperature=0.7, top_p=0.9, repetition_penalty=1.2).strip()

    chat_history.append(("User", user_input))
    chat_history.append(("AI", output))
    return output

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    return jsonify({"response": generate_response(data.get("message", ""))})

@app.route("/clear", methods=["POST"])
def clear_chat():
    chat_history.clear()
    user_booking_data.clear()
    return jsonify({"message": "Chat history cleared"})

if __name__ == "__main__":
    app.run(debug=True)

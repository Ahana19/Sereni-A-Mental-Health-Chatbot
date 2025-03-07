import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from torch.utils.data import Dataset

# Step 1: Load the model and tokenizer
def load_chatbot_model():
    model_name = "microsoft/GODEL-v1_1-base-seq2seq"  # Model optimized for empathetic responses
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

# Step 2: Fine-tune the model
def fine_tune_model(model, tokenizer):
    dataset = load_dataset("bdotloh/empathetic-dialogues-contexts",trust_remote_code=True)
    
    # Preprocessing the data
    def preprocess_data(example):
        user_input = example.get("context", "")
        bot_response = example.get("utterance", "")
        input_text = f"User: {user_input} Bot:"
        target_text = bot_response
        return {"input_ids": tokenizer.encode(input_text, truncation=True, max_length=512), 
                "labels": tokenizer.encode(target_text, truncation=True, max_length=512)}
    
    # Apply preprocessing
    train_dataset = dataset["train"].map(preprocess_data, remove_columns=dataset["train"].column_names)
    val_dataset = dataset["validation"].map(preprocess_data, remove_columns=dataset["validation"].column_names)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs"
    )

    # Define Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    
    # Train the model
    trainer.train()
    return model

# Step 3: Generate a response
def generate_response(user_input, model, tokenizer, chat_history):
    empathetic_prompt = f"You are an empathetic mental health assistant. Be kind and supportive. User says: '{user_input}'"
    context = f"{chat_history} User: {user_input}"
    inputs = tokenizer.encode(empathetic_prompt + " " + context, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs, max_length=1000, top_k=50, top_p=0.95, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Step 4: Tailored responses (optional)
def tailored_responses(user_input, model_response):
    if "sad" in user_input.lower():
        return "I'm sorry you're feeling this way. It's okay to feel sad sometimes. Do you want to talk more about what's troubling you?"
    elif "anxious" in user_input.lower():
        return "Anxiety can be overwhelming. Try taking a few deep breaths with me. I'm here to help."
    elif "happy" in user_input.lower():
        return "That's wonderful to hear! What made you happy today?"
    else:
        return model_response

# Step 5: Chatbot loop
def start_chatbot():
    print("Mental Health Chatbot: Hi! I'm here to listen and help. How are you feeling today? (Type 'exit' to quit.)")
    model, tokenizer = load_chatbot_model()
    
    # Fine-tune the model
    print("Fine-tuning the model for better empathetic responses...")
    model = fine_tune_model(model, tokenizer)

    # Start the conversation loop
    chat_history = ""
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Mental Health Chatbot: Take care! Remember, you're not alone. Goodbye!")
            break
        model_response = generate_response(user_input, model, tokenizer, chat_history)
        final_response = tailored_responses(user_input, model_response)
        chat_history += f"User: {user_input}\nBot: {final_response}\n"
        print(f"Mental Health Chatbot: {final_response}")

if __name__ == "__main__":
    start_chatbot()

import threading
import time
from llama_cpp import Llama

print("Starting model load...", flush=True)
model_path = "Model/gemma-3-12b-it-Q8_0.gguf"
llm = None

def load_model():
    global llm
    llm = Llama(model_path=model_path)

t = threading.Thread(target=load_model)
t.start()

while t.is_alive():
    print("Model is loading...", end="\r")
    time.sleep(1)

print("\nGemma 12B Q8 with long-term memory is ready! Type 'exit' or 'quit' to stop.")

chat_history = []
MAX_HISTORY = 6
SUMMARY_THRESHOLD = 12
conversation_summary = ""

def summarize_history_async(messages, existing_summary="", callback=None):
    def worker():
        nonlocal existing_summary
        if existing_summary:
            prompt = f"Update this conversation summary based on new messages.\nCurrent summary: {existing_summary}\n\nNew messages:\n"
        else:
            prompt = "Summarize the following conversation briefly:\n\n"
        prompt += "\n".join(messages) + "\n\nUpdated summary:"
        response = llm(prompt=prompt, max_tokens=150)
        updated_summary = response["choices"][0]["text"].strip()
        if callback:
            callback(updated_summary)
    threading.Thread(target=worker, daemon=True).start()

def update_summary(new_summary):
    global conversation_summary
    conversation_summary = new_summary

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    chat_history.append(f"You: {user_input}")

    if len(chat_history) > SUMMARY_THRESHOLD:
        old_messages = chat_history[:-MAX_HISTORY]
        summarize_history_async(old_messages, conversation_summary, callback=update_summary)
        chat_history = [f"[Summary of earlier conversation: {conversation_summary}]"] + chat_history[-MAX_HISTORY:]

    prompt = "\n".join(chat_history) + "\nGemma:"
    response = llm(
        prompt=prompt,
        max_tokens=200,
        stop=["You:", "Gemma:"]
    )

    generated_text = response["choices"][0]["text"].strip()
    print(f"Gemma: {generated_text}")
    chat_history.append(f"Gemma: {generated_text}")

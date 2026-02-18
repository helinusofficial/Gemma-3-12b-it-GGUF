# chat_smart_memory.py
from llama_cpp import Llama

# Load the model
model_path = "Model/gemma-3-12b-it-Q8_0.gguf"
llm = Llama(model_path=model_path)

print("Gemma 12B Q8 is ready for chat! Type 'exit' or 'quit' to stop.")

chat_history = []          # full conversation history
MAX_HISTORY = 6            # number of recent messages to include directly
SUMMARY_THRESHOLD = 12     # if more messages than this, summarize older ones
conversation_summary = ""  # summarized version of older messages

def summarize_history(messages):
    """
    Summarize a list of messages using the model itself.
    """
    prompt = "Summarize the following conversation briefly:\n\n"
    prompt += "\n".join(messages) + "\n\nSummary:"
    response = llm(prompt=prompt, max_tokens=150)
    return response["choices"][0]["text"].strip()

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    chat_history.append(f"You: {user_input}")

    # If history is too long, summarize older messages
    if len(chat_history) > SUMMARY_THRESHOLD:
        old_messages = chat_history[:-MAX_HISTORY]
        conversation_summary = summarize_history(old_messages)
        chat_history = [f"[Summary of earlier conversation: {conversation_summary}]"] + chat_history[-MAX_HISTORY:]

    # Build prompt for the model
    prompt = "\n".join(chat_history) + "\nGemma:"

    response = llm(
        prompt=prompt,
        max_tokens=200,
        stop=["You:", "Gemma:"]
    )

    generated_text = response["choices"][0]["text"].strip()
    print(f"Gemma: {generated_text}")

    chat_history.append(f"Gemma: {generated_text}")
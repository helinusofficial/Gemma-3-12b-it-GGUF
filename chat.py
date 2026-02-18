# chat_optimized.py
from llama_cpp import Llama

# Load the model
model_path = "Model/gemma-3-12b-it-Q8_0.gguf"
print("Loading model...")
llm = Llama(model_path=model_path)
print("Gemma 12B Q8 is ready for chat! Type 'exit' or 'quit' to stop.")

# Use a list to store messages
chat_history = []

# Maximum number of recent messages to keep in prompt
MAX_HISTORY = 3

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Add user input to history
    chat_history.append(f"You: {user_input}")

    # Build the prompt using the last MAX_HISTORY messages
    recent_history = chat_history[-MAX_HISTORY:]
    prompt = "\n".join(recent_history) + "\nGemma:"

    response = llm(
        prompt=prompt,
        max_tokens=200,
        stop=["You:", "Gemma:"]
    )

    generated_text = response["choices"][0]["text"].strip()
    print(f"Gemma: {generated_text}")

    # Add model response to history
    chat_history.append(f"Gemma: {generated_text}")
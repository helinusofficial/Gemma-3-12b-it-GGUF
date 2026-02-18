import threading
import time
from llama_cpp import Llama

print("Starting model load...", flush=True)
model_path = "Model/gemma-3-12b-it-Q8_0.gguf"
llm = None

def load_model():
    global llm
    custom_template = "{{ bos_token }}\n{%- for message in messages -%}\n{{ message['content'] | trim }}\n{%- endfor -%}"
    llm = Llama(model_path=model_path, chat_template=custom_template)


t = threading.Thread(target=load_model)
t.start()

while t.is_alive():
    print("Model is loading...", end="\r")
    time.sleep(1)

print("\nGemma 12B Q8 Ready!\n")

# =============================
# SYSTEM PROMPTS
# =============================

PROMPTS = {
    "1": "You are a helpful and friendly AI assistant.",
    "2": "You are a concise technical expert. Give structured and precise answers.",
    "3": "You are a senior Python developer. Provide clean production-ready code.",
    "4": "You are an IT and DevOps expert. Give practical commands and solutions.",
    "5": "You are a creative storyteller.",
    "6": "You are strict, analytical and direct.",
    "7": "You are a caring and affectionate virtual girlfriend. Be warm, supportive and emotionally intelligent.",
    "8": "You are a caring and supportive virtual boyfriend.",
    "9": "You are a loving and emotionally mature virtual wife.",
    "10": "You are a calm, responsible and protective virtual husband."
}

print("Select personality:\n")
for k in PROMPTS:
    print(f"{k}) {PROMPTS[k][:60]}...")

print("11) Custom")

choice = input("\nEnter number: ").strip()

if choice == "11":
    system_prompt = input("\nEnter your custom system prompt:\n> ").strip()
else:
    system_prompt = PROMPTS.get(choice, PROMPTS["1"])

base_personality = system_prompt

lock_choice = input("\nLock personality? (y/n, default y): ").strip().lower()
PERSONALITY_LOCKED = True if lock_choice in ["", "y", "yes"] else False

# =============================
# Temperature & Top_p
# =============================
try:
    temperature = float(input("\nTemperature (how creative the AI is, 0.1-1.2, default 0.7): ") or 0.7)
except:
    temperature = 0.7

try:
    top_p = float(input("Top_p (how varied the AI word choices are, 0.1-1.0, default 0.9): ") or 0.9)
except:
    top_p = 0.9

print("\nChat started. Type 'exit' to quit.\n")

# =============================
# Memory
# =============================

chat_history = []
important_memory = []
MAX_HISTORY = 8

def build_prompt(user_input):
    prompt = f"{system_prompt}\n"

    if important_memory:
        prompt += "[Important user facts:]\n"
        for mem in important_memory[-10:]:
            prompt += f"- {mem}\n"

    for msg in chat_history[-MAX_HISTORY:]:
        role, text = msg.split(": ", 1)
        if role == "You":
            prompt += f"User: {text}\n"
        else:
            prompt += f"Assistant: {text}\n"

    prompt += f"User: {user_input}\nAssistant:"
    return prompt

def extract_important_info(text):
    keywords = ["my name is", "i am", "i work", "my project", "remember that"]
    for k in keywords:
        if k in text.lower():
            important_memory.append(text)

# =============================
# Context Auto Mode
# =============================

def auto_adjust_prompt(user_input):
    global system_prompt

    if PERSONALITY_LOCKED:
        return

    lower = user_input.lower()

    if any(word in lower for word in ["code", "python", "function", "class"]):
        system_prompt = base_personality + " When giving code, be precise and structured."
    elif "story" in lower or "write a story" in lower:
        system_prompt = base_personality + " Be expressive and descriptive."
    elif any(word in lower for word in ["server", "linux", "docker", "nginx"]):
        system_prompt = base_personality + " Provide practical technical steps and commands."
    else:
        system_prompt = base_personality

# =============================
# Chat Loop with Streaming
# =============================

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    extract_important_info(user_input)
    auto_adjust_prompt(user_input)

    prompt = build_prompt(user_input)

    print("Gemma: ", end="", flush=True)

    response = llm(
        prompt=prompt,
        max_tokens=300,
        temperature=temperature,
        top_p=top_p,
        stop=["User:"],
        stream=True
    )

    full_response = ""

    for chunk in response:
        token = chunk["choices"][0]["text"]
        print(token, end="", flush=True)
        full_response += token

    print()

    chat_history.append(f"You: {user_input}")
    chat_history.append(f"Gemma: {full_response.strip()}")

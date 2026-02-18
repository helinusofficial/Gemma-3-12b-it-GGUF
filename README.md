# Gemma 3 12B It – Chat Model

## Overview

**Model Name:** Gemma 3 12B It (Quantized Q8_0)  
**Architecture:** gemma3  
**Base Model:** Gemma 3 12B Pt (by Google)  
**Quantization:** Q8_0 (8-bit integer quantization for reduced memory usage)  
**File Format:** GGUF V3  
**Model Size:** ~11.64 GiB (quantized)  
**Context Length:** 131072 tokens  
**Embedding Size:** 3840  
**Feedforward Hidden Size:** 15360  
**Number of Layers:** 48  
**Attention Heads:** 16 (8 key/value heads)  
**Sliding Window Attention:** 1024 tokens  
**ROPE Scaling:** linear, factor 8.0  
**Tokenizer Type:** llama-compatible  
**License:** gemma  

Gemma 3 12B It is a large-scale generative language model fine-tuned for instruction-following tasks and interactive chat. It is based on the pre-trained Gemma 3 12B Pt model by Google, with additional instruction tuning for Italian (hence “It” in the name). The model is designed to support text-based conversations and instruction-following tasks, providing responsive, context-aware dialogue.

The Q8_0 quantization significantly reduces memory requirements while maintaining efficient inference on CPU-only machines. This makes it accessible for desktops and laptops without GPUs.



## Key Features

- **Instruction Following:** Optimized for responding to user instructions accurately.  
- **Long-Term Context Support:** Can handle conversations up to 131072 tokens.  
- **Streaming Output:** Supports real-time token streaming for interactive chat.  
- **Customizable System Prompts:** Define personalities or modes for the assistant.  
- **Persistent Memory:** Keep track of important conversation details.  
- **CPU-Efficient:** Runs efficiently on modern CPUs without requiring GPU.  



## Usage in Python

You can load and use the model with [`llama_cpp`](https://github.com/abetlen/llama-cpp-python):

python
from llama_cpp import Llama

model_path = "Model/gemma-3-12b-it-Q8_0.gguf"

# Load the model
llm = Llama(
    model_path=model_path,
    n_threads=8,   # Number of CPU threads for inference
    n_batch=16     # Token batch size
)

# Example usage
prompt = "You are a helpful assistant. Explain what AI is in simple terms."
response = llm(prompt=prompt, max_tokens=150)
print(response["choices"][0]["text"])


## Python Scripts

### `chat.py`

A **simple, minimal chat interface** for interacting with the model.

- Intended for short conversations.  
- Streams tokens in real time.  
- Does not include persistent memory or conversation summarization.  
- Ideal for testing the model quickly or running a lightweight chat session.  

### `chat_long_memory.py`

A **structured chat interface** designed for longer, context-rich conversations.

- Maintains **persistent memory** to store important user information.  
- Supports **conversation summarization** to manage long histories.  
- Allows **custom system prompts and personalities**.  
- Can dynamically adjust context or auto-mode depending on user input.  
- Best for virtual assistants, storytelling, or multi-turn instruction-following chats.  

Both scripts use the same Gemma 12B Q8 model but provide different levels of complexity depending on the use case.

---

## Download

You can download the Gemma 3 12B It (Q8_0 GGUF) model from Hugging Face:

[Gemma 3 12B It Q8_0 GGUF](https://huggingface.co/second-state/gemma-3-12b-it-GGUF/blob/main/gemma-3-12b-it-Q8_0.gguf)

---

## Notes

- The model may include automatic English translations in some versions; you can remove them programmatically if desired.  
- Quantized Q8_0 models reduce RAM usage but have slightly lower numerical precision compared to FP16.  
- Optimal CPU usage can be achieved by adjusting `n_threads` and `n_batch` parameters.  
- Supports persistent memory and context-aware auto mode for interactive chat.  
- Ideal for chatbots, virtual assistants, interactive storytelling, and technical Q&A applications.

---

## License

The model is released under the **Gemma license**. Please review the terms before using in commercial or public applications.

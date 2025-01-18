import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set random seed for reproducibility
torch.manual_seed(0)

# Define the Hugging Face model repository
model_name = "microsoft/Phi-3.5-mini-instruct"

# Check for GPU availability
is_gpu_available = torch.cuda.is_available()

print("\033[96m")  # Cyan text
print("GPU detected. Using GPU-optimized model." if is_gpu_available else "No GPU detected. Using CPU-optimized model.")
print("\033[0m")  # Reset text color

# Load model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically handles device placement
    torch_dtype="auto",  # Ensures proper data type for GPU/CPU
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the pipeline for inference
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)  # `device` is unnecessary here as `device_map="auto"` is used

# System prompt and conversation history
system_prompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information than requested by the users."
conversation_history = [{"role": "system", "content": system_prompt}]

print("\033[96mAsk your question. Type 'CLH' to clear history and start a new session. Press Enter to exit.\033[0m")

# Chat loop
while True:
    print("\n\033[94mQ: \033[0m", end="")  # Blue text for user input
    user_input = input()
    if not user_input.strip():
        break

    # Clear history if the user types 'CLH'
    if user_input.upper() == "CLH":
        conversation_history = [{"role": "system", "content": system_prompt}]
        print("\033[93mHistory cleared. Starting a new session.\033[0m")  # Yellow text
        continue

    # Append user input to the conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Generate response
    print("\033[92mPhi3.5: \033[0m", end="")  # Green text for model output
    try:
        generation_args = {
            "max_new_tokens": 256,
            "return_full_text": False,
            "temperature": 0.7,
            "do_sample": True,
        }
        output = pipe(conversation_history, **generation_args)
        assistant_response = output[0]["generated_text"]
        print(assistant_response)

        # Append the assistant's response to the conversation history
        conversation_history.append({"role": "assistant", "content": assistant_response})
    except Exception as e:
        print(f"\033[91mFailed to generate response: {e}\033[0m")  # Red text

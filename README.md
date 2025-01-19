# **SLM Demo: Conversational AI with Phi-3.5-mini-instruct**

This project demonstrates the use of **Small Language Models (SLMs)**, specifically **Phi-3.5-mini-instruct**, to create a lightweight conversational AI system. This Python-based demo is part of Alon Fliess's **SLM Lecture**, showcasing how to run and interact with a small, efficient model locally without requiring cloud infrastructure.

---

## **Features**

- Utilizes Hugging Face's **Phi-3.5-mini-instruct** for conversational AI tasks.
- **Supports GPU Acceleration**: Automatically detects GPU availability and optimizes model execution.
- **Session History Management**: Maintains a conversation history for context-aware responses.
- **Interactive Chat Loop**: Provides a user-friendly console interface to interact with the AI.
- **History Reset**: Allows resetting the conversation history with the "CLH" command.
- **Customizable Behavior**: Uses a system prompt to define the assistant's personality and behavior.

---

## **How It Works**

### **Setup Phase**
1. **Model and Tokenizer Loading**:
   - The script loads the **Phi-3.5-mini-instruct** model and tokenizer from Hugging Face's `transformers` library.
   - The `device_map="auto"` option ensures automatic placement of the model on GPU or CPU, depending on availability.

2. **Pipeline Initialization**:
   - A Hugging Face **pipeline** is created for text generation, leveraging the loaded model and tokenizer.

3. **System Prompt**:
   - A system-level prompt defines the assistant's behavior, ensuring consistent and helpful responses.

---

### **Main Chat Loop**
1. **User Interaction**:
   - The user inputs questions in the console, and the assistant generates responses based on the conversation history.
   - The command `CLH` clears the history and starts a new session.

2. **Context-Aware Responses**:
   - The assistant maintains a conversation history, allowing it to provide contextually relevant answers.

3. **Response Generation**:
   - The model generates responses using the Hugging Face pipeline with specified arguments:
     - `max_new_tokens`: Limits the length of the generated response.
     - `temperature`: Controls response randomness for creative or deterministic outputs.
     - `do_sample`: Enables sampling for non-deterministic response generation.

---

## **Example Interaction**

### **Console Input**:
```plaintext
Q: What is the capital of France?
Phi3.5: The capital of France is Paris.
```

### **Clearing History**:
```plaintext
Q: CLH
History cleared. Starting a new session.
```

### **Multi-turn Conversation**:
```plaintext
Q: Tell me about bananas.
Phi3.5: Bananas are a rich source of potassium and make for a healthy snack.
Q: What can I do with bananas?
Phi3.5: You can eat bananas raw, blend them into smoothies, or use them in baking recipes.
```

---

## **Prerequisites**

- **Python**: Ensure Python 3.8 or higher is installed.
- **Hugging Face Transformers**: Install the required library for loading and interacting with the model.
- **GPU (Optional)**: The script utilizes GPU acceleration if available.

---

## **Setup Instructions**

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python src/main.py
   ```

---

## **Code Highlights**

### **Device Detection**
The script checks for GPU availability and adapts the model loading accordingly:
```python
is_gpu_available = torch.cuda.is_available()
device = "cuda" if is_gpu_available else "cpu"
```

### **Conversation Management**
Maintains session context using a `conversation_history` list:
```python
conversation_history = [{"role": "system", "content": system_prompt}]
```

### **Model Pipeline**
Uses Hugging Face's pipeline API for text generation:
```python
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
```

### **History Reset**
Clears conversation history with the `CLH` command:
```python
if user_input.upper() == "CLH":
    conversation_history = [{"role": "system", "content": system_prompt}]
```

---

## **Customizing the Assistant**

To change the assistant's behavior, modify the `system_prompt`:
```python
system_prompt = "You are an AI assistant that provides concise, fact-based answers."
```

---

## **Use Cases**

- Demonstrating the capabilities of Small Language Models in real-world scenarios.
- Showcasing lightweight, efficient AI solutions for devices with limited resources.
- Educating developers on integrating Hugging Face models into Python applications.

---

This project is designed to illustrate the power and efficiency of **Small Language Models** and inspire developers to explore their potential in diverse applications. Feel free to modify and extend the script to suit your needs!

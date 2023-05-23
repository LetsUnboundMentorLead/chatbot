import tkinter as tk
from tkinter import scrolledtext
from transformers import TFAutoModelForCausalLM, AutoTokenizer

# Load the pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
model = TFAutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to handle user input and generate chatbot response
def send_message():
    user_input = user_entry.get()

    # Tokenize user input
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="tf")

    # Generate model response
    response = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the response
    chatbot_response = tokenizer.decode(response[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Display chatbot response in the conversation
    conversation_text.configure(state='normal')
    conversation_text.tag_config('user', foreground='blue')
    conversation_text.tag_config('chatbot', foreground='green')
    conversation_text.insert(tk.END, "You: ", 'user')
    conversation_text.insert(tk.END, f"{user_input}\n")
    conversation_text.insert(tk.END, "ChatBot: ", 'chatbot')
    conversation_text.insert(tk.END, f"{chatbot_response}\n")
    conversation_text.configure(state='disabled')

    # Clear user input entry
    user_entry.delete(0, tk.END)

# Create main window
window = tk.Tk()
window.title("ChatBot")
window.geometry("400x500")

# Set background color of the window
window.configure(background='lightblue')

# Create conversation display area
conversation_text = scrolledtext.ScrolledText(window, state='disabled', wrap='word', width=40, height=20)
conversation_text.pack(pady=10)

# Create user input entry field
user_entry = tk.Entry(window, width=30)
user_entry.pack(pady=10)

# Create send button
send_button = tk.Button(window, text="Send", command=send_message)
send_button.pack(pady=10)

# Start the GUI main loop
window.mainloop()

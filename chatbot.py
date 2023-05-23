from transformers import TFAutoModelForCausalLM, AutoTokenizer

# Load the pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
model = TFAutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Chatbot interaction loop
while True:
    user_input = input("You: ")

    # Tokenize user input
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="tf")

    # Generate model response
    response = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode and print the response
    chatbot_response = tokenizer.decode(response[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("ChatBot:", chatbot_response)

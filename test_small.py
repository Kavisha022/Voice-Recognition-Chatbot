from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("./t5_qa_model", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("./t5_qa_model")

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Prepare input
input_text = "question:What is F1-Score?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate answer
output_ids = model.generate(input_ids, max_new_tokens=64)
answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Answer:", answer)

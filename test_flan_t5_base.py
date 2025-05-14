from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = r"D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\voice recognition system\flan_t5_base_qa_model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

input_text = "Question: What is sentiment analysis?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
output = model.generate(input_ids, max_length=128)
answer = tokenizer.decode(output[0], skip_special_tokens=True)

print("Answer:", answer)
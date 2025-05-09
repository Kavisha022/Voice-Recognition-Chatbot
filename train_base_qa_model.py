from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Load custom QA data
# df = pd.read_excel(r"D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\voice recognition system\data_science_qa_200.xlsx")
df = pd.read_csv(r"D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\voice recognition system\DataScience QA.csv")
df['input_text'] = "Question: " + df['Question']
df['target_text'] = df['Answer']

# Convert to dataset format
dataset = Dataset.from_pandas(df[['input_text', 'target_text']])

# Tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Tokenize
def preprocess(example):
    model_inputs = tokenizer(example['input_text'], padding="max_length", truncation=True, max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example['target_text'], padding="max_length", truncation=True, max_length=128)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Tokenizing thevdataset
tokenized_dataset = dataset.map(preprocess, batched=True)

# Training
args = TrainingArguments(
    output_dir=r"D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\voice recognition system\t5_base_qa_model",
    per_device_train_batch_size=4,
    num_train_epochs=20,
    logging_steps=10,
    save_steps=10,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset
)

# Training the model
trainer.train()

# trainer.evaluate()
model.save_pretrained(r"D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\voice recognition system\t5_base_qa_model")
tokenizer.save_pretrained(r"D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\voice recognition system\t5_base_qa_model")
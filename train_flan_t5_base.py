from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Load your CSV
df = pd.read_csv(r"D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\voice recognition system\DataScience QA.csv")

# Prepare the input and target columns
df['input_text'] = "Question: " + df['Question'].astype(str)
df['target_text'] = df['Answer'].astype(str)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[['input_text', 'target_text']])

# Split into training and validation
dataset = dataset.train_test_split(test_size=0.2)

# Load tokenizer and model
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Preprocessing function
def preprocess(example):
    inputs = tokenizer(example['input_text'], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(example['target_text'], padding="max_length", truncation=True, max_length=128)
    inputs['labels'] = targets['input_ids']
    return inputs

# Tokenize the dataset
tokenized = dataset.map(preprocess, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=r"D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\voice recognition system\flan_t5_base_qa_model",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    logging_steps=5,
    save_steps=10,
    save_total_limit=1,
    #evaluation_strategy="epoch",
    logging_dir="./logs",
    #load_best_model_at_end=True,
    #greater_is_better=False,
    report_to="none",  # Set to "wandb" or "tensorboard" if needed
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
)

# Train the model
trainer.train()

# Save model and tokenizer
model.save_pretrained(r"D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\voice recognition system\flan_t5_base_qa_model")
tokenizer.save_pretrained(r"D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\voice recognition system\flan_t5_base_qa_model")

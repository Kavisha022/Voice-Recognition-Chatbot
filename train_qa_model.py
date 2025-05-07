from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
# from sklearn.model_selection import train_test_split
import pandas as pd
# import numpy as np
# import evaluate

# Load custom QA data
df = pd.read_excel(r"D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\voice recognition system\qa_data.xlsx")
df['input_text'] = "question: " + df['question']
df['target_text'] = df['answer']

# Convert to dataset format
dataset = Dataset.from_pandas(df[['input_text', 'target_text']])

# # Split into train and eval sets
# train_df, eval_df = train_test_split(df[['input_text', 'target_text']], test_size=0.2, random_state=42)
# train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
# eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))


# Tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Tokenize
# def preprocess(example):
#     model_inputs = tokenizer(example['input_text'], padding="max_length", truncation=True, max_length=64)
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(example['target_text'], padding="max_length", truncation=True, max_length=64)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

def preprocess(data):
    inputs = tokenizer(data['input_text'], padding="max_length", truncation=True, max_length=64)
    targets = tokenizer(data['target_text'], padding="max_length", truncation=True, max_length=64)
    inputs['labels'] = targets['input_ids']
    return inputs

# Tokenizing thevdataset
tokenized_dataset = dataset.map(preprocess, batched=True)

# # Tokenize datasets
# tokenized_train = train_dataset.map(preprocess, batched=True)
# tokenized_eval = eval_dataset.map(preprocess, batched=True)

# # Load metric
# metric = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     decoded_preds = tokenizer.batch_decode(np.argmax(predictions, axis=-1), skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#     return metric.compute(predictions=decoded_preds, references=decoded_labels)


# Training
args = TrainingArguments(
     output_dir="./t5_qa_model",
    # evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    # per_device_eval_batch_size=4,
    num_train_epochs=10,
    # logging_dir="./logs",
    # save_total_limit=1,
    logging_steps=10,
    save_steps=10,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=args,
    # train_dataset=tokenized_train,
    # eval_dataset=tokenized_eval,
    # compute_metrics=compute_metrics,
    train_dataset=tokenized_dataset
)

# Training the model
trainer.train()

# trainer.evaluate()
model.save_pretrained("./t5_qa_model")
tokenizer.save_pretrained("./t5_qa_model")


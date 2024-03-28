from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import AutoTokenizer

#sentences = ["<1>", "<3>", "<1>A beginning", "<1>A beginning is the time for taking the most delicate care", "<3>A beginning", "<3>A beginning is the time for taking the most delicate care"]
sentences = ["He was the"]
output_path = "distilbert/distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(output_path)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(output_path)

tokenized = tokenizer(sentences, return_tensors="pt", padding=True)
outputs = model.generate(**tokenized, max_new_tokens=30, pad_token_id = tokenizer.eos_token_id)
returned_outputs = tokenizer.batch_decode(outputs)
print(returned_outputs)
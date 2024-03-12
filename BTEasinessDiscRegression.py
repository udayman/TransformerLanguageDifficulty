from transformers import AutoTokenizer
import csv
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding

import evaluate
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from transformers import AdamW, get_linear_schedule_with_warmup

#dataset class for training with Hugging Face
class ClassificationDataset(Dataset):
    """Tokenize data when we call __getitem__"""
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
    	#tokenizer setting max length to 30
        inputs = self.tokenizer(self.data[i]['source'], truncation=True, max_length=30)
        inputs['labels'] = self.data[i]['target']
        return inputs
#data method to return training and test data
def return_data(path):
	with open(path, newline="", encoding="utf8") as f:
	  reader = csv.reader(f)
	  next(reader)
	  training_data = [{"source": row[14], "target": float(row[22])} for row in reader if row[-1] == "Train"]
	with open(path, newline="", encoding="utf8") as f:
	  reader = csv.reader(f)
	  next(reader)
	  test_data = [{"source": row[14], "target": float(row[22])} for row in reader if row[-1] == "Test"]
	return training_data, test_data

#metrics for training
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    result = dict()
    result["rmse"] = rmse
    result["mse"] = mse
    result["mae"] = mae

    return result

#CSV path
path = "CLEAR Corpus 6.01 - CLEAR Corpus 6.01.csv"

#Using DeBERTa small - defining tokenizer and model
model_path = 'microsoft/deberta-v3-small'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels = 1)

model.to("cuda")

#training and test dataset
ptraining_data, ptest_data = return_data(path)
training_data = []
test_data = []

for item in ptraining_data:
    item_source = item["source"]
    item_target = item["target"]
    item_source_split = item_source.split()
    num_source_item = len(item_source_split)
    for i in range(0, num_source_item - 30, 30):
        cur_element_source = " ".join(item_source_split[i:i+30])
        training_data.append({"source": cur_element_source, "target": item_target})

for item in ptest_data:
    item_source = item["source"]
    item_target = item["target"]
    item_source_split = item_source.split()
    num_source_item = len(item_source_split)
    for i in range(0, num_source_item - 30, 30):
        cur_element_source = " ".join(item_source_split[i:i+30])
        test_data.append({"source": cur_element_source, "target": item_target})

train_dataset = ClassificationDataset(training_data, tokenizer)
test_dataset = ClassificationDataset(test_data, tokenizer)

#collate data with padding to pad in a batch by batch basis
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#testing from here/commented out since only for testing purposes
'''
from torch.utils.data.dataloader import DataLoader
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, 
collate_fn=data_collator)

for step, batch in enumerate(train_dataloader): 
    batch.to("cuda") 
    print(batch)        
    outputs = model(**batch)
    print(outputs)
    quit()

quit()
'''
num_epochs = 2
optimizer = AdamW(model.parameters(), correct_bias='True', lr=5e-5)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataset) * num_epochs)

training_args = TrainingArguments(
   output_dir="classifier_current",
   learning_rate=2e-5,
   per_device_train_batch_size=3,
   per_device_eval_batch_size=3,
   num_train_epochs = num_epochs,
   weight_decay=0.01,
   evaluation_strategy="epoch",
   save_strategy="epoch",
   load_best_model_at_end=True,
)

trainer = Trainer(

   model=model,
   args=training_args,
   train_dataset=train_dataset,
   eval_dataset=test_dataset,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
   optimizers = (optimizer, lr_scheduler)
)

trainer.train()

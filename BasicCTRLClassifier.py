from transformers import AutoTokenizer
import csv
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from tqdm import tqdm
import os
import torch

import evaluate
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

from transformers import AdamW, get_linear_schedule_with_warmup
from utils import evaluate_list_quantile, evaluate_single_quantile, produce_quantiles, get_value_quantile_train
from torch.utils.data.dataloader import DataLoader

import warnings
warnings.filterwarnings("ignore")

#dataset class for training with Hugging Face
class CausalDataset(Dataset):
    """Tokenize data when we call __getitem__"""
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
    	#tokenizer setting max length to 256
        inputs = self.tokenizer(self.data[i], truncation=True, max_length=33)
        return inputs

#dataset class for evaluating with Hugging Face
class EvaluationDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
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

#CSV path
path = "CLEAR Corpus 6.01 - CLEAR Corpus 6.01.csv"

#Using DistilGPT2 - defining tokenizer and model
model_path = 'distilbert/distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to("cuda")

#load classifier
classifier_path = "classifier_20t"
classifier = AutoModelForSequenceClassification.from_pretrained(classifier_path)
classifier.to("cuda")
classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path)


#training and test dataset
ptraining_data, ptest_data = return_data(path)
training_data = []
test_data = []
training_eval = []

for item in ptraining_data:
    item_source = item["source"]
    item_target = item["target"]
    item_source_split = item_source.split()
    num_source_item = len(item_source_split)

    '''
    for i in range(0, num_source_item - 30, 30):
        cur_element_source = " ".join(item_source_split[i:i+30])
        training_data.append({"source": cur_element_source, "target": item_target})
    '''
    
    training_data.append({"source": " ".join(item_source_split[:20]), "target": item_target})

quantiles = produce_quantiles() 
training_data = ["<" + str(evaluate_single_quantile(quantiles, data["target"])) + ">" + data["source"] for data in training_data]

for item in ptest_data:
    item_source = item["source"]
    item_target = item["target"]

    item_source_split = item_source.split()
    cur_element_source = " ".join(item_source_split[:10])

    cur_element_source = cur_element_source
    cur_element_source = "<" + str(evaluate_single_quantile(quantiles, item_target)) + ">" + cur_element_source
    test_data.append({"source": cur_element_source, "target": get_value_quantile_train(quantiles, evaluate_single_quantile(quantiles, item_target))})

for item in ptraining_data:
    item_source = item["source"]
    item_target = item["target"]

    item_source_split = item_source.split()
    cur_element_source = " ".join(item_source_split[:10])

    cur_element_source = "<" + str(evaluate_single_quantile(quantiles, item_target)) + ">" + cur_element_source
    training_eval.append({"source": cur_element_source, "target": get_value_quantile_train(quantiles, evaluate_single_quantile(quantiles, item_target))})

train_dataset = CausalDataset(training_data, tokenizer)
test_dataset = EvaluationDataset(test_data, tokenizer)
training_evalset = EvaluationDataset(training_eval, tokenizer)

#collating model for language modeling
data_collator_eval = DataCollatorWithPadding(tokenizer=tokenizer)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

num_epochs = 5
optimizer = AdamW(model.parameters(), correct_bias='True', lr=5e-4)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataset) * num_epochs)

batch_size = 64
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
eval_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, collate_fn=data_collator_eval)
train_eval_dataloader = DataLoader(training_evalset, shuffle=False, batch_size=batch_size, collate_fn=data_collator_eval)

num_training_steps = int(len(train_dataloader) * num_epochs)

directory = "basic_classifier_CTRLfinetune"
if not os.path.exists(directory):
    # Create the directory
    os.makedirs(directory)

best_val_loss = float("inf")
progress_bar = tqdm(range(num_training_steps))
num_generated_tokens = 10

model.eval()
test_mse = 0
test_rmse = 0
test_mae = 0
for batch_i, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
    with torch.no_grad():
        batch_inputs = {'input_ids': batch['input_ids'].to("cuda"), 'attention_mask': batch['attention_mask'].to("cuda")}
        
        model_outputs = model.generate(**batch_inputs, max_new_tokens=num_generated_tokens, pad_token_id = tokenizer.eos_token_id)
        output_strings = tokenizer.batch_decode(model_outputs)
        
        inputs = classifier_tokenizer(output_strings, return_tensors="pt", padding=True)
        inputs.to("cuda")
        classifier_outputs = classifier(**inputs).logits
        classifier_outputs = classifier_outputs.cpu().flatten()
        labels = batch['labels']
        
        test_mse += mean_squared_error(labels, classifier_outputs)
        test_rmse += root_mean_squared_error(labels, classifier_outputs)
        test_mae += mean_absolute_error(labels, classifier_outputs)

test_mse = test_mse / len(eval_dataloader)
test_rmse = test_rmse / len(eval_dataloader)
test_mae = test_mae / len(eval_dataloader)
print(f"Validation mse: {test_mse}")
print(f"Validation rmse: {test_rmse}")
print(f"Validation mae: {test_mae}")

for epoch in range(num_epochs):
    # training
    model.train()
    for batch_i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        batch.to("cuda")

        output = model(**batch)

        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)

    # validation
    model.eval()
    test_mse = 0
    test_rmse = 0
    test_mae = 0
    for batch_i, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        with torch.no_grad():
            batch_inputs = {'input_ids': batch['input_ids'].to("cuda"), 'attention_mask': batch['attention_mask'].to("cuda")}
            
            model_outputs = model.generate(**batch_inputs, max_new_tokens=num_generated_tokens, pad_token_id = tokenizer.eos_token_id)
            output_strings = tokenizer.batch_decode(model_outputs)
            
            inputs = classifier_tokenizer(output_strings, return_tensors="pt", padding=True)
            inputs.to("cuda")
            classifier_outputs = classifier(**inputs).logits
            classifier_outputs = classifier_outputs.cpu().flatten()
            labels = batch['labels']
            
            test_mse += mean_squared_error(labels, classifier_outputs)
            test_rmse += root_mean_squared_error(labels, classifier_outputs)
            test_mae += mean_absolute_error(labels, classifier_outputs)

    test_mse = test_mse / len(eval_dataloader)
    test_rmse = test_rmse / len(eval_dataloader)
    test_mae = test_mae / len(eval_dataloader)
    print(f"Validation mse: {test_mse}")
    print(f"Validation rmse: {test_rmse}")
    print(f"Validation mae: {test_mae}")

    if test_mse < best_val_loss:
        model.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        best_val_loss = test_mse

    test_mse = 0
    test_rmse = 0
    test_mae = 0
    for batch_i, batch in tqdm(enumerate(train_eval_dataloader), total=len(train_eval_dataloader)):
        with torch.no_grad():
            batch_inputs = {'input_ids': batch['input_ids'].to("cuda"), 'attention_mask': batch['attention_mask'].to("cuda")}
            
            model_outputs = model.generate(**batch_inputs, max_new_tokens=num_generated_tokens, pad_token_id = tokenizer.eos_token_id)
            output_strings = tokenizer.batch_decode(model_outputs)
            
            inputs = classifier_tokenizer(output_strings, return_tensors="pt", padding=True)
            inputs.to("cuda")
            classifier_outputs = classifier(**inputs).logits
            classifier_outputs = classifier_outputs.cpu().flatten()
            labels = batch['labels']
            
            test_mse += mean_squared_error(labels, classifier_outputs)
            test_rmse += root_mean_squared_error(labels, classifier_outputs)
            test_mae += mean_absolute_error(labels, classifier_outputs)

    test_mse = test_mse / len(train_eval_dataloader)
    test_rmse = test_rmse / len(train_eval_dataloader)
    test_mae = test_mae / len(train_eval_dataloader)
    print(f"Training mse: {test_mse}")
    print(f"Training rmse: {test_rmse}")
    print(f"Training mae: {test_mae}")

#Replacing hugging face trainer with torch version

'''
training_args = TrainingArguments(
   output_dir="final_classifier",
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
'''
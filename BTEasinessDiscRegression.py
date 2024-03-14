from transformers import AutoTokenizer
import csv
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding

import evaluate
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data.dataloader import DataLoader
import os
import torch
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

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
        inputs = self.tokenizer(self.data[i]['source'], truncation=True, max_length=60)
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
    '''
    for i in range(0, num_source_item - 15, 50):
        cur_element_source = " ".join(item_source_split[i:i+50])
        training_data.append({"source": cur_element_source, "target": item_target})
    '''
    
    
    training_data.append({"source": " ".join(item_source_split[:20]), "target": item_target})

for item in ptest_data:
    item_source = item["source"]
    item_target = item["target"]
    item_source_split = item_source.split()
    
    num_source_item = len(item_source_split)
    '''
    for i in range(0, num_source_item - 15, 50):
        cur_element_source = " ".join(item_source_split[i:i+50])
        test_data.append({"source": cur_element_source, "target": item_target})
    '''
    
    
    test_data.append({"source": " ".join(item_source_split[:20]), "target": item_target})

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
num_epochs = 20
optimizer = AdamW(model.parameters(), correct_bias='True', lr=5e-5)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataset) * num_epochs)

batch_size = 64

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
eval_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)

num_training_steps = int(len(train_dataset)//batch_size * num_epochs)

directory = "classifier_50t"
if not os.path.exists(directory):
    # Create the directory
    os.makedirs(directory)

model.eval()

test_mse = 0
test_rmse = 0
test_mae = 0
test_loss = 0
for batch_i, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
    with torch.no_grad():
        batch.to("cuda")

        output = model(**batch)
        classifier_outputs = output.logits.cpu()
        labels = batch['labels'].cpu()

        test_mse += mean_squared_error(labels, classifier_outputs)
        test_rmse += root_mean_squared_error(labels, classifier_outputs)
        test_mae += mean_absolute_error(labels, classifier_outputs)
        test_loss += output.loss


test_mse = test_mse / len(eval_dataloader)
test_rmse = test_rmse / len(eval_dataloader)
test_mae = test_mae / len(eval_dataloader)
test_loss = test_loss / len(eval_dataloader)
print(f"Validation mse: {test_mse}")
print(f"Validation rmse: {test_rmse}")
print(f"Validation mae: {test_mae}")
print(f"Validation loss: {test_loss}")

best_val_loss = float("inf")
progress_bar = tqdm(range(num_training_steps))
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
    test_loss = 0
    for batch_i, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        with torch.no_grad():
            batch.to("cuda")

            output = model(**batch)
            classifier_outputs = output.logits.cpu()
            labels = batch['labels'].cpu()

            test_mse += mean_squared_error(labels, classifier_outputs)
            test_rmse += root_mean_squared_error(labels, classifier_outputs)
            test_mae += mean_absolute_error(labels, classifier_outputs)
            test_loss += output.loss


    test_mse = test_mse / len(eval_dataloader)
    test_rmse = test_rmse / len(eval_dataloader)
    test_mae = test_mae / len(eval_dataloader)
    test_loss = test_loss / len(eval_dataloader)
    print(f"Validation mse: {test_mse}")
    print(f"Validation rmse: {test_rmse}")
    print(f"Validation mae: {test_mae}")
    print(f"Validation loss: {test_loss}")

    if test_loss < best_val_loss:
        '''
        torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_loss': test_loss,
                },
                str(directory) + "/best_model.pt"
            )
        '''
        model.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        best_val_loss = test_loss


    test_mse = 0
    test_rmse = 0
    test_mae = 0
    test_loss = 0
    for batch_i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        with torch.no_grad():
            batch.to("cuda")

            output = model(**batch)
            classifier_outputs = output.logits.cpu()
            labels = batch['labels'].cpu()

            test_mse += mean_squared_error(labels, classifier_outputs)
            test_rmse += root_mean_squared_error(labels, classifier_outputs)
            test_mae += mean_absolute_error(labels, classifier_outputs)
            test_loss += output.loss

    test_mse = test_mse / len(eval_dataloader)
    test_rmse = test_rmse / len(eval_dataloader)
    test_mae = test_mae / len(eval_dataloader)
    test_loss = test_loss / len(eval_dataloader)
    print(f"Training mse: {test_mse}")
    print(f"Training rmse: {test_rmse}")
    print(f"Training mae: {test_mae}")
    print(f"Training loss: {test_loss}")


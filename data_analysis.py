from transformers import AutoTokenizer
import csv
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding

import evaluate
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, precision_score, recall_score, f1_score, accuracy_score

from utils import evaluate_list_quantile_eval, evaluate_single_quantile, produce_quantiles, get_value_quantile_train
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
	  training_data = [{"source": row[14], "target": float(row[22]), "se": float(row[23])} for row in reader if row[-1] == "Train"]
	with open(path, newline="", encoding="utf8") as f:
	  reader = csv.reader(f)
	  next(reader)
	  test_data = [{"source": row[14], "target": float(row[22]), "se": float(row[23])} for row in reader if row[-1] == "Test"]
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

def run_main(data_augmentation = False):
    #CSV path
    path = "CLEAR Corpus 6.01 - CLEAR Corpus 6.01.csv"

    #training and test dataset
    ptraining_data, ptest_data = return_data(path)

    training_data = []
    training_data_easy =0
    training_data_hard =0
    training_data_se = 0

    test_data = []
    test_data_easy =0
    test_data_hard =0
    test_data_se = 0

    training_eval = []
    training_eval_easy =0
    training_eval_hard =0
    training_eval_se =0

    quantiles = produce_quantiles()
    for item in ptraining_data:
        item_source = item["source"]
        item_target = item["target"]
        item_source_split = item_source.split()

        '''
        for i in range(0, num_source_item - 30, 30):
            cur_element_source = " ".join(item_source_split[i:i+30])
            training_data.append({"source": cur_element_source, "target": item_target})
        '''
        #if (evaluate_single_quantile(quantiles, item_target)) == 2:
            #continue
        
        if data_augmentation == False:
            training_data.append({"source": " ".join(item_source_split[:50]), "target": item_target})
            training_data_se += item["se"]
        else:
            num_source_item = len(item_source_split)
            for i in range(0, num_source_item , 50):
                cur_element_source = " ".join(item_source_split[i:i+50])
                training_data.append({"source": cur_element_source, "target": item_target})
                if (evaluate_single_quantile(quantiles, item_target)) == 1:
                    training_data_easy+=1
                elif (evaluate_single_quantile(quantiles, item_target)) == 3:
                    training_data_hard+=1
                training_data_se += item["se"]
 
    training_data = ["<" + str(evaluate_single_quantile(quantiles, data["target"])) + ">" + data["source"] for data in training_data]

    for item in ptest_data:
        item_source = item["source"]
        item_target = item["target"]

        #if (evaluate_single_quantile(quantiles, item_target)) == 2:
            #continue

        item_source_split = item_source.split()
        cur_element_source = " ".join(item_source_split[:1])

        cur_element_source = cur_element_source
        cur_element_source = "<" + str(evaluate_single_quantile(quantiles, item_target)) + ">" + cur_element_source
        test_data.append({"source": cur_element_source, "target": get_value_quantile_train(quantiles, evaluate_single_quantile(quantiles, item_target))})
        if (evaluate_single_quantile(quantiles, item_target)) == 1:
            test_data_easy+=1
        elif (evaluate_single_quantile(quantiles, item_target)) == 3:
            test_data_hard+=1
        test_data_se += item["se"]

    for item in ptraining_data:
        item_source = item["source"]
        item_target = item["target"]
        #if (evaluate_single_quantile(quantiles, item_target)) == 2:
            #continue

        item_source_split = item_source.split()
        cur_element_source = " ".join(item_source_split[:1])

        cur_element_source = "<" + str(evaluate_single_quantile(quantiles, item_target)) + ">" + cur_element_source
        training_eval.append({"source": cur_element_source, "target": get_value_quantile_train(quantiles, evaluate_single_quantile(quantiles, item_target))})
        if (evaluate_single_quantile(quantiles, item_target)) == 1:
            training_eval_easy+=1
        elif (evaluate_single_quantile(quantiles, item_target)) == 3:
            training_eval_hard+=1
        training_eval_se += item["se"]

    print("training data")
    print(len(training_data))
    print(training_data_easy)
    print(training_data_hard)
    print(training_data_se/len(training_data))
    print("test data")
    print(len(test_data))
    print(test_data_easy)
    print(test_data_hard)
    print(test_data_se/len(test_data))
    print("training data def")
    print(len(training_eval))
    print(training_eval_easy)
    print(training_eval_hard)
    print(training_eval_se/len(training_eval))


if __name__ == "__main__":
    run_main(data_augmentation = True)

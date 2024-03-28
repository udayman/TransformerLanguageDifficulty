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
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, precision_score, recall_score, f1_score, accuracy_score

from transformers import AdamW, get_linear_schedule_with_warmup
from utils import evaluate_list_quantile_eval, evaluate_single_quantile, produce_quantiles, get_value_quantile_train
from torch.utils.data.dataloader import DataLoader

import warnings
warnings.filterwarnings("ignore")


sentences = ["<1>", "<3>"]
output_path = "CTRL_beam_likelihood_1"
tokenizer = AutoTokenizer.from_pretrained(output_path)
model = AutoModelForCausalLM.from_pretrained(output_path)

tokenized = tokenizer(sentences, return_tensors="pt", padding=True)
outputs = model.generate(**tokenized, max_new_tokens=30, pad_token_id = tokenizer.eos_token_id)
returned_outputs = tokenizer.batch_decode(outputs)
print(returned_outputs)

classifier = "classifier_50t_aug"
tokenizer = AutoTokenizer.from_pretrained(classifier)
model = AutoModelForSequenceClassification.from_pretrained(classifier)
tokenized = tokenizer(returned_outputs, return_tensors="pt", padding=True)
outputs = model(**tokenized)
print(outputs.logits)


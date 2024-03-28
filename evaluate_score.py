from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import AutoTokenizer

import csv
import random

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

'''
path = "CLEAR Corpus 6.01 - CLEAR Corpus 6.01.csv"
ptraining_data, ptest_data = return_data(path)
random.shuffle(ptest_data)
evaluated_set = [" ".join(data["source"].split(" ")[:1]) for data in ptest_data[:5]]
print(evaluated_set)
'''

evaluated_set = ['What', 'It', 'To', 'We', 'Crows,']
all_models = ["basic_classifier_CTRLfinetune_1", "CTRL_likelihood_1", "CTRL_unlikelihood_1", "CTRL_beam_likelihood_1", "CTRL_beam_unlikelihood_1"]

random_shuffling = []

model_f = all_models[4]
tokenizer = AutoTokenizer.from_pretrained(model_f)
model = AutoModelForCausalLM.from_pretrained(model_f)

for i in range(5):
	print("New Set")
	print("-----------")
	starting_easy = "<1>" + evaluated_set[i]
	starting_hard = "<3>" + evaluated_set[i]
	easy_hard = [1,3]

	random.shuffle(easy_hard)
	if easy_hard[0] == 1:
		sents = [starting_easy, starting_hard]
	else:
		sents = [starting_hard, starting_easy]

	random_shuffling.append(easy_hard)
	tokenizer.padding_side='left'

	tokenized = tokenizer(sents, return_tensors="pt", padding=True)
	outputs = model.generate(**tokenized, max_new_tokens=30, pad_token_id = tokenizer.eos_token_id)
	returned_outputs = tokenizer.batch_decode(outputs)
	for i in returned_outputs:
		print("----")
		print(i[3:])

print(random_shuffling)
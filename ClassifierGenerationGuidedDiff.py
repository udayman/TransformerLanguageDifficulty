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
from utils import evaluate_list_quantile_eval, evaluate_single_quantile, produce_quantiles, get_value_quantile_train, return_difficulty
from torch.utils.data.dataloader import DataLoader

import warnings
warnings.filterwarnings("ignore")

#dataset class for evaluating with Hugging Face
class CausalDataset(Dataset):
    """Tokenize data when we call __getitem__"""
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        #tokenizer setting max length to 256
        inputs = self.tokenizer(self.data[i]['source'], truncation=True, max_length=60)
        inputs['indices'] = i
        inputs['midlabels'] = self.data[i]['target']
        inputs['prefix_length'] = self.data[i]['prefix_length']
        return inputs 

class EvaluationDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        inputs = self.tokenizer(self.data[i]['source'], truncation=True, max_length=60)
        inputs['labels'] = self.data[i]['target']
        inputs['prefix_length'] = self.data[i]['prefix_length']
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
def run_main(data_augmentation = False, classifier_path = "classifier_50t", output_path = "guided_CTRLfinetune_prompt"):
    path = "CLEAR Corpus 6.01 - CLEAR Corpus 6.01.csv"

    #Using DistilGPT2 - defining tokenizer and model
    model_path = 'distilbert/distilgpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to("cuda")

    #load classifier
    classifier = AutoModelForSequenceClassification.from_pretrained(classifier_path)
    classifier.to("cuda")
    classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path)


    #training and test dataset
    ptraining_data, ptest_data = return_data(path)
    training_data = []
    test_data = []
    training_eval = []

    quantiles = produce_quantiles() 

    for item in ptraining_data:
        item_source = item["source"]
        item_target = item["target"]
        item_source_split = item_source.split()
        num_source_item = len(item_source_split)

        if (evaluate_single_quantile(quantiles, item_target)) == 2:
            continue

        prefix = "This is written by a  " + return_difficulty(evaluate_single_quantile(quantiles, item_target)) + ": "

        if data_augmentation == False:
            training_data.append({"source": " ".join(item_source_split[:10]), "target": item_target, "prefix_length": len(prefix)})
        else:
            num_source_item = len(item_source_split)
            for i in range(0, num_source_item, 50):
                cur_element_source = " ".join(item_source_split[i:i+10])
                training_data.append({"source": prefix + cur_element_source, "target": item_target, "prefix_length": len(prefix)})

    training_data = [{"source": data["source"], "target": get_value_quantile_train(quantiles, evaluate_single_quantile(quantiles, data["target"])), "prefix_length":data["prefix_length"]} for data in training_data]

    for item in ptest_data:
        item_source = item["source"]
        item_target = item["target"]

        if (evaluate_single_quantile(quantiles, item_target)) == 2:
            continue

        item_source_split = item_source.split()
        cur_element_source = " ".join(item_source_split[:1])

        prefix = "This is written by a  " + return_difficulty(evaluate_single_quantile(quantiles, item_target)) + ": "
        cur_element_source = prefix + cur_element_source
        test_data.append({"source": cur_element_source, "target": get_value_quantile_train(quantiles, evaluate_single_quantile(quantiles, item_target)), "prefix_length":len(prefix)})

    for item in ptraining_data:
        item_source = item["source"]
        item_target = item["target"]

        if (evaluate_single_quantile(quantiles, item_target)) == 2:
            continue

        item_source_split = item_source.split()
        cur_element_source = " ".join(item_source_split[:1])

        prefix = "This is written by a  " + return_difficulty(evaluate_single_quantile(quantiles, item_target)) + ": "
        cur_element_source = prefix + cur_element_source
        training_eval.append({"source": cur_element_source, "target": get_value_quantile_train(quantiles, evaluate_single_quantile(quantiles, item_target)), "prefix_length":len(prefix)})

    train_dataset = CausalDataset(training_data, tokenizer)
    test_dataset = EvaluationDataset(test_data, tokenizer)
    training_evalset = EvaluationDataset(training_eval, tokenizer)

    #collating model for language modeling
    tokenizer.padding_side = 'left'
    data_collator_eval = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    num_epochs = 1
    optimizer = AdamW(model.parameters(), correct_bias='True', lr=5e-4)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataset) * num_epochs)

    batch_size = 64
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator)

    eval_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, collate_fn=data_collator_eval)
    train_eval_dataloader = DataLoader(training_evalset, shuffle=False, batch_size=batch_size, collate_fn=data_collator_eval)

    num_training_steps = int(len(train_dataloader) * num_epochs)

    directory = output_path
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)

    best_val_loss = 0
    progress_bar = tqdm(range(num_training_steps))
    num_generated_tokens = 30
    top_k = 10

    model.eval()
    test_mse = 0
    test_rmse = 0
    test_mae = 0
    test_accuracy = 0
    test_recall = 0
    test_f1 = 0

    for batch_i, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        with torch.no_grad():
            batch_inputs = {'input_ids': batch['input_ids'].to("cuda"), 'attention_mask': batch['attention_mask'].to("cuda")}
            
            model_outputs = model.generate(**batch_inputs, max_new_tokens=num_generated_tokens, pad_token_id = tokenizer.eos_token_id)
            output_strings = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
            output_strings = [output_strings[i][batch['prefix_length'][i]:] for i in range(len(output_strings))]
            
            inputs = classifier_tokenizer(output_strings, return_tensors="pt", padding=True)
            inputs.to("cuda")
            classifier_outputs = classifier(**inputs).logits
            classifier_outputs = classifier_outputs.cpu().flatten()
            labels = batch['labels']
            
            test_mse += mean_squared_error(labels, classifier_outputs)
            test_rmse += root_mean_squared_error(labels, classifier_outputs)
            test_mae += mean_absolute_error(labels, classifier_outputs)

            target_classes = evaluate_list_quantile_eval(quantiles, labels)
            classifier_classes = evaluate_list_quantile_eval(quantiles, classifier_outputs)
            test_accuracy += accuracy_score(target_classes, classifier_classes, normalize = True)
            test_recall += recall_score(target_classes, classifier_classes, average="macro")
            test_f1 += f1_score(target_classes, classifier_classes, average="macro")

    test_mse = test_mse / len(eval_dataloader)
    test_rmse = test_rmse / len(eval_dataloader)
    test_mae = test_mae / len(eval_dataloader)
    test_accuracy = test_accuracy / len(eval_dataloader)
    test_recall = test_recall / len(eval_dataloader)
    test_f1 = test_f1 / len(eval_dataloader)
    print(f"Validation mse: {test_mse}")
    print(f"Validation rmse: {test_rmse}")
    print(f"Validation mae: {test_mae}")
    print(f"Validation accuracy: {test_accuracy}")
    print(f"Validation recall: {test_recall}")
    print(f"Validation f1: {test_f1}")

    for epoch in range(num_epochs):
        # training
        model.train()
        total_loss = 0
        
        for batch_i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            batch_inputs = {'input_ids': batch['input_ids'].to("cuda"), 'attention_mask': batch['attention_mask'].to("cuda")}
            output_strings = [training_data[i]['source'] for i in batch['indices']]
            output_strings = [output_strings[i][batch['prefix_length'][i]:] for i in range(len(output_strings))]
            len_output_strings = len(output_strings)

            output = model(**batch_inputs)

            best_next_encodings = output.logits[:,-1,:]

            best_next_encodings_top_k = torch.topk(best_next_encodings, top_k, dim=1)
            softmax_next_top_k = torch.nn.functional.softmax(best_next_encodings_top_k[0], dim=1)
            next_top_k_inds = best_next_encodings_top_k[1]
            best_next_words = tokenizer.batch_decode(torch.flatten(next_top_k_inds).unsqueeze(1))

            string_index = 0
            total_loss = 0
            for i in range(len_output_strings):
                output_strings_topk = []
                for j in range(top_k):
                    output_strings_topk.append(output_strings[i] + " " + best_next_words[i*top_k + j])
                inputs = classifier_tokenizer(output_strings_topk, return_tensors="pt", padding=True)
                inputs.to("cuda")
                cur_logits = classifier(**inputs).logits
                cur_logits = cur_logits.flatten()
                
                l_loss_diff = torch.abs(cur_logits - torch.Tensor([batch['midlabels'][i] for _ in range(top_k)]).to("cuda"))
                l_loss_diff = torch.sum(l_loss_diff * softmax_next_top_k[i])

                u_loss_diff = torch.sum(-l_loss_diff * (1-softmax_next_top_k[i]))

                total_loss += l_loss_diff + u_loss_diff

            loss = total_loss/len_output_strings
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)

        print(f"Training loss: {total_loss/len(train_dataloader)}")

        # validation
        model.eval()
        test_mse = 0
        test_rmse = 0
        test_mae = 0
        test_accuracy = 0
        test_recall = 0
        test_f1 = 0
        for batch_i, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            with torch.no_grad():
                batch_inputs = {'input_ids': batch['input_ids'].to("cuda"), 'attention_mask': batch['attention_mask'].to("cuda")}
                
                model_outputs = model.generate(**batch_inputs, max_new_tokens=num_generated_tokens, pad_token_id = tokenizer.eos_token_id)
                output_strings = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
                output_strings = [output_strings[i][batch['prefix_length'][i]:] for i in range(len(output_strings))]
                
                inputs = classifier_tokenizer(output_strings, return_tensors="pt", padding=True)
                inputs.to("cuda")
                classifier_outputs = classifier(**inputs).logits
                classifier_outputs = classifier_outputs.cpu().flatten()
                labels = batch['labels']

                test_mse += mean_squared_error(labels, classifier_outputs)
                test_rmse += root_mean_squared_error(labels, classifier_outputs)
                test_mae += mean_absolute_error(labels, classifier_outputs)
                
                target_classes = evaluate_list_quantile_eval(quantiles, labels)
                classifier_classes = evaluate_list_quantile_eval(quantiles, classifier_outputs)
                test_accuracy += accuracy_score(target_classes, classifier_classes, normalize = True)
                test_recall += recall_score(target_classes, classifier_classes, average="macro")
                test_f1 += f1_score(target_classes, classifier_classes, average="macro")

        test_mse = test_mse / len(eval_dataloader)
        test_rmse = test_rmse / len(eval_dataloader)
        test_mae = test_mae / len(eval_dataloader)
        test_accuracy = test_accuracy / len(eval_dataloader)
        test_recall = test_recall / len(eval_dataloader)
        test_f1 = test_f1 / len(eval_dataloader)
        print(f"Validation mse: {test_mse}")
        print(f"Validation rmse: {test_rmse}")
        print(f"Validation mae: {test_mae}")
        print(f"Validation accuracy: {test_accuracy}")
        print(f"Validation recall: {test_recall}")
        print(f"Validation f1: {test_f1}")

        if test_accuracy > best_val_loss:
            print("Saved!-----------")
            model.save_pretrained(directory)
            tokenizer.save_pretrained(directory)
            best_val_loss = test_accuracy

        #training validation
        test_mse = 0
        test_rmse = 0
        test_mae = 0
        test_accuracy = 0
        test_recall = 0
        test_f1 = 0
        for batch_i, batch in tqdm(enumerate(train_eval_dataloader), total=len(train_eval_dataloader)):
            with torch.no_grad():
                batch_inputs = {'input_ids': batch['input_ids'].to("cuda"), 'attention_mask': batch['attention_mask'].to("cuda")}
                
                model_outputs = model.generate(**batch_inputs, max_new_tokens=num_generated_tokens, pad_token_id = tokenizer.eos_token_id)
                output_strings = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
                output_strings = [output_strings[i][batch['prefix_length'][i]:] for i in range(len(output_strings))]
                
                inputs = classifier_tokenizer(output_strings, return_tensors="pt", padding=True)
                inputs.to("cuda")
                classifier_outputs = classifier(**inputs).logits
                classifier_outputs = classifier_outputs.cpu().flatten()
                labels = batch['labels']

                test_mse += mean_squared_error(labels, classifier_outputs)
                test_rmse += root_mean_squared_error(labels, classifier_outputs)
                test_mae += mean_absolute_error(labels, classifier_outputs)
                
                target_classes = evaluate_list_quantile_eval(quantiles, labels)
                classifier_classes = evaluate_list_quantile_eval(quantiles, classifier_outputs)
                test_accuracy += accuracy_score(target_classes, classifier_classes, normalize = True)
                test_recall += recall_score(target_classes, classifier_classes, average="macro")
                test_f1 += f1_score(target_classes, classifier_classes, average="macro")

        test_mse = test_mse / len(train_eval_dataloader)
        test_rmse = test_rmse / len(train_eval_dataloader)
        test_mae = test_mae / len(train_eval_dataloader)
        test_accuracy = test_accuracy / len(train_eval_dataloader)
        test_recall = test_recall / len(train_eval_dataloader)
        test_f1 = test_f1 / len(train_eval_dataloader)
        print(f"Training mse: {test_mse}")
        print(f"Training rmse: {test_rmse}")
        print(f"Training mae: {test_mae}")
        print(f"Training accuracy: {test_accuracy}")
        print(f"Training recall: {test_recall}")
        print(f"Training f1: {test_f1}")

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

if __name__ == "__main__":
    print("-----------------new Run!")
    run_main(data_augmentation = False, classifier_path = "classifier_50t_aug", output_path = "guided_promptfinetune_1")
    print("-----------------new Run!")
    run_main(data_augmentation = True, classifier_path = "classifier_50t_aug", output_path = "guided_promptfinetune_2")
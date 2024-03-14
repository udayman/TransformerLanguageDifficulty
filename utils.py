import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


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

def evaluate_list_quantile(quantiles, list):
  ans = []
  for val in list:
    if val < quantiles[0]:
      ans.append(3)
    elif val <= quantiles[1]:
      ans.append(2)
    else:
      ans.append(1)
  return ans

def evaluate_list_quantile_eval(quantiles, list):
  ans = []
  for val in list:
    if val < (quantiles[0] + quantiles[1])/2:
      ans.append(3)
    else:
      ans.append(1)
  return ans

def evaluate_single_quantile(quantiles, val):
  if val < quantiles[0]:
    return 3
  elif val <= quantiles[1]:
    return 2
  else:
    return 1

def get_value_quantile_train(quantiles, val):
  if (val == 1):
    return quantiles[-1]
  elif (val == 2):
    return (quantiles[0] + quantiles[1])/2
  elif (val == 3):
    return quantiles[-2]

def get_value_quantile_evaluate(quantiles, val):
  if (val == 1):
    return (quantiles[-1] + quantiles[1])/2
  elif (val == 2):
    return (quantiles[0] + quantiles[1])/2
  elif (val == 3):
    return (quantiles[-2] + quantiles[0])/2

def return_difficulty(val):
  if (val == 1):
    return "1st grader"
  elif (val == 2):
    return "6th grader"
  else:
    return "PhD student"

def produce_quantiles():
  with open("CLEAR Corpus 6.01 - CLEAR Corpus 6.01.csv", newline="", encoding="utf8") as f:
    reader = csv.reader(f)
    next(reader)
    values = [float(row[22]) for row in reader if row[-1] == "Train"]
  values.sort()
  quantiles = [values[i*len(values)//3]for i in range(1,3)]
  quantiles.append(values[0])
  quantiles.append(values[-1])
  return quantiles


def check_regression_performance_on_quantiles(quantiles):
  classifier_path = "classifier_20t"
  classifier = AutoModelForSequenceClassification.from_pretrained(classifier_path)
  classifier.to("cuda")
  classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
  path = "CLEAR Corpus 6.01 - CLEAR Corpus 6.01.csv"

  with open(path, newline="", encoding="utf8") as f:
    reader = csv.reader(f)
    next(reader)
    test_data = [{"source": row[14], "target": float(row[22])} for row in reader if row[-1] == "Train"]

  test_dataset = ClassificationDataset(test_data, classifier_tokenizer)
  data_collator = DataCollatorWithPadding(tokenizer=classifier_tokenizer)

  batch_size = 64
  eval_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
  for batch_i, batch in enumerate(eval_dataloader):
        batch.to("cuda")

        output = classifier(**batch)
        classifier_outputs = output.logits.cpu()
        print(classifier_outputs)
        labels = batch['labels'].cpu()
        print(labels)
        quit()


if __name__ == "__main__":
  print(produce_quantiles())
  #check_regression_performance()
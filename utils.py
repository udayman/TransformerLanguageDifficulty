import csv

def evaluate_list_quantile(quantiles, list):
  ans = []
  for val in list:
    if val >= quantiles[0]:
      ans.append(1)
    elif val >= quantiles[1]:
      ans.append(2)
    elif val >= quantiles[2]:
      ans.append(3)
    elif val >= quantiles[3]:
      ans.append(4)
    else:
      ans.append(5)
  return ans

def evaluate_single_quantile(quantiles, val):
  if val >= quantiles[0]:
    return 1
  elif val >= quantiles[1]:
    return 2
  elif val >= quantiles[2]:
    return 3
  elif val >= quantiles[3]:
    return 4
  else:
    return 5

def produce_quantiles():
  with open("CLEAR Corpus 6.01 - CLEAR Corpus 6.01.csv", newline="", encoding="utf8") as f:
    reader = csv.reader(f)
    next(reader)
    values = [row[22] for row in reader if row[-1] == "Train"]
  values.sort()
  quantiles = [float(values[i*len(values)//5]) for i in range(1,5)]
  return quantiles


if __name__ == "__main__":
  print(produce_quantiles())
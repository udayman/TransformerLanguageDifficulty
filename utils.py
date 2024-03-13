import csv

def evaluate_list_quantile(quantiles, list):
  ans = []
  for val in list:
    if val <= quantiles[0]:
      ans.append(5)
    elif val <= quantiles[1]:
      ans.append(4)
    elif val <= quantiles[2]:
      ans.append(3)
    elif val <= quantiles[3]:
      ans.append(2)
    else:
      ans.append(1)
  return ans

def evaluate_single_quantile(quantiles, val):
  if val <= quantiles[0]:
    return 5
  elif val <= quantiles[1]:
    return 4
  elif val <= quantiles[2]:
    return 3
  elif val <= quantiles[3]:
    return 2
  else:
    return 1

def evaluate_midpoint_quantile(quantiles, val):
  if (val == 1):
    return (quantiles[3] + quantiles[-1])/2
  elif (val == 2):
    return (quantiles[2] + quantiles[3])/2
  elif (val == 3):
    return (quantiles[1] + quantiles[2])/2
  elif (val == 4):
    return (quantiles[0] + quantiles[1])/2
  elif (val == 5):
    return (quantiles[0] + quantiles[-2])/2

def produce_quantiles():
  with open("CLEAR Corpus 6.01 - CLEAR Corpus 6.01.csv", newline="", encoding="utf8") as f:
    reader = csv.reader(f)
    next(reader)
    values = [float(row[22]) for row in reader if row[-1] == "Train"]
  values.sort()
  quantiles = [values[i*len(values)//5]for i in range(1,5)]
  quantiles.append(values[0])
  quantiles.append(values[-1])
  return quantiles


if __name__ == "__main__":
  print(produce_quantiles())
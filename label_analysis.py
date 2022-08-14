import json
import matplotlib.pyplot as plt
import csv

def plot_label_counts(data_file, title):
  with open(data_file, 'r') as file:
    docs = json.load(file)

  label_counts = {}
  labels = []

  for doc in docs:
      annotations = doc['annotations'][0]['result']

      for annotation in annotations:
          label = annotation['value']['labels'][0]
          labels.append(label)

          if label in label_counts:
              label_counts[label] += 1
          else:
              label_counts[label] = 1

  for index,value in enumerate(label_counts.values()):
    plt.text(value, index, str(value))

  plt.barh(list(label_counts.keys()), label_counts.values())
  plt.xticks(rotation='vertical')
  plt.title(title)
  plt.show()

  return label_counts

def read_f1_per_label(filepath):

  results = {}

  with open(filepath+'/f1_per_label.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    row_count = 0
    for row in spamreader:
      if row_count <= 1:
        row_count +=1
        continue
      if row[4] == '':
        results[row[3]] = 0
      else:
        results[row[3]] = row[4]
      row_count +=1

  return results

def plot_f1_vs_label(f1s_dict_list, titles, label_counts):

  sorted_counts = {key: value for key, value in sorted(label_counts.items())}
  counts = [float(x) for x in sorted_counts.values()]
  
  fig, ax = plt.subplots()

  for i in range(len(titles)):
    sorted_f1 = {key: value for key, value in sorted(f1s_dict_list[i].items())}
    f1s = [float(x) for x in sorted_f1.values()]
    ax.scatter(counts, f1s, label=titles[i])

  # show point label
  # for i, txt in enumerate(list(sorted_f1.keys())):
      # ax.annotate(txt, (counts[i], f1s[i]))
  plt.legend()
  plt.show()
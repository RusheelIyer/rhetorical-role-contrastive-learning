import json
import random

file_name = 'paheli-data.json'
number_train_samples = 40
number_dev = 5

train_path = 'paheli-train.json'
dev_path = 'paheli-dev.json'
test_path = 'paheli-test.json'

with open(file_name, 'r') as file:
    annotated_docs = json.load(file)

random.shuffle(annotated_docs)

train_data = annotated_docs[:number_train_samples]
dev_data = annotated_docs[number_train_samples:number_train_samples+number_dev]
test_data = annotated_docs[number_train_samples+number_dev:]

print(len(train_data))
print(len(dev_data))
print(len(test_data))

with open(train_path, 'w') as train_file:
    json.dump(train_data, train_file)

with open(dev_path, 'w') as dev_file:
    json.dump(dev_data, dev_file)

with open(test_path, 'w') as test_file:
    json.dump(test_data, test_file)
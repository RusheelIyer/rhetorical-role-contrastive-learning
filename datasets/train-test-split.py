import json
import random

file_name = 'vetclaims-all-data.json'
number_train_samples = 60
number_dev = 8

train_path = 'vetclaims-train.json'
dev_path = 'vetclaims-dev.json'
test_path = 'vetclaims-test.json'

with open(file_name, 'r') as file:
    annotated_docs = json.load(file)

random.shuffle(annotated_docs)

train_data = annotated_docs[:number_train_samples]
dev_data = annotated_docs[number_train_samples:number_train_samples+number_dev]
test_data = annotated_docs[number_train_samples+number_dev:]

print("Train samples: ", len(train_data))
print("Dev samples: ", len(dev_data))
print("Test samples: ", len(test_data))

with open(train_path, 'w') as train_file:
    json.dump(train_data, train_file)

with open(dev_path, 'w') as dev_file:
    json.dump(dev_data, dev_file)

with open(test_path, 'w') as test_file:
    json.dump(test_data, test_file)
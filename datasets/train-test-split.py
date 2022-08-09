import json
import random

file_name = 'vetclaims-all-data.json'
number_train_samples = 65

train_path = 'vetclaims-train.json'
test_path = 'vetclaims-dev.json'

with open(file_name, 'r') as file:
    annotated_docs = json.load(file)

random.shuffle(annotated_docs)

train_data = annotated_docs[:number_train_samples]
test_data = annotated_docs[number_train_samples:]

with open(train_path, 'w') as train_file:
    json.dump(train_data, train_file)

with open(test_path, 'w') as test_file:
    json.dump(test_data, test_file)
import json
import os
import re
from tqdm import tqdm

dataset_path = 'datasets/paheli-dataset/'

directory = os.fsencode(dataset_path)

file_id = 1
sent_id = 1
json_files = []
for file in tqdm(os.listdir(directory)):
    filepath = os.fsdecode(file)
    if not filepath.endswith('.txt'):
        continue
    filename = filepath.split('.')[0]

    with open(dataset_path+filepath, 'r') as text_file:
        json_path = dataset_path+'json/'+filename+'.json'
        lines = text_file.readlines()

    annotations = [
        {
            "result": []
        }
    ]
    start = 0
    for line in tqdm(lines):
        sentence, label = line.split('\t')
        label = label.strip()

        line_dict = {
            'id': 's'+str(sent_id),
            'value': {
                "start": start,
                "end": start+len(sentence),
                "text": sentence,
                "labels": [label]
            }
        }

        start += len(sentence)
        sent_id += 1
        annotations[0]['result'].append(line_dict)

    file_dict = {
        'id': file_id,
        'annotations': annotations
    }
    
    json_files.append(file_dict)

    file_id +=1

with open('paheli-data.json', 'w') as outfile:
    json.dump(json_files, outfile)

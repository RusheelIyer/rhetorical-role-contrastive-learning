import json
import os

dataset_path = 'paheli-dataset/'

directory = os.fsencode(dataset_path)

file_id = 1
for file in os.listdir(directory):
    filepath = os.fsdecode(file)
    if not filepath.endswith('.txt'):
        continue
    filename = filepath.split('.')[0]

    with open(dataset_path+filepath, 'r') as text_file:
        json_path = dataset_path+'json/'+filename+'.json'

        file_dict = {
            'id': file_id,
            'data': {
                'text': text_file.read()
            }
        }

        with open(json_path, 'w') as outfile:
            json.dump(file_dict, outfile)

    file_id +=1
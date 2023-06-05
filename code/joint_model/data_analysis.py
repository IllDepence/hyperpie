import os 
import json
import numpy as np 
import pandas as pd 
from tabulate import tabulate


# deal with each item




if __name__ == '__main__':
    
    data_path = './data/'
    files = os.listdir(data_path)
    processed_data = {}
    for file in files:
        if file.endswith(".json"):
            fname = os.path.join(data_path, file)
            # read json
            with open(fname, 'r') as f:
                data = json.load(f)
            for i in range(len(data)):
                partial_data = data[i:i+1][0]
                # print(partial_data)
                if len(partial_data['annotation']['relations']) != 0:
                    # add label to relations type
                    for key in partial_data['annotation']['relations'].keys(): 
                        source = partial_data['annotation']['relations'][key]['source']
                        target = partial_data['annotation']['relations'][key]['target']
                        label = source[0]+target[0]
                        partial_data['annotation']['relations'][key]['label'] = label                
                # else: 
                #     # how much 'nan' should we keep?     
                #     partial_data['annotation']['relations'] = 'nan'
                    
                    # annotation --> entities + relations
                    anno_dict = partial_data.pop("annotation") 
                    partial_data.update(anno_dict)
                    #  entities[surface_forms] --> entities[text, start, end]
                    # surface_dic = partial_data.pop("surface_forms")
                    # partial_data.update(surface_dic)
                    del partial_data['annotation_raw']
                    processed_key = partial_data['document_id'] + '_' + str(partial_data['paragraph_index'])
                    processed_data[processed_key] = partial_data
                                      
            print(len(processed_data))            
    # save data
    processed_fname = './data/data.json'
    with open(processed_fname, 'w') as file:
        json.dump(processed_data, file)

            
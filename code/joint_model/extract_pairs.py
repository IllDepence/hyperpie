import os 
import json
import numpy as np
import pandas as pd 


if __name__ == '__main__':
    df = pd.read_json('./data/v2/tsa_processed.json')
    # only save the dataframe with at least one relation
    joint_dataset = {}

    num_parg = 0
    num_rels = 0

    for idx in df.index:
        text = df['text'][idx] 
        annotation = df['annotation'][idx]
        relations = annotation['relations']

        if relations:
            # save
            id = str(df['document_id'][idx]) + '_paragraph' + str(df['paragraph_index'][idx])
            info = {
                    'id': id,
                    #'phase': 'train',
                    #'paragraph': df['paragraph_index'][idx],
                    'text': df['text'][idx],
                    'entity': annotation['entities'],
                    'relation': annotation['relations'],
            }
            joint_dataset[id] = info
            num_parg += 1
            num_rels += len(relations)

    print('number of paragraph with relations: ',num_parg)
    print('number of relations: ',num_rels)

    entity_pair_list = []
    for paragraph in joint_dataset.keys():
        for rel in joint_dataset[paragraph]['relation']:
            text = joint_dataset[paragraph]['text']
            rel_context = joint_dataset[paragraph]['relation'][rel]

            # entity name (like entity type v1 p1 a1...)
            source = rel_context['source']
            target = rel_context['target']
            # entity informstion 
            entity_context = joint_dataset[paragraph]['entity']
            s_entity = entity_context[source]
            t_entity = entity_context[target]

            # entity name
            s_positon_start = s_entity['surface_forms'][0]['start']
            s_positon_end = s_entity['surface_forms'][0]['end']
            t_positon_start = t_entity['surface_forms'][0]['start']
            t_positon_end = t_entity['surface_forms'][0]['end']
            s_name = text[s_positon_start: s_positon_end]
            
            t_name = text[t_positon_start: t_positon_end]
            entity_pair = (s_name, t_name)
            entity_pair_list.append(entity_pair)
            print(entity_pair,([s_positon_start,s_positon_end], [t_positon_start,t_positon_end]) ,text)
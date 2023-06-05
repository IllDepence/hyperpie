import os 
import json 
import tqdm 
from datetime import datetime
from transformers import AutoModel, AutoConfig, AutoTokenizer

# %%%%%%%%%%%%%%%%%
def parse_file(fname, tokenizer=None, verbose=False):
    """
    fname: path of input file
    tokenizer: 
    verbose: print
    """
    
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")

    # load file
    loaded_file = json.load(open(fname, "r")) 
    # save results
    output = []

    kept_changed = [0, 0]
    for document_title in loaded_file:
        document = loaded_file[document_title]
        # print(document)
        parsed_document = {}

        # tokenize text
        tokenized_text = tokenizer.encode_plus(document['text'], return_offsets_mapping=True)
        token_bounds = []
        token_start = {}
        token_end = {}
        for i, osm in enumerate(tokenized_text['offset_mapping'][1:-1]):
            token_bounds.extend(osm)
            token_start[osm[0]] = i+1
            token_end[osm[1]] = i+1

        parsed_document['id'] = document['document_id']
        parsed_document['source'] = document
        parsed_document['source']['file'] = os.path.basename(fname)
        parsed_document['input_ids'] = list(tokenized_text['input_ids'])
        parsed_document['offset_mapping'] = list(tokenized_text['offset_mapping'])

        token_bounds = sorted(token_bounds)

        # generate labels, if we have any
        # extract entity spans from file
        print(document.keys())
        if 'entities' in document.keys():
            # print(document.keys())
            # create entity labels on token level
            entities = document['entities']
            entity_bounds = []
            entity_labels = []
            entities_by_name = {}
            #print(token_start)
            for ent in entities.values():
                # print('surface_forms:  ',ent['surface_forms'][0]['start'])
                surface_forms = ent['surface_forms'][0]
                # start, end = ent['start'], ent['end'] 
                
                start, end = surface_forms['start'], surface_forms['end'] 
                
                if surface_forms['start'] not in token_start.keys():
                    for x in reversed(sorted(token_start.keys())):
                        if x < surface_forms['start']:
                            start = x
                            break
                if surface_forms['end'] not in token_end.keys():
                    for x in sorted(token_end.keys()):
                        if x > surface_forms['end']:
                            end = x
                            break
                    if end == surface_forms['end']:
                        end = sorted(token_end.keys())[-1]
            
                if (start, end) != (surface_forms['start'], surface_forms['end']):
                    if verbose:
                        print(f"adjusted entity bounds due to tokenization '{document['text'][ent['start']:ent['end']]}' {(ent['start'], ent['end'])} -> '{document['text'][start:end]}' {(start, end)}")
                    # print(len(document['text']))
                    kept_changed[1]+=1
                else:
                    kept_changed[0]+=1
                entity_bounds.append(start)
                entity_bounds.append(end)
                entity_labels.append([token_start[start], token_end[end], document['text'][start:end], ent['type']])

                entities_by_name[ent['id']] = [token_start[start], token_end[end], document['text'][start:end], ent['type']]

            parsed_document['entities'] = entity_labels

            # create relation labels on token level
            relation_labels = []
            # print('entities_by_name:::,,,', entities_by_name)
            
            for relation in document['relations'].values():
                try:
                    label = {
                        "r": relation['label'],
                        "h": entities_by_name[relation['source']][:2],
                        "type_h": entities_by_name[relation['source']][3],
                        "t": entities_by_name[relation['target']][:2],
                        "type_t": entities_by_name[relation['target']][3],
                        "evidence": 0,
                        }
                except:
                    # print('Not exsit!!!')
                    print(relation)
                    # print(document['entity'])
                    raise
                print(entities_by_name[relation['source']])
                relation_labels.append(label)
            print('relation_labels:::::, ', relation_labels)
            parsed_document['relations'] = relation_labels
        else:            
            parsed_document['relations'] = []

        output.append(parsed_document)
    if verbose:
        print(kept_changed)
    return output


if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
    
    data_path = './data/'
    train_files = os.listdir(data_path)
    parsed_files = []
    for file in train_files:
        if file.endswith(".json"):
            fname = os.path.join(data_path, file)
            output = parse_file(fname, tokenizer=tokenizer)
            # parsed_files.extend(output)
    # print(len(output))
    # print(output[0])
    

# %%
import os
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

import pylatexenc # LaTex to Unicode



os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        # print(document.keys())
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
                    # print(relation)
                    # print(document['entity'])
                    raise
                # print(entities_by_name[relation['source']])
                relation_labels.append(label)
            
            # print('relation_labels:::::, ', relation_labels)
            parsed_document['relations'] = relation_labels
        else:            
            parsed_document['relations'] = []

        output.append(parsed_document)
    if verbose:
        print(kept_changed)
    return output



def recursive_parse(nodelist, chardict, math_map, symbol_candidates):
    for node in nodelist:
        #print(node)
        if node is None:
            continue
        if type(node) == pylatexenc.latexwalker.LatexCharsNode:
            for i, char in enumerate(node.chars):
                chardict[node.pos+i] = char

            if math_map[node.pos] == 1:
                # we are in math mode
                pass

        elif type(node) == pylatexenc.latexwalker.LatexSpecialsNode:
            for i, char in enumerate(node.specials_chars):
                chardict[node.pos+i] = char
        elif type(node) == pylatexenc.latexwalker.LatexMacroNode:
            if node.macroname == "cite":
                continue
            for i, char in enumerate(node.macroname):
                chardict[node.pos+i] = char
            
            if math_map[node.pos] == 1:
                # we are in math mode
                pass

            if node.nodeargd is not None:
                recursive_parse(node.nodeargd.argnlist, chardict, math_map, symbol_candidates)
        elif type(node) == pylatexenc.latexwalker.LatexMathNode:
            math_map[node.pos:node.pos+node.len] = [1 for _ in range(node.len)]
            math_map[node.pos] = 2
            math_map[node.pos+node.len-1] = 2
            recursive_parse(node.nodelist, chardict, math_map, symbol_candidates)
            symbol_candidates.append((node.pos+len(node.delimiters[0]), node.pos+node.len-len(node.delimiters[1])))
            
            chardict[node.pos] = "$"
            chardict[node.pos+node.len-1] = "$"

            """
            if chardict[max(0, node.pos-1)] != None:
                for i, char in enumerate(node.delimiters[0]):
                    chardict[node.pos+i] = char
                for i, char in enumerate(node.delimiters[1]):
                    chardict[node.pos+node.len-len(node.delimiters[1])+i] = char
            """
        elif type(node) == pylatexenc.latexwalker.LatexGroupNode:
            recursive_parse(node.nodelist, chardict, math_map, symbol_candidates)
            if chardict[max(0, node.pos-1)] != None:
                #print(node.delimiters)
                chardict[node.pos] = node.delimiters[0]
                chardict[node.pos+node.len-1] = node.delimiters[1]
        elif type(node) == pylatexenc.latexwalker.LatexEnvironmentNode:
            recursive_parse(node.nodelist, chardict, math_map, symbol_candidates)
        else:
            #print("ignored:",node)
            pass

def tokenize_and_segment_symbols(s):
    # tokenize text and math separately while keeping track of the mapping to the original character indexing
    latexwalker = pylatexenc.latexwalker.LatexWalker(s)
    cleaned_chars = {}
    for i in range(len(s)):
        cleaned_chars[i] = None
    symbol_candidates = []
    math_map = [0 for _ in cleaned_chars.keys()]

    recursive_parse(latexwalker.get_latex_nodes()[0], cleaned_chars, math_map, symbol_candidates)

    # for symbols which are surrounded by $...$, $ needs to be part of the annotated span. APPARENTLY NOT, according to annotation guide... but in the test data it sometimes is...
    temp = []
    for sc in symbol_candidates:
        if math_map[sc[0]-1] == 2 and math_map[sc[1]] == 2:
            #print("yes:",s[sc[0]-1:sc[1]+1], math_map[sc[0]-1:sc[1]+1], math_map[sc[0]-1], math_map[sc[1]])
            temp.append((sc[0]-1, sc[1]+1))
        temp.append(sc)
    symbol_candidates = temp

    for i in cleaned_chars.keys():
        if math_map[i] != 0:
            cleaned_chars[i] = s[i]
    # print("".join([str(x) for x in math_map]))
    #print([s[i] for i,x in enumerate(math_map) if x == 2])

    return [(i,x) for i,x in enumerate(cleaned_chars.values()) if x is not None]



def generate_candidate_spans(num_tokens, max_len=15):
    output = []
    for i in range(1, num_tokens-1):
        for j in range(i, min(i+max_len, num_tokens-1)):
            output.append([i, j])
    return output

class Collate_Fn_Manager:

    def __init__(self, max_span_len=15):
        self.max_span_len = max_span_len

    def collate_fn(self, batch):
        max_len = max([len(x['input_ids']) for x in batch])

        input_ids = [b["input_ids"] + [0] * (max_len - len(b["input_ids"])) for b in batch]
        attention_masks = [[1.0] * len(b["input_ids"]) + [0.0] * (max_len - len(b["input_ids"])) for b in batch]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.float)

        relation_labels = [b["relations"] for b in batch]

        candidate_spans = [generate_candidate_spans(len(x['input_ids'])) for x in batch]
        offset_mapping = [x['offset_mapping'] for x in batch]
        ids = [x['id'] for x in batch]
        sources = [x['source'] for x in batch]

        return input_ids, attention_masks, candidate_spans, relation_labels, offset_mapping, ids, sources

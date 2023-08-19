import json
import os
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification


def prep_data(doc_key, sents, ner, rel):
    """ Given a paragrapgh’s sentences w/ NER and REL info,
        generate X and y for training as well as a mapping
        from sample to entity offset pair.
    """

    X = []
    y = []
    sample_idx_to_entity_offset_pair = []

    # collect entity pairs
    send_offset = 0
    for sent_idx in range(len(sents)):
        # get tokens, entities, and relations for this sentence
        tkns = sents[sent_idx]
        ents = ner[sent_idx]
        rels = rel[sent_idx]
        pairs = []
        have_rel = []
        # get all entity candidate pairs
        for ei, ent_from_raw in enumerate(ents):
            for ej, ent_to_raw in enumerate(ents):
                if ei == ej:
                    # no self-relations
                    continue
                # only collect raw information here, do determine
                # derivative features such as relative distance later
                ent_from = {
                    'tokens': tkns[
                        ent_from_raw[0]-send_offset:
                        ent_from_raw[1]+1-send_offset
                    ],
                    'start': ent_from_raw[0],
                    'end': ent_from_raw[1],
                    'type': ent_from_raw[2],
                }
                ent_to = {
                    'tokens': tkns[
                        ent_to_raw[0]-send_offset:
                        ent_to_raw[1]+1-send_offset
                    ],
                    'start': ent_to_raw[0],
                    'end': ent_to_raw[1],
                    'type': ent_to_raw[2],
                }
                pairs.append((ent_from, ent_to))
                # check if there is a relation between these two entities
                rel_check = [
                    ent_from_raw[0],
                    ent_from_raw[1],
                    ent_to_raw[0],
                    ent_to_raw[1],
                    'USED-FOR'
                ]
                have_rel.append(rel_check in rels)
                print(
                    f'{ent_from["tokens"]} -> {ent_to["tokens"]}\n'
                    f'({ent_from["type"]} -> {ent_to["type"]})\n'
                    f'  rel: {rel_check in rels}'
                )
                input()
        send_offset += len(tkns)

    # process entity pair features

    # - create flat representation
    # - calculate derivative features such as relative distance
    # - normalize features to [0, 1]

    return X, y, sample_idx_to_entity_offset_pair


with open('10foldcross/fold_0/train.jsonl') as f:
    train = [json.loads(line) for line in f]

smpl = train[76]
prep_data(smpl['doc_key'], smpl['sentences'], smpl['ner'], smpl['relations'])

import json
import sys


def get_surface_forms(entity_dict):
    surface_forms = [
        f'"{sf["surface_form"]}"' for sf in entity_dict["surface_forms"]
    ]
    return " / ".join(surface_forms)


def print_chain(entities, relation):
    source_id = relation["source"]
    target_id = relation["target"]
    print(f'[{target_id}] {get_surface_forms(entities[target_id])}\n↑')
    print(f'[{source_id}] {get_surface_forms(entities[source_id])}\n')


def print_annot(data):
    # Then, we'll extract the entities and relations from the annotation
    entities = data["annotation"]["entities"]
    relations = data["annotation"]["relations"]

    # Finally, we'll print all the chains of relations
    for relation in relations.values():
        print_chain(entities, relation)


if __name__ == '__main__':
    fp = sys.argv[1]
    with open(fp) as f:
        data = json.load(f)
    if type(data) == list:
        data = data[0]
    print_annot(data)

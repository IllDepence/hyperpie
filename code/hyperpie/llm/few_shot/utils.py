# Utility functions
#


from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List
from tqdm.auto import tqdm

import faiss
import numpy as np
import yaml

from langchain.vectorstores import VectorStore, FAISS
from langchain.docstore.document import Document

from .sonar import SonarEmbeddings


def load_data(data_path: Path) -> List[dict]:
    """Load json annotation data
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def construct_docs(data: List[dict]) -> List[Document]:
    """Construct documents set
    """
    docs = []
    for name, row in enumerate(data):
        metadata = {
            "row": name,
            "source": row['document_id'],
            "paragraph_index": row['paragraph_index'],
        }
        for doc_id in set([doc['document_id'] for doc in data]):
            metadata[doc_id] = "test" if doc_id == row['document_id'] else "train"

        # Note: in vicuna, '\n' = '\x0a' is a special token for LineFeed
        text = row['text'].replace('\n', ' ').replace('\xa0', ' ')
        doc = Document(page_content=text, metadata=metadata)
        docs.append(doc)
    return docs


def search_neighbours(docs: List[dict], vectorstore_path: Path, top_k: int=10) -> List[Document]:
    """Search similar paragraphs
    """
    def _get_vectorstore(vectorstore_path: Path):
        # We use SONAR to embed the texts
        # See https://github.com/facebookresearch/SONAR

        embeddings = SonarEmbeddings(source_lang="eng_Latn")
        if vectorstore_path.is_dir():
            vectorstore = FAISS.load_local(vectorstore_path.as_posix(), embeddings)
        else:
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(vectorstore_path.as_posix())
        return vectorstore

    def _extract_vectors(vectorstore: VectorStore) -> np.ndarray:
        assert isinstance(vectorstore, FAISS)
        num_docs = vectorstore.index.ntotal
        emb_dim = vectorstore.index.d
        xb = vectorstore.index.get_xb()
        return faiss.rev_swig_ptr(xb, num_docs*emb_dim).reshape(num_docs, emb_dim)

    vectorstore = _get_vectorstore(vectorstore_path)
    emb = _extract_vectors(vectorstore)
    print('emb', emb.shape)
    for doc in tqdm(docs):
        i = doc.metadata['row']

        retrieved = vectorstore.similarity_search_by_vector(
            emb[i, :], k=top_k*10, filter={doc.metadata['source']: "train"}
        )
        rows = [row.metadata['row'] for row in retrieved if row.metadata['row'] != i]
        #assert len(rows) == len(retrieved)
        #if len(rows) < top_k:
        #    print(i, len(rows))

        doc.metadata['neighbours'] = rows[:top_k]
    return docs



class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)

def get_yaml(annotation: Dict) -> str:
    """Get yaml style formatting of an annotation
    """
    lookup = {}
    for k, v in annotation['entities'].items():
        lookup[k] = v['surface_forms'][0]['surface_form']

    tmp = {k: defaultdict(list) for k, _ in lookup.items() if k[0] in 'apv'}
    for _, v in annotation['relations'].items():
        s = v['source']
        t = v['target']
        if t.startswith('a') or t.startswith('p'):
            if s.startswith('p'):
                k = 'params'
            elif s.startswith('v'):
                k = 'values'
        elif s.startswith('c'):
            k = 'contexts'
        else:
            print(s, t)
        tmp[t][k].append(s)

    const = {'has_entities': False, 'entities': []}
    for k, v in tmp.items():
        if k.startswith('a'):
            e = {
                'id': k,
                'name': '"'+lookup[k]+'"',
                'has_parameters': True if 'params' in v else False,
            }
            if 'params' in v:
                params = []
                for param in v['params']:
                    p = {
                        'id': param,
                        'name': '"'+lookup[param]+'"',
                        'has_values': True if 'values' in tmp[param] else False,
                    }
                    if 'values' in tmp[param]:
                        values = []
                        for value in tmp[param]['values']:
                            c = {
                                'value_id': value,
                                'value': '"'+lookup[value]+'"',
                                'context': None,
                                'context_id': None,
                            }
                            if 'contexts' in tmp[value]:
                                context = tmp[value]['contexts'][0]
                                c['context'] = '"'+lookup[context]+'"'
                                c['context_id'] = context

                            values.append({f'value{value[1:]}': c})
                        p['values'] = values
                    params.append({f'parameter{param[1:]}': p})
                e['parameters'] = params
            const['entities'].append({f'entity{k[1:]}': e})

    if len(const['entities']) > 0:
        const['has_entities'] = True
    else:
        const = {'has_entities': False}

    yaml_str = str(yaml.dump(const, sort_keys=False, Dumper=IndentDumper))
    return yaml_str.replace("'\"", "\"").replace("\"'", "\"")


def yaml2json(yaml_string: str) -> str:
    """Convert yaml string to json string
    """
    yaml_string = yaml_string.replace('\\"', '').replace('\\', '\\\\')
    json_string = json.dumps(yaml.load(yaml_string, Loader=yaml.SafeLoader), indent=2)
    return json_string.replace('\\\\', '\\') + '\n'

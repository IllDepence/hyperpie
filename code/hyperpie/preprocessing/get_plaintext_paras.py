""" Filter unarXivw 2022 papers for categories
        cs.CL, cs.LG, cs.CV, cs.DL, cs.CV
    and generate paper derivates with plain text
    paragraphs usable for distant supervision.
"""

import json
import os
import re
import sys


def generate_cit_marker_lookup_dict(ref):
    """ For a given reference section, assign a citation marker
        in the form of [<counter>] to each reference entry.
    """

    citation_marker_replacements = {}
    for i, ref_id in enumerate(ref.keys()):
        citation_marker_replacements[ref_id] = "[" + str(i) + "]"
    return citation_marker_replacements


def generate_formula_lookup_dict(ref):
    """ Fill the (global) lookup dictionary for formulas
    """

    formulas = {}
    for ref_id, entries in ref.items():
        if entries['type'] == 'formula':
            formulas[ref_id] = "\(" + entries['latex'] + "\)"
    return formulas


def replace_tags(text, tag_type, values):
    pattern = re.compile(f"{{{{{tag_type}:(.+?)}}}}")
    return pattern.sub(lambda x: values[x.group(1)], text)


def replace_tabsandfig(text, tag_type):
    pattern = re.compile(f"{{{tag_type}:(.+?)}}")
    if tag_type == "figure":
        return pattern.sub("FIGURE", text)
    elif tag_type == "table":
        return pattern.sub("TABLE", text)


def transform_ppr(ppr_dict):
    formulas = generate_formula_lookup_dict(ppr_dict['ref_entries'])
    citation_marker_replacements = generate_cit_marker_lookup_dict(
        ppr_dict['bib_entries']
    )

    paras = ppr_dict['body_text']

    plain_paras = []
    for para in paras:
        # get paragraph text
        plain_context = para['text']
        # replace markers of referenced non-text content with actual content
        plain_context = replace_tags(plain_context, "formula", formulas)
        plain_context = replace_tags(
            plain_context, "cite", citation_marker_replacements
        )
        plain_context = replace_tabsandfig(plain_context, "figure")
        plain_context = replace_tabsandfig(plain_context, "table")
        plain_paras.append(plain_context)

    transformed_ppr = {
        'id': ppr_dict['paper_id'],
        'categories': ppr_dict['metadata']['categories'],
        'paragraphs': plain_paras,
    }

    return transformed_ppr


def generate(in_fp):
    categories = ['cs.CL', 'cs.LG', 'cs.CV', 'cs.DL', 'cs.CV']
    out_fn = 'transformed_pprs.jsonl'

    # get all JSONL files from in_dir
    ppr_fns = []
    for root, dirs, files in os.walk(in_fp):
        for file in files:
            if file.endswith(".jsonl"):
                ppr_fns.append(
                    os.path.join(root, file)
                )

    for ppr_fn in ppr_fns:
        print(f"Processing {ppr_fn.split('/')[-1]}")
        with open(os.path.join(in_fp, ppr_fn), 'r') as f:
            for line in f:
                ppr_dict = json.loads(line)
                prime_cat = ppr_dict['metadata']['categories'].split(' ')[0]
                if prime_cat not in categories:
                    continue
                transformed_ppr = transform_ppr(ppr_dict)
                with open(out_fn, 'a') as out_f:
                    json.dump(transformed_ppr, out_f)
                    out_f.write('\n')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python3 get_plaintext_paras.py /path/to/unarXive/')
        sys.exit(1)
    in_fp = sys.argv[1]
    generate(in_fp)

""" Filter unarXive 2022 papers for categories
        cs.CL, cs.LG, cs.CV, cs.DL, cs.CV
    and generate paper derivates with plain text
    paragraphs usable for distant supervision.

    result
    $ wc -l transformed_pprs.jsonl
    143203 transformed_pprs.jsonl
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


def _hyperparam_paragraph(para, formulas):
    """ Gauge if a paragraph is likely to contain hyperparameter information.
    """

    sec_title = para['section']
    sec_text = para['text']

    if sec_title is None or sec_title == "":
        # we need a section title
        return False

    # if paragraph is too short or too long, treat it as “noise”
    # (enumeration items, broken section cut-offs, etc.)
    if len(sec_text) < 200 or len(sec_text) > 2000:
        return False

    # we don't want to include paragraphs that only refer to tables
    # or report on metrics
    negative_keywords = [
        'table',
        'achieve',
        'metric',
        'measure'
    ]
    if any(kw in sec_text.lower() for kw in negative_keywords):
        return False

    # check section title for keywords
    sec_title_keywords = [
        'hyperparameter'
        'hyper-parameter',
        'hyper parameter',
        'model parameters',
        'training parameters',
        'training setup',
        'training settings',
        'training configuration'
        # - considered but in the end not used -
        # 'experiment setup',
        # 'experimental setup',
        # 'experiment settings',
        # 'experimental settings',
        # 'baseline',
        # 'implementation details',
        # 'dataset',
        # 'training details',
    ]

    promising_sec_title = any(
        kw in sec_title.lower() for kw in sec_title_keywords
    )

    # check section text for keywords, values, or mathematical notation
    sec_text = replace_tags(sec_text, "formula", formulas)
    text_assigner_keywords = [
        ' use ', ' used', ' using', ' set ', '='
    ]
    text_value_patt = re.compile(r'(=|\s|\\\()\d+(\s|,|\.|;|e|\^|\\\))')
    promising_sec_text = (
        any(
            kw in sec_text.lower() for kw in text_assigner_keywords
        )
        and
        text_value_patt.search(sec_text)
    )

    return promising_sec_title and promising_sec_text


def transform_ppr(ppr_dict, pre_filter=False):
    formulas = generate_formula_lookup_dict(ppr_dict['ref_entries'])
    citation_marker_replacements = generate_cit_marker_lookup_dict(
        ppr_dict['bib_entries']
    )

    paras = ppr_dict['body_text']

    plain_paras = []
    for para in paras:
        if pre_filter and not _hyperparam_paragraph(para, formulas):
            continue
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


def generate(in_fp, pre_filter=False):
    """ Generate transformed papers with plain text paragraphs.

        If pre_filter is True, heuristically filter for paragraphs
        that are likely to contain descriptions of hyperparameters.
    """

    categories = ['cs.CL', 'cs.LG', 'cs.CV', 'cs.DL', 'cs.CV']
    out_fn = 'transformed_pprs.jsonl'
    if pre_filter:
        out_fn = 'transformed_pprs_filtered.jsonl'

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
                prime_cat = ppr_dict['metadata'].get(
                    'categories', ''
                ).split(' ')[0]
                if prime_cat not in categories:
                    continue
                transformed_ppr = transform_ppr(ppr_dict, pre_filter)
                if len(transformed_ppr['paragraphs']) == 0:
                    continue
                with open(out_fn, 'a') as out_f:
                    json.dump(transformed_ppr, out_f)
                    out_f.write('\n')


if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print(
            'Usage: python3 get_plaintext_paras.py '
            '/path/to/unarXive/ [pre_filter]'
        )
        sys.exit(1)
    in_fp = sys.argv[1]
    if len(sys.argv) == 3 and sys.argv[2] == 'pre_filter':
        print('- * - * - * - * - * - * - * - * - * - * - *')
        print('pre-filtering for hyperparameter paragraphs')
        print('- * - * - * - * - * - * - * - * - * - * - *')
        pre_filter = True
    else:
        pre_filter = False
    generate(in_fp, pre_filter)

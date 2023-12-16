""" Pre filter sentences in papers heuristically such that the resulting
    sentences have a high likelyhood to contain descriptions of research
    artifacts, their parameters and the values these are set to.
"""


import os
import pickle
import re
import random
import sqlite3
import sys
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from xml import etree


def load_formulas_for_doc(db_cur, aid):
    """ Load math notation contens from external SQLite DB.
    """

    db_cur.execute('select * from formula where in_doc=?', (aid, ))
    rows = db_cur.fetchall()
    formula_dict = dict()
    for r in rows:
        formula_dict[r[0]] = {
            'id': r[0],
            'in_doc': r[1],
            'latex': r[2],
            'mathml': r[3]
        }
    return formula_dict


def load_sentences_for_doc(base_path, fn):
    """ Read paper txt and tokenize into sentences.

        Return a list of (<section>, <sentence>) tuples.
    """

    sents = []
    with open(os.path.join(base_path, fn)) as f:
        txt = f.read()
        # ugly custom preprocecssing of some common abbreviations
        multipunkt_abbrv = ['etal', 'wrt', 'idd', 'eg', 'ie', 'cf']
        for mpa in multipunkt_abbrv:
            txt = re.sub(
                '\.?\s*'.join(list(mpa)) + '\.',
                mpa,
                txt,
                flags=re.I
            )
        singlepunkt_abbrv = ['fig', 'sec', 'chap', 'tab', 'tbl']
        for spa in singlepunkt_abbrv:
            txt = re.sub(
                spa + '\.',
                spa,
                txt,
                flags=re.I
            )
        # | using PunktSentenceTokenizer with custom
        # | PunktParameter abbrev_types makes the tokenizer fail on
        # v cases that work with the out of the box sent_tokenize
        sents_pre = sent_tokenize(txt)
        sents = []
        section = 'DOC_START'
        for sent_pre in sents_pre:
            # | this fails for cases where an entity annotation is within
            # v the section header
            sec_start_match = re.search(
                (
                    r'^<(?:sub)?(?:sub)?section>'     # starting tag
                    r'([^<]+)'                        # section header
                    r'</(?:sub)?(?:sub)?section>\n?'  # ending tag
                ),
                sent_pre.strip()
            )
            if sec_start_match:
                sent_without_sec_head = sent_pre[sec_start_match.end():]
                section = sec_start_match.group(1)
                sents.append(
                    (section, sent_without_sec_head)
                )
            else:
                sents.append(
                    (section, sent_pre)
                )
    return sents


def process_paper(base_path, db_cur, fn):
    """ Given a paper’s file name, go through the contents sentence by
        sentence and pre-filter for sentences likely to contain math
        notation that describes used research artifacts more closely
        (e.g. parameters of a model/method).
    """

    # regex
    cite_patt = re.compile(r'\{\{cite:([^\}]+)\}\}')
    formula_patt = re.compile(r'\{\{formula:([a-zA-Z0-9-]{36})\}\}')
    figure_patt = re.compile(r'\{\{figure:([a-zA-Z0-9-]{36})\}\}')
    table_patt = re.compile(r'\{\{table:([a-zA-Z0-9-]{36})\}\}')
    artifact_patt = re.compile(r'<([a-z]{4,7})_used>([^<]+)</[a-z]{4,7}_used>')
    number_patt = re.compile(r'\d')

    # markers around math notation and artifacts for manual inspection
    dspl_sent_math_pre = None  # gets set individually later
    dspl_sent_math_post = '</span>'
    dspl_sent_enti_pre = '<span class="entity">'
    dspl_sent_enti_post = '</span>'

    # get sentence tokenized paper and math notation contents
    aid = fn[:-4]
    formula_dict = load_formulas_for_doc(db_cur, aid)
    sents = load_sentences_for_doc(base_path, fn)
    display_sentences = []
    candidate_sentences = []
    entity_type_counts = defaultdict(int)

    # iterate over sentences
    for s_idx, s_tpl in enumerate(sents):
        section, s = s_tpl
        sent_id = '{}_{}'.format(aid, s_idx)
        # require math notation and at least one number in the sentence.
        has_math = formula_patt.search(s)
        has_display_math = False
        has_num = False  # search later in formula plain render + normal text
        has_cit_marker = bool(cite_patt.search(s))
        has_var_assignment = False  # for heuristic pre-selection
        quick_math = False          # 2 + 2 = 4, - 1 = 3
        mfs = []
        mas = []
        display_sentence = [s]
        prev_cut_end = 0
        # process math notation
        for num_f_repl, m in enumerate(formula_patt.finditer(s)):
            f_dict = formula_dict[m.group(1)]
            f_dict['span'] = m.span(0)
            # mathml hotfix
            if f_dict['mathml'][:15] == 'mode="display">':
                mathjax_start = '$$'
                mathjax_end = '$$'
                has_display_math = True
                mathml_inner = f_dict['mathml'][15:]
                # FIXME: is the closing >> here a bug?    v
                f_dict['mathml'] = '<mathml mode="display">>' + \
                    mathml_inner + '</mathml>'
            else:
                mathjax_start = '\\('
                mathjax_end = '\\)'
                f_dict['mathml'] = '<mathml>' + f_dict['mathml'] + '</mathml>'
            mathml_tree = etree.ElementTree.fromstring(
                f_dict['mathml']
            )
            plain_math = ''.join(mathml_tree.itertext())
            f_dict['plain'] = plain_math
            mfs.append(f_dict)
            if not has_num:
                has_num = number_patt.search(plain_math)

            # display_sentence[-1] is the unprocessed part
            pre = s[prev_cut_end:m.start()]
            post = s[m.end():]
            prev_cut_end = m.end()

            dspl_sent_math_pre = '<span class="math" id="{}">'.format(
                f_dict['id']
            )
            math_markup = '{} {}{}{} {}'.format(
                dspl_sent_math_pre,
                mathjax_start,
                f_dict['latex'],
                mathjax_end,
                dspl_sent_math_post
            )
            display_sentence = \
                display_sentence[:-1] + \
                [
                    pre,  # text (might contain annotated entities)
                    (f_dict['id'], math_markup),  # math tuple
                    post  # unprocessed text
                ]

            # # variable assignments (for heuristic filtering)
            if len(plain_math) < 15 and '=' in f_dict['latex']:
                has_var_assignment = True
            if len(plain_math) < 7:
                quick_math = True

        # indicator phrases (for heuristic filtering)
        has_indicator_phraes = False
        if re.search(r'we(\s+have)? (use|chos|set)', s, flags=re.I) or \
                'set to' in s or \
                'set the' in s or \
                ('set' in s and 'parameter' in s) or \
                ('use' in s and 'parameter' in s) or \
                ('set' in s and 'factor' in s) or \
                ('use' in s and 'factor' in s) or \
                ('use' in s and 'setting' in s) or \
                ('set' in s and 'value' in s) or \
                ('use' in s and 'value' in s) or \
                (quick_math and '=' in s):
            has_indicator_phraes = True

        # process research artifacts
        has_annotated_artifact = False
        num_a_repl = 0
        for m in artifact_patt.finditer(s):
            num_a_repl += 1
            has_annotated_artifact = True
            mas.append({
                'type': m.group(1),
                'name': m.group(2),
                'tag_span': m.span(0),
                'name_span': m.span(2)
            })
            entity_type_counts[m.group(2)] += 1

        # process and other markup in display sentence
        for i, fragment in enumerate(display_sentence):
            if type(fragment) == tuple:
                # already processed math notation, skip
                continue
            # process annotated artifacts
            for m in artifact_patt.finditer(fragment):
                entity_markup = '{}{}{}'.format(
                    dspl_sent_enti_pre,
                    m.group(2),
                    dspl_sent_enti_post
                )
                pre = fragment[:m.start()]
                post = fragment[m.end():]
                display_sentence = display_sentence[:i] + \
                    [
                        pre,
                        (m.group(1), entity_markup),
                        post
                    ] + \
                    display_sentence[i+1:]

        for i, fragment in enumerate(display_sentence):
            if type(fragment) == str:
                nice_frag = cite_patt.sub('[CIT]', fragment)
                nice_frag = figure_patt.sub('(FIG)', nice_frag)
                nice_frag = table_patt.sub('(TAB)', nice_frag)
                display_sentence[i] = nice_frag
                if not has_num:
                    has_num = number_patt.search(nice_frag)

        display_sentences.append(
            (section, display_sentence)
        )

        if has_math and \
                has_num and \
                has_indicator_phraes and \
                not has_display_math:
            candi = {
                'id': sent_id,
                'in_doc': aid,
                'sent_idx': s_idx,
                'heuristic_filtering': {
                    'has_var_assignment': has_var_assignment,
                    'has_indicator_phraes': has_indicator_phraes,
                    'has_annotated_artifact': has_annotated_artifact,
                    'has_cit_marker': has_cit_marker
                },
                'section': section,
                'display_sentence': display_sentence,
                'raw_sentence': s,
                'formulas': mfs,
                'artifacts': mas
            }
            candidate_sentences.append(candi)

    res = {
        'sentences': sents,
        'display_sentences': display_sentences,
        'candidate_sentences': candidate_sentences,
        'entity_type_counts': entity_type_counts.copy()
    }

    return res


def persist_sample(base_path, fns, result_id):
    # full DB for retrieving formulas
    conn_full = sqlite3.connect(
        os.path.join(
            base_path,
            'formulas.sqlite'
        )
    )
    db_cur_full = conn_full.cursor()
    # filtered DB to fill
    conn_filt = sqlite3.connect(
        'sample_{}_formulas_filtered.sqlite'.format(
            result_id
        )
    )
    db_cur_filt = conn_filt.cursor()
    # create table
    db_cur_filt.execute(
        (
            "create table formula("  # FIXME: table already exists!?
            "'id' text, 'in_doc' text, 'latex' text, 'mathml' text)"
        )
    )
    # fill table
    with open('sample_{}_paper_ids'.format(result_id), 'w') as f:
        for fn in fns:
            if os.path.splitext(fn)[-1] != '.txt':
                continue
            aid = fn[:-4]
            f.write('{}\n'.format(aid))
            formula_dict = load_formulas_for_doc(db_cur_full, aid)
            for row in formula_dict.values():
                db_cur_filt.execute(
                    (
                        "insert into formula "
                        "('id','in_doc','latex','mathml')"
                        "values(?,?,?,?)"
                    ),
                    (
                        row['id'],
                        row['in_doc'],
                        row['latex'],
                        row['mathml']
                    )
                )
    conn_filt.commit()


def pre_filter(base_path, result_id, sample_size, demo=False):
    # setup
    if demo:
        base_path = (
            '../2022_update/unarXive_2022_wip_'
            '2018data_parsed_pwcmatched'
        )
    fns = os.listdir(base_path)

    conn = sqlite3.connect(os.path.join(base_path, 'formulas.sqlite'))
    db_cur = conn.cursor()

    # stats
    num_sents_total = 0
    num_candi_total = 0
    entity_type_counts_total = defaultdict(int)
    res_all = dict()

    random.seed(27)
    random.shuffle(fns)

    if sample_size > 0:
        # sample papers
        fns = fns[:sample_size]
        # FIXME ↓
        # prior to the cut, fns contains *not only* the .txt files
        # here, but also a .json and .tsv file for most papers
        # sample size will therefore roughly be a third of what is
        # given as the sample size parameter
        # FIXME ↑

        # create sample specific formulas DB
        persist_sample(base_path, fns, result_id)
        pkl_fn = 'sample_{}_pre_filtered_results.pkl'.format(result_id)
    else:
        pkl_fn = 'pre_filtered_results_{}.pkl'.format(result_id)
    # for all papers in given directory
    for fn in fns:
        if os.path.splitext(fn)[-1] != '.txt':
            continue
        res = process_paper(base_path, db_cur, fn)
        res_all[fn] = res
        num_sents_total += len(res['sentences'])
        num_candi_total += len(res['candidate_sentences'])
        for k, v in res['entity_type_counts'].items():
            entity_type_counts_total[k] += v

        if demo:
            # manual preview
            for candi in res['candidate_sentences']:
                print(candi['heuristic_filtering'])
                print()
                print(candi['in_doc'], candi['sent_idx'], candi['section'])
                print(candi['display_sentence'])
                print()
                input()
    print(len(res_all))

    with open(pkl_fn, 'wb') as f:
        pickle.dump(res_all, f)
    res_dev = dict()
    i = 0
    for k, v in res_all.items():
        res_dev[k] = v
        i += 1
        if i > 25:
            break
    with open(pkl_fn.replace('.pkl', '_dev.pkl'), 'wb') as f:
        pickle.dump(res_dev, f)

    return res_all


if __name__ == '__main__':
    base_path = sys.argv[1]
    result_id = sys.argv[2]
    if len(sys.argv) == 4:
        sample_size = int(sys.argv[3])
    else:
        sample_size = -1
    pre_filter(base_path, result_id, sample_size)

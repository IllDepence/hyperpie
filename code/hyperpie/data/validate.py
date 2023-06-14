""" Sub-package for validation of data.
"""

from collections import defaultdict
from hyperpie.data import load as load_data


def _val_processed_surf_offsets_single(para):
    """ Validate entity surface form offsets for a single paragraph.
    """

    text = para['text']
    num_correct_sufs = 0
    num_faulty_sufs = 0
    faulty_entities = []

    for entity_id, entity in para['annotation']['entities'].items():
        surface_forms = entity['surface_forms']
        faulty_surfs = []
        for surface_form in surface_forms:
            start = surface_form['start']
            end = surface_form['end']
            text_attrib = surface_form['surface_form']
            text_offset = text[start:end]
            if text_attrib != text_offset:
                num_faulty_sufs += 1
                faulty_surfs.append({
                    'id': surface_form['id'],
                    'start': start,
                    'end': end,
                    'text_attrib': text_attrib,
                    'text_offset': text_offset
                })
            else:
                num_correct_sufs += 1
        if len(faulty_surfs) > 0:
            faulty_entities.append({
                'entity_id': entity['id'],
                'faulty_surfs': faulty_surfs
            })

    return faulty_entities, num_correct_sufs, num_faulty_sufs


def validate_surface_form_offsets(paras):
    """ For all entities in all paragraphs, check if the offsets
        given for entity surface forms actually match with the
        paragraph text.
    """

    faulty_docs = defaultdict(list)
    num_correct_sufs = 0
    num_faulty_sufs = 0

    for para in paras:
        doc_id = para['document_id']
        para_index = para['paragraph_index']
        faulty_ents, cor, fal = _val_processed_surf_offsets_single(para)
        num_correct_sufs += cor
        num_faulty_sufs += fal
        if len(faulty_ents) > 0:
            faulty_docs[doc_id].append({
                'paragraph_index': para_index,
                'faulty_entities': faulty_ents
            })

    # report
    report_str = ''
    if len(faulty_docs) > 0:
        report_str += 'Found faulty surface form offsets:\n'
        report_str += f'  Correct: {num_correct_sufs}\n'
        report_str += f'  Faulty: {num_faulty_sufs}\n'
        report_str += f'  Faulty documents: {len(faulty_docs)}\n'
        for doc_id, faulty_paras in faulty_docs.items():
            report_str += f'  {doc_id}\n'
            for faulty_para in faulty_paras:
                report_str += \
                    f'    Paragraph {faulty_para["paragraph_index"]}\n'
                for faulty_entity in faulty_para['faulty_entities']:
                    report_str += \
                        f'      Entity {faulty_entity["entity_id"]}\n'
                    for fsurf in faulty_entity['fsurfs']:
                        report_str += \
                            f'        Surface form {fsurf["id"]}:\n'
                        report_str += \
                            f'          start: {fsurf["start"]}\n'
                        report_str += \
                            f'          end: {fsurf["end"]}\n'
                        report_str += \
                            f'          text_attrib: {fsurf["text_attrib"]}\n'
                        report_str += \
                            f'          text_offset: {fsurf["text_offset"]}\n'

    return report_str


def validate_all_surface_form_offsets():
    data_variants = [
        load_data.load_annotated(),
        load_data.load_annotated(only_full=True),
        load_data.load_annotated(with_parent=True)
    ]
    data_var_names = [
        'processed',
        'processed_onlyfull',
        'processed_withparent'
    ]

    reports_digest = []
    reports_full = []
    for i, paras in enumerate(data_variants):
        report = validate_surface_form_offsets(paras)
        if len(report) > 0:
            reports_digest.append(f'Found faulty surface form offsets(s) '
                                  f'in {data_var_names[i]} data.')
            reports_full.append(report)
        else:
            reports_digest.append(f'No faulty surface form offsets(s) '
                                  f'in {data_var_names[i]} data.')
            reports_full.append('')

    print('\n\n'.join(reports_digest))
    if len(reports_full) > 0:
        choice = input('Show full reports? (y/n) ')
        if choice == 'y':
            print('\n\n'.join(reports_full))

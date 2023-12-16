import json
import re
from difflib import SequenceMatcher

annot_patt = re.compile(r'\[([a-z])([0-9\.]+)\|([^]]+)]')

with open('annot_comp_pair.json') as f:
    annot_comp_pairs = json.load(f)

llm_text = annot_comp_pairs['annot']
orig_text = annot_comp_pairs['orig']

# strip llm text of annotations
# replacing e.g. [a1|text] with text
llm_text = annot_patt.sub(r'\3', llm_text)

# llm_text and orig_text differ only slightly by a few characters
# such as an additional or missing space, some non-ascii characters
# being added or missing etc.

# create a character level mapping from llm_text to orig_text, such
# that the characters in llm_text are mapped to the corresponding
# characters in orig_text
# i.e., if llm_text = 'abc' and orig_text = 'abbc', then the mapping
# would be {0:0, 1:1, 2:3}

# make the necessary imports

# create a mapping from llm_text to orig_text
llm_to_orig = {}

# get non-overlapping matching subsequences
blocks = SequenceMatcher(None, llm_text, orig_text).get_matching_blocks()

# from blocks, build a complete mapping from llm_text to orig_text
# i.e. every character in llm_text is mapped to a character in orig_text
for i, j, n in blocks:
    for k in range(n):
        llm_to_orig[i+k] = j+k

# print result as
# llm_text_index> llm_text_char orig_text_char <orig_text_index
for k, v in llm_to_orig.items():
    print(f'{k}> {llm_text[k]} {orig_text[v]} <{v}')

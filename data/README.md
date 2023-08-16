* **Manually annotated ground truth**
    * raw: tsa.json
    * **main**: tsa\_processed.json
    * filtered variants:
        * tsa\_processed\_withparent.json (see `code/hyperpie/data/filter_annots.py:require_apv_single()`)
        * tsa\_processed\_onlyfull.json (see `code/hyperpie/data/filter_annots.py:require_parent_single()`)
    * converted variants:
        * PL-Marker format: tsa\_processed\_plmarker.jsonl
* **Pilot study data**
    * annot\_pilot
* **Distant supervision**
    * raw: transformed\_pprs\_filtered.jsonl.xz
    * annotated by LLM: transformed\_pprs\_filtered\_llm\_annotated.jsonl.xz
* **Data used during development**
    * eval\_test\_data

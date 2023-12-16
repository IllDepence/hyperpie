<a name="README"></a>

* **Manually annotated ground truth**
    * raw: tsa.json
    * **main**: tsa\_processed.json ([see format description](preprocessed_data_format.md))
    * filtered variants:
        * tsa\_processed\_withparent.json (see `code/hyperpie/data/filter_annots.py:require_apv_single()`)
        * tsa\_processed\_onlyfull.json (see `code/hyperpie/data/filter_annots.py:require_parent_single()`)
    * converted variants:
        * PL-Marker format: tsa\_processed\_plmarker.jsonl
        * PL-Marker format (5 fold cross eval data splits): 5fold\_strat.tar.gz
* **Pilot study data**
    * annot\_pilot
* **Distant supervision** (not used in final experiments)
    * raw: transformed\_pprs\_filtered.jsonl.xz
    * annotated by LLM: transformed\_pprs\_filtered\_llm\_annotated.jsonl.xz

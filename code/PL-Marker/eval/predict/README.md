### file notes

* `predict/`
    * contains files used for prediction

* `sample_idx_to_entity_offset_pair.json`
    * map from sample indices to document/sample ID, sentence index, relation offsets
* `y_pred.json`
    * JSON list w/ predictions (i.e. decisions if relation between entity pair exists)

### use

* filter samples from `sample_idx_to_entity_offset_pair.json` for only predicted ones using `y_pred.json`
* go though paragraphs (=lines) in `predict/all/ent_pred_test.json`
    - fetch all samples where `doc_key` and `para_idx` match
    - insert into predicted rels at sentence number of sample

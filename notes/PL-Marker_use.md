# PoC test runs

## Cheat data splits (train=dev=test)

### Run NER

```
CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_0/ --learning_rate 2e-5 --num_train_epochs 15 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file train.jsonl --test_file train.jsonl --output_dir ./sciner-hyperpie_cheat --output_results
```

**Results**

```
08/09/2023 17:27:22 - INFO - __main__ -   ***** Running evaluation  *****
08/09/2023 17:27:22 - INFO - __main__ -     Num examples = 1980
08/09/2023 17:27:22 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|████████████████████████████████████████████| 124/124 [00:34<00:00,  3.63it/s]
08/09/2023 17:27:56 - INFO - __main__ -     Evaluation done in total 34.221572 secs (57.858241 example per second)
08/09/2023 17:27:56 - INFO - __main__ -   Result: {"f1": 0.985498108448928, "f1_overlap": 0.985498108448928, "precision": 1.0, "recall": 0.9714108141702921}
08/09/2023 17:27:56 - INFO - __main__ -   Result: {"dev_best_f1": 0.985498108448928, "f1_": 0.985498108448928, "f1_overlap_": 0.985498108448928, "precision_": 1.0, "recall_": 0.9714108141702921}
```

### Copy intermediate results

`cp sciner-hyperpie_cheat/ent_pred_* hyperpie/fold_0/`

### Run RE

```
CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_0 --learning_rate 2e-5  --num_train_epochs 15 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16 --train_file train.jsonl --test_file ent_pred_test.json --dev_file ent_pred_dev.json --use_ner_results --output_dir ./scire-hyperpie_cheat
```

**Result**

```
08/09/2023 17:41:53 - INFO - __main__ -   ***** Running evaluation  *****
08/09/2023 17:41:53 - INFO - __main__ -     Batch size = 16
08/09/2023 17:41:53 - INFO - __main__ -     Num examples = 1552
Evaluating: 100%|██████████████████████████████████████████████| 97/97 [00:04<00:00, 22.52it/s]
08/09/2023 17:41:57 - INFO - __main__ -     Evaluation done in total 4.328220 secs (374.056798 example per second)
08/09/2023 17:41:57 - INFO - __main__ -   Result: {"f1": 0.9097222222222222, "f1_with_ner": 0.9097222222222222, "ner_f1": 0.9904458598726115}
{'dev_best_f1': 0.9097222222222222, 'f1_': 0.9097222222222222, 'f1_with_ner_': 0.9097222222222222, 'ner_f1_': 0.9904458598726115}
```

## Proper data splits

### Run NER

```
CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --
model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_0/ --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file dev.jsonl --test_file test.jsonl --output_dir ./sciner-hyperpie_toast --output_result
```

**Results**

```
08/09/2023 19:18:15 - INFO - __main__ -   ***** Running evaluation  *****
08/09/2023 19:18:15 - INFO - __main__ -     Num examples = 270
08/09/2023 19:18:15 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|██████████████████████████████████████████████| 17/17 [00:05<00:00,  3.28it/s]
08/09/2023 19:18:20 - INFO - __main__ -     Evaluation done in total 5.197367 secs (51.949381 example per second)
08/09/2023 19:18:20 - INFO - __main__ -   Result: {"f1": 0.8640776699029126, "f1_overlap": 0.8687350835322196, "precision": 0.8599033816425121, "recall": 0.8682926829268293}
08/09/2023 19:18:21 - INFO - __main__ -   Result: {"dev_best_f1": 0.7849829351535836, "f1_": 0.8640776699029126, "f1_overlap_": 0.8687350835322196, "precision_": 0.8599033816425121, "recall_": 0.8682926829268293}
```

### Copy over intermediate results

`cp sciner-hyperpie_toast/ent_pred_* hyperpie/fold_0/`

### Run RE

```
CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_0 --learning_rate 2e-5  --num_train_epochs 10 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16 --train_file train.jsonl --test_file ent_pred_test.json --dev_file ent_pred_dev.json --use_ner_results --output_dir ./scire-hyperpie_toast
```

**Results**

```
08/09/2023 19:40:57 - INFO - __main__ -   ***** Running evaluation  *****
08/09/2023 19:40:57 - INFO - __main__ -     Batch size = 16
08/09/2023 19:40:57 - INFO - __main__ -     Num examples = 207
Evaluating: 100%|██████████████████████████████████████████████| 13/13 [00:00<00:00, 13.95it/s]
08/09/2023 19:40:58 - INFO - __main__ -     Evaluation done in total 0.936744 secs (246.598786 example per second)
08/09/2023 19:40:58 - INFO - __main__ -   Result: {"f1": 0.0, "f1_with_ner": 0.0, "ner_f1": 0.8661800486618005}
{'dev_best_f1': 0.2702702702702703, 'f1_': 0.0, 'f1_with_ner_': 0.0, 'ner_f1_': 0.8661800486618005}
```

... re-run w/ 25 epochs

```
08/09/2023 20:01:03 - INFO - __main__ -   ***** Running evaluation  *****
08/09/2023 20:01:03 - INFO - __main__ -     Batch size = 16
08/09/2023 20:01:03 - INFO - __main__ -     Num examples = 207
Evaluating: 100%|██████████████████████████████████████████████| 13/13 [00:00<00:00, 15.48it/s]
08/09/2023 20:01:04 - INFO - __main__ -     Evaluation done in total 0.846159 secs (272.998478 example per second)
08/09/2023 20:01:04 - INFO - __main__ -   Result: {"f1": 0.0, "f1_with_ner": 0.0, "ner_f1": 0.8661800486618005}
{'dev_best_f1': 0.22857142857142856, 'f1_': 0.0, 'f1_with_ner_': 0.0, 'ner_f1_': 0.8661800486618005}
```

### Run NER on slpit 1

```
CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_1/ --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file dev.jsonl --test_file test.jsonl --output_dir ./sciner-hyperpie_toast_1 --output_results
```

**Results**

```
08/09/2023 21:21:42 - INFO - __main__ -   ***** Running evaluation  *****
08/09/2023 21:21:42 - INFO - __main__ -     Num examples = 263
08/09/2023 21:21:42 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|██████████████████████████████████████████████| 17/17 [00:05<00:00,  3.37it/s]
08/09/2023 21:21:47 - INFO - __main__ -     Evaluation done in total 5.051516 secs (52.063574 example per second)
08/09/2023 21:21:47 - INFO - __main__ -   Result: {"f1": 0.8828828828828829, "f1_overlap": 0.8805309734513275, "precision": 0.8868778280542986, "recall": 0.8789237668161435}
08/09/2023 21:21:47 - INFO - __main__ -   Result: {"dev_best_f1": 0.851581508515815, "f1_": 0.8828828828828829, "f1_overlap_": 0.8805309734513275, "precision_": 0.8868778280542986, "recall_": 0.8789237668161435}
```

### Copy over intermediate results

`cp sciner-hyperpie_toast_1/ent_pred_* hyperpie/fold_1/`

### Run RE

```
CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_1 --learning_rate 2e-5  --num_train_epochs 10 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16 --train_file train.jsonl --test_file ent_pred_test.json --dev_file ent_pred_dev.json --use_ner_results --output_dir ./scire-hyperpie_toast_1
```

```
08/10/2023 08:06:51 - INFO - __main__ -   ***** Running evaluation  *****
08/10/2023 08:06:51 - INFO - __main__ -     Batch size = 16
08/10/2023 08:06:51 - INFO - __main__ -     Num examples = 218
Evaluating: 100%|██████████████████████████████████████████████| 14/14 [00:00<00:00, 16.03it/s]
08/10/2023 08:06:52 - INFO - __main__ -     Evaluation done in total 0.876779 secs (241.794168 example per second)
08/10/2023 08:06:52 - INFO - __main__ -   Result: {"f1": 0.0, "f1_with_ner": 0.0, "ner_f1": 0.8899082568807339}
{'dev_best_f1': 0.0, 'f1_': 0.0, 'f1_with_ner_': 0.0, 'ner_f1_': 0.8899082568807339}
```

### Run NER on slpit 2

```
CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_2/ --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file dev.jsonl --test_file test.jsonl --output_dir ./sciner-hyperpie_toast_2 --output_results
```

**Results**

```
08/10/2023 09:42:19 - INFO - __main__ -   ***** Running evaluation  *****
08/10/2023 09:42:19 - INFO - __main__ -     Num examples = 240
08/10/2023 09:42:19 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|██████████████████████████████████████████████| 15/15 [00:04<00:00,  3.25it/s]
08/10/2023 09:42:24 - INFO - __main__ -     Evaluation done in total 4.632923 secs (51.803146 example per second)
08/10/2023 09:42:24 - INFO - __main__ -   Result: {"f1": 0.8606811145510835, "f1_overlap": 0.8536585365853657, "precision": 0.8424242424242424, "recall": 0.879746835443038}
08/10/2023 09:42:24 - INFO - __main__ -   Result: {"dev_best_f1": 0.8672566371681415, "f1_": 0.8606811145510835, "f1_overlap_": 0.8536585365853657, "precision_": 0.8424242424242424, "recall_": 0.879746835443038}
```

### Copy over intermediate results

`cp sciner-hyperpie_toast_2/ent_pred_* hyperpie/fold_2/`

### Run RE

```
CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_2 --learning_rate 2e-5  --num_train_epochs 10 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16 --train_file train.jsonl --test_file ent_pred_test.json --dev_file ent_pred_dev.json --use_ner_results --output_dir ./scire-hyperpie_toast_2
```

```
... division by zero error b/c recall=0
```


## Test LLM extra training data

```
$ CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_0_wLLMdist/ --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file dev.jsonl --test_file test.jsonl --output_dir ./sciner-hyperpie_toast_0_dist --output_results
```

```
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:699: indexSelectLargeIndex: block: [588,0,0], thread: [127,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
Iteration:   4%|█▌                                           | 43/1216 [00:21<09:52,  1.98it/s]
Epoch:   0%|                                                            | 0/50 [00:21<?, ?it/s]
Traceback (most recent call last):
  File "run_acener.py", line 1085, in <module>
    main()
  File "run_acener.py", line 1018, in main
    global_step, tr_loss, best_f1 = train(args, model, tokenizer)
  File "run_acener.py", line 556, in train
    outputs = model(**inputs)
  File "/home/ws/ys8950/dev/PL-Marker/venv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/ws/ys8950/dev/PL-Marker/transformers/src/transformers/modeling_bert.py", line 3244, in forward
    outputs = self.bert(
  File "/home/ws/ys8950/dev/PL-Marker/venv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/ws/ys8950/dev/PL-Marker/transformers/src/transformers/modeling_bert.py", line 794, in forward
    embedding_output = self.embeddings(
  File "/home/ws/ys8950/dev/PL-Marker/venv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/ws/ys8950/dev/PL-Marker/transformers/src/transformers/modeling_bert.py", line 175, in forward
    position_embeddings = self.position_embeddings(position_ids)
  File "/home/ws/ys8950/dev/PL-Marker/venv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/ws/ys8950/dev/PL-Marker/venv/lib/python3.8/site-packages/torch/nn/modules/sparse.py", line 158, in forward
    return F.embedding(
  File "/home/ws/ys8950/dev/PL-Marker/venv/lib/python3.8/site-packages/torch/nn/functional.py", line 2044, in embedding
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
RuntimeError: CUDA error: device-side assert triggered
```

debugging:

* made sure only valid entity types are used ✔
* capping length of sentences to 200 words → fixed issue

**Results**

```
08/11/2023 00:22:12 - INFO - __main__ -   ***** Running evaluation  *****
08/11/2023 00:22:12 - INFO - __main__ -     Num examples = 270
08/11/2023 00:22:12 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|██████████████████████████████████████████████| 17/17 [00:05<00:00,  3.14it/s]
08/11/2023 00:22:17 - INFO - __main__ -     Evaluation done in total 5.432165 secs (49.703941 example per second)
08/11/2023 00:22:17 - INFO - __main__ -   Result: {"f1": 0.8238213399503722, "f1_overlap": 0.8275862068965516, "precision": 0.8383838383838383, "recall": 0.8097560975609757}
08/11/2023 00:22:17 - INFO - __main__ -   Result: {"dev_best_f1": 0.7066246056782334, "f1_": 0.8238213399503722, "f1_overlap_": 0.8275862068965516, "precision_": 0.8383838383838383, "recall_": 0.8097560975609757}
```

**Comparison to no dist. sup.**

* w/o: `Result: {"f1": 0.8640776699029126, "f1_overlap": 0.8687350835322196, "precision": 0.8599033816425121, "recall": 0.8682926829268293}`
* w/: `Result: {"f1": 0.8238213399503722, "f1_overlap": 0.8275862068965516, "precision": 0.8383838383838383, "recall": 0.8097560975609757}`

* w/o: `Result: {"dev_best_f1": 0.7849829351535836, "f1_": 0.8640776699029126, "f1_overlap_": 0.8687350835322196, "precision_": 0.8599033816425121, "recall_": 0.8682926829268293}`
* w/: `Result: {"dev_best_f1": 0.7066246056782334, "f1_": 0.8238213399503722, "f1_overlap_": 0.8275862068965516, "precision_": 0.8383838383838383, "recall_": 0.8097560975609757}`

→ splits done by shuffling all samples → w/o dist sup. benefit from same paper data, w/ dist sup. introduce out of sample data
→ do stratified sampling train/dev/test split wrt paper ID

## Stratified data splits

### Fold 0

#### NER

```
CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_0/ --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file dev.jsonl --test_file test.jsonl --output_dir ./sciner-hyperpie_fold_0 --output_results
```

```
08/11/2023 11:13:02 - INFO - __main__ -   ***** Running evaluation  *****
08/11/2023 11:13:02 - INFO - __main__ -     Num examples = 258
08/11/2023 11:13:02 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|██████████████████████████████████████████████| 17/17 [00:04<00:00,  3.44it/s]
08/11/2023 11:13:07 - INFO - __main__ -     Evaluation done in total 4.942271 secs (52.202719 example per second)
08/11/2023 11:13:07 - INFO - __main__ -   Result: {"f1": 0.702127659574468, "f1_overlap": 0.6984126984126985, "precision": 0.7951807228915663, "recall": 0.6285714285714286}
08/11/2023 11:13:07 - INFO - __main__ -   Result: {"dev_best_f1": 0.7180277349768875, "f1_": 0.702127659574468, "f1_overlap_": 0.6984126984126985, "precision_": 0.7951807228915663, "recall_": 0.6285714285714286}
```

#### NER w/ dist

```
CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_0_wLLMdist/ --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file dev.jsonl --test_file test.jsonl --output_dir ./sciner-hyperpie_fold_0_wLLMdist --output_results
```

```
08/11/2023 17:49:28 - INFO - __main__ -   ***** Running evaluation  *****
08/11/2023 17:49:28 - INFO - __main__ -     Num examples = 258
08/11/2023 17:49:28 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|██████████████████████████████████████████████| 17/17 [00:05<00:00,  3.37it/s]
08/11/2023 17:49:33 - INFO - __main__ -     Evaluation done in total 5.055305 secs (51.035495 example per second)
08/11/2023 17:49:33 - INFO - __main__ -   Result: {"f1": 0.6035502958579881, "f1_overlap": 0.6017699115044247, "precision": 0.796875, "recall": 0.4857142857142857}
08/11/2023 17:49:33 - INFO - __main__ -   Result: {"dev_best_f1": 0.5960264900662252, "f1_": 0.6035502958579881, "f1_overlap_": 0.6017699115044247, "precision_": 0.796875, "recall_": 0.4857142857142857}
```

#### Copy over intermediate results

`cp sciner-hyperpie_fold_0/ent_pred_* hyperpie/fold_0/`

#### RE

```
CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_0 --learning_rate 2e-5  --num_train_epochs 10 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16 --train_file train.jsonl --test_file ent_pred_test.json --dev_file ent_pred_dev.json --use_ner_results --output_dir ./scire-hyperpie_fold_0
```

```
08/15/2023 08:36:05 - INFO - __main__ -   ***** Running evaluation  *****
08/15/2023 08:36:05 - INFO - __main__ -     Batch size = 16
08/15/2023 08:36:05 - INFO - __main__ -     Num examples = 165
Evaluating: 100%|██████████████████████████████████████████████| 11/11 [00:00<00:00, 14.60it/s]
08/15/2023 08:36:06 - INFO - __main__ -     Evaluation done in total 0.757086 secs (265.491726 example per second)
08/15/2023 08:36:06 - INFO - __main__ -   Result: {"f1": 0.0, "f1_with_ner": 0.0, "ner_f1": 0.7040000000000001}
{'dev_best_f1': 0.10526315789473684, 'f1_': 0.0, 'f1_with_ner_': 0.0, 'ner_f1_': 0.7040000000000001}
```

**try to fix**
* play w/ `run_re.py` params
    * `--no_sym`: no change
    * `--no_sym --eval_unidirect`: dev\_best\_f1 0.105 → 0.111, otherweise no change
    * `--use_typemarker`: f1 0.0 → 0.16 (full result below)

### RE w/ use\_typemarker

```
CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_0 --learning_rate 2e-5  --num_train_epochs 10 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16 --train_file train.jsonl --test_file ent_pred_test.json --dev_file ent_pred_dev.json --use_ner_results --output_dir ./scire-hyperpie_fold_0_use_typemarker --use_typemarker
```

```
08/15/2023 16:17:58 - INFO - __main__ -   ***** Running evaluation  *****
08/15/2023 16:17:58 - INFO - __main__ -     Batch size = 16
08/15/2023 16:17:58 - INFO - __main__ -     Num examples = 165
Evaluating: 100%|██████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 15.75it/s]
08/15/2023 16:17:59 - INFO - __main__ -     Evaluation done in total 0.702042 secs (286.307834 example per second)
08/15/2023 16:17:59 - INFO - __main__ -   Result: {"f1": 0.16, "prec": 1.0, "rec": 0.08695652173913043, "f1_with_ner": 0.16, "prec_w_ner": 1.0, "rec_w_ner": 0.08695652173913043, "ner_f1": 0.7040000000000001}
{'dev_best_f1': 0.10526315789473684, 'f1_': 0.16, 'prec_': 1.0, 'rec_': 0.08695652173913043, 'f1_with_ner_': 0.16, 'prec_w_ner_': 1.0, 'rec_w_ner_': 0.08695652173913043, 'ner_f1_': 0.7040000000000001}
```

### Fold 1

#### NER

```
CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_1/ --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file dev.jsonl --test_file test.jsonl --output_dir ./sciner-hyperpie_fold_1 --output_results
```

```
08/15/2023 11:46:40 - INFO - __main__ -   ***** Running evaluation  *****
08/15/2023 11:46:40 - INFO - __main__ -     Num examples = 188
08/15/2023 11:46:40 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|██████████████████████████████████████████████████████████| 12/12 [00:03<00:00,  3.15it/s]
08/15/2023 11:46:44 - INFO - __main__ -     Evaluation done in total 3.822012 secs (49.188746 example per second)
08/15/2023 11:46:44 - INFO - __main__ -   Result: {"f1": 0.621160409556314, "f1_overlap": 0.5967213114754097, "precision": 0.6026490066225165, "recall": 0.6408450704225352}
08/15/2023 11:46:44 - INFO - __main__ -   Result: {"dev_best_f1": 0.7268170426065163, "f1_": 0.621160409556314, "f1_overlap_": 0.5967213114754097, "precision_": 0.6026490066225165, "recall_": 0.6408450704225352}
```

#### RE

```
CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_1 --learning_rate 2e-5  --num_train_epochs 10 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16 --train_file train.jsonl --test_file ent_pred_test.json --dev_file ent_pred_dev.json --use_ner_results --output_dir ./scire-hyperpie_fold_1_use_typemarker --use_typemarker
```

```
08/15/2023 16:25:31 - INFO - __main__ -   ***** Running evaluation  *****
08/15/2023 16:25:31 - INFO - __main__ -     Batch size = 16
08/15/2023 16:25:31 - INFO - __main__ -     Num examples = 151
Evaluating: 100%|██████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 15.08it/s]
08/15/2023 16:25:32 - INFO - __main__ -     Evaluation done in total 0.667094 secs (244.343362 example per second)
08/15/2023 16:25:32 - INFO - __main__ -   Result: {"f1": 0.22222222222222224, "prec": 0.5, "rec": 0.14285714285714285, "f1_with_ner": 0.22222222222222224, "prec_w_ner": 0.5, "rec_w_ner": 0.14285714285714285, "ner_f1": 0.621160409556314}
{'dev_best_f1': 0.0, 'f1_': 0.22222222222222224, 'prec_': 0.5, 'rec_': 0.14285714285714285, 'f1_with_ner_': 0.22222222222222224, 'prec_w_ner_': 0.5, 'rec_w_ner_': 0.14285714285714285, 'ner_f1_': 0.621160409556314}
```

### Fold 2

#### NER

```
CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_2/ --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file dev.jsonl --test_file test.jsonl --output_dir ./sciner-hyperpie_fold_2 --output_results
```

```
08/15/2023 14:28:06 - INFO - __main__ -   ***** Running evaluation  *****
08/15/2023 14:28:06 - INFO - __main__ -     Num examples = 135
08/15/2023 14:28:06 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|████████████████████████████████████████████████████████████| 9/9 [00:02<00:00,  3.25it/s]
08/15/2023 14:28:09 - INFO - __main__ -     Evaluation done in total 2.782101 secs (48.524478 example per second)
08/15/2023 14:28:09 - INFO - __main__ -   Result: {"f1": 0.7414634146341463, "f1_overlap": 0.7535545023696683, "precision": 0.6877828054298643, "recall": 0.8042328042328042}
08/15/2023 14:28:09 - INFO - __main__ -   Result: {"dev_best_f1": 0.6811594202898551, "f1_": 0.7414634146341463, "f1_overlap_": 0.7535545023696683, "precision_": 0.6877828054298643, "recall_": 0.8042328042328042
```


#### RE

```
CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_2 --learning_rate 2e-5  --num_train_epochs 10 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16 --train_file train.jsonl --test_file ent_pred_test.json --dev_file ent_pred_dev.json --use_ner_results --output_dir ./scire-hyperpie_fold_2_use_typemarker --use_typemarker
```

```
08/15/2023 16:31:46 - INFO - __main__ -   ***** Running evaluation  *****
08/15/2023 16:31:46 - INFO - __main__ -     Batch size = 16
08/15/2023 16:31:46 - INFO - __main__ -     Num examples = 221
Evaluating: 100%|██████████████████████████████████████████████████████████| 14/14 [00:00<00:00, 16.34it/s]
08/15/2023 16:31:47 - INFO - __main__ -     Evaluation done in total 0.864295 secs (146.940549 example per second)
08/15/2023 16:31:47 - INFO - __main__ -   Result: {"f1": 0.0, "prec": 0.0, "rec": 0, "f1_with_ner": 0.0, "prec_w_ner": 0.0, "rec_w_ner": 0, "ner_f1": 0.7414634146341463}
{'dev_best_f1': 0.22222222222222224, 'f1_': 0.0, 'prec_': 0.0, 'rec_': 0, 'f1_with_ner_': 0.0, 'prec_w_ner_': 0.0, 'rec_w_ner_': 0, 'ner_f1_': 0.7414634146341463}
```

### Fold 3

#### NER

```
CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_3/ --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file dev.jsonl --test_file test.jsonl --output_dir ./sciner-hyperpie_fold_3 --output_results
```

```
08/15/2023 16:11:02 - INFO - __main__ -   ***** Running evaluation  *****
08/15/2023 16:11:02 - INFO - __main__ -     Num examples = 201
08/15/2023 16:11:02 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|██████████████████████████████████████████████████████████| 13/13 [00:03<00:00,  3.25it/s]
08/15/2023 16:11:06 - INFO - __main__ -     Evaluation done in total 4.014124 secs (50.073192 example per second)
08/15/2023 16:11:06 - INFO - __main__ -   Result: {"f1": 0.6754385964912282, "f1_overlap": 0.6838709677419355, "precision": 0.6637931034482759, "recall": 0.6875}
08/15/2023 16:11:06 - INFO - __main__ -   Result: {"dev_best_f1": 0.7726161369193154, "f1_": 0.6754385964912282, "f1_overlap_": 0.6838709677419355, "precision_": 0.6637931034482759, "recall_": 0.6875}
```

#### RE

```
CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_3 --learning_rate 2e-5  --num_train_epochs 10 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16 --train_file train.jsonl --test_file ent_pred_test.json --dev_file ent_pred_dev.json --use_ner_results --output_dir ./scire-hyperpie_fold_3_use_typemarker --use_typemarker
```

```
08/15/2023 16:44:57 - INFO - __main__ -   ***** Running evaluation  *****
08/15/2023 16:44:57 - INFO - __main__ -     Batch size = 16
08/15/2023 16:44:57 - INFO - __main__ -     Num examples = 231
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 16.95it/s]
08/15/2023 16:44:58 - INFO - __main__ -     Evaluation done in total 0.893054 secs (154.525942 example per second)
08/15/2023 16:44:58 - INFO - __main__ -   Result: {"f1": 0.1111111111111111, "prec": 0.3333333333333333, "rec": 0.06666666666666667, "f1_with_ner": 0.1111111111111111, "prec_w_ner": 0.3333333333333333, "rec_w_ner": 0.06666666666666667, "ner_f1": 0.6754385964912282}
{'dev_best_f1': 0.0, 'f1_': 0.1111111111111111, 'prec_': 0.3333333333333333, 'rec_': 0.06666666666666667, 'f1_with_ner_': 0.1111111111111111, 'prec_w_ner_': 0.3333333333333333, 'rec_w_ner_': 0.06666666666666667, 'ner_f1_': 0.6754385964912282}
```

### Fold 4

#### NER

```
CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_4/ --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file dev.jsonl --test_file test.jsonl --output_dir ./sciner-hyperpie_fold_4 --output_results
```

```
08/15/2023 18:52:15 - INFO - __main__ -   ***** Running evaluation  *****
08/15/2023 18:52:15 - INFO - __main__ -     Num examples = 254
08/15/2023 18:52:15 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|██████████████████████████████████████████████| 16/16 [00:04<00:00,  3.40it/s]
08/15/2023 18:52:19 - INFO - __main__ -     Evaluation done in total 4.725872 secs (53.746699 example per second)
08/15/2023 18:52:19 - INFO - __main__ -   Result: {"f1": 0.7733333333333333, "f1_overlap": 0.7733333333333333, "precision": 0.6666666666666666, "recall": 0.9206349206349206}
08/15/2023 18:52:19 - INFO - __main__ -   Result: {"dev_best_f1": 0.6837209302325582, "f1_": 0.7733333333333333, "f1_overlap_": 0.7733333333333333, "precision_": 0.6666666666666666, "recall_": 0.9206349206349206
```

#### RE

```
CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_4 --learning_rate 2e-5  --num_train_epochs 10 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16 --train_file train.jsonl --test_file ent_pred_test.json --dev_file ent_pred_dev.json --use_ner_results --output_dir ./scire-hyperpie_fold_4_use_typemarker --use_typemarker
```

```
08/15/2023 19:00:14 - INFO - __main__ -   ***** Running evaluation  *****
08/15/2023 19:00:14 - INFO - __main__ -     Batch size = 16
08/15/2023 19:00:14 - INFO - __main__ -     Num examples = 87
Evaluating: 100%|████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.47it/s]
08/15/2023 19:00:15 - INFO - __main__ -     Evaluation done in total 0.482832 secs (445.289294
example per second)
08/15/2023 19:00:15 - INFO - __main__ -   Result: {"f1": 0.0, "prec": 0, "rec": 0.0, "f1_with_ner": 0.0, "prec_w_ner": 0, "rec_w_ner": 0.0, "ner_f1": 0.7733333333333333}
{'dev_best_f1': 0.09230769230769231, 'f1_': 0.0, 'prec_': 0, 'rec_': 0.0, 'f1_with_ner_': 0.0,
'prec_w_ner_': 0, 'rec_w_ner_': 0.0, 'ner_f1_': 0.7733333333333333}
```

### Fold 5

#### NER

```
CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_5/ --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file dev.jsonl --test_file test.jsonl --output_dir ./sciner-hyperpie_fold_5 --output_results
```

```
08/15/2023 21:34:41 - INFO - __main__ -   ***** Running evaluation  *****
08/15/2023 21:34:41 - INFO - __main__ -     Num examples = 277
08/15/2023 21:34:41 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|██████████████████████████████████████████████| 18/18 [00:05<00:00,  3.45it/s]
08/15/2023 21:34:46 - INFO - __main__ -     Evaluation done in total 5.233445 secs (52.928812 example per second)
08/15/2023 21:34:46 - INFO - __main__ -   Result: {"f1": 0.6552380952380954, "f1_overlap": 0.655367231638418, "precision": 0.7644444444444445, "recall": 0.5733333333333334}
08/15/2023 21:34:46 - INFO - __main__ -   Result: {"dev_best_f1": 0.7785234899328859, "f1_": 0.6552380952380954, "f1_overlap_": 0.655367231638418, "precision_": 0.7644444444444445, "recall_": 0.5733333333333334}
```

#### RE

```
CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_5 --learning_rate 2e-5  --num_train_epochs 10 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16 --train_file train.jsonl --test_file ent_pred_test.json --dev_file ent_pred_dev.json --use_ner_results --output_dir ./scire-hyperpie_fold_5_use_typemarker --use_typemarker
```

```
08/16/2023 07:11:21 - INFO - __main__ -   ***** Running evaluation  *****
08/16/2023 07:11:21 - INFO - __main__ -     Batch size = 16
08/16/2023 07:11:21 - INFO - __main__ -     Num examples = 225
Evaluating: 100%|██████████████████████████████████████████████| 15/15 [00:00<00:00, 19.55it/s]
08/16/2023 07:11:22 - INFO - __main__ -     Evaluation done in total 0.771252 secs (313.775600 example per second)
08/16/2023 07:11:22 - INFO - __main__ -   Result: {"f1": 0.11764705882352941, "prec": 0.75, "rec": 0.06382978723404255, "f1_with_ner": 0.11764705882352941, "prec_w_ner": 0.75, "rec_w_ner": 0.06382978723404255, "ner_f1": 0.6577437858508605}
{'dev_best_f1': 0.0, 'f1_': 0.11764705882352941, 'prec_': 0.75, 'rec_': 0.06382978723404255, 'f1_with_ner_': 0.11764705882352941, 'prec_w_ner_': 0.75, 'rec_w_ner_': 0.06382978723404255, 'ner_f1_': 0.6577437858508605
```

### Fold 6

#### NER

```
CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_6/ --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file dev.jsonl --test_file test.jsonl --output_dir ./sciner-hyperpie_fold_6 --output_results
```

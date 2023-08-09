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

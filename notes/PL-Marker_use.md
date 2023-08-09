# PoC test runs

### Run NER

```
CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_0/ --learning_rate 2e-10 --num_train_epochs 15 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file dev.jsonl --test_file test.jsonl --output_dir ./sciner-hyperpie_toast --output_results
```

**Results**

```
08/09/2023 16:02:24 - INFO - __main__ -   ***** Running evaluation  *****
08/09/2023 16:02:24 - INFO - __main__ -     Num examples = 270
08/09/2023 16:02:24 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|██████████████████████████████████████████████| 17/17 [00:05<00:00,  3.07it/s]
08/09/2023 16:02:30 - INFO - __main__ -     Evaluation done in total 5.628079 secs (47.973744 example per second)
08/09/2023 16:02:30 - INFO - __main__ -   Result: {"f1": 0.0066919473566807944, "f1_overlap": 0.0024151032309368834, "precision": 0.0035063113604488078, "recall": 0.07317073170731707}
08/09/2023 16:02:30 - INFO - __main__ -   Result: {"dev_best_f1": 0.004675324675324675, "f1_": 0.0066919473566807944, "f1_overlap_": 0.0024151032309368834, "precision_": 0.0035063113604488078, "recall_": 0.07317073170731707}
```

### Copy over intermediate results

`cp sciner-hyperpie_toast/ent_pred_* hyperpie/fold_0/`

### Run RE

```
CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie/fold_0 --learning_rate 2e-15  --num_train_epochs 15 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16 --train_file train.jsonl --test_file ent_pred_test.json --dev_file ent_pred_dev.json --use_ner_results --output_dir ./scire-hyperpie_toast
```

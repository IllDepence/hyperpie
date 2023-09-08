NER+RE eval on 5-fold cross validation w/ folds as stratified samples based on number of relations (= class balance @RE step)

## Numbers

```
NER precision: 81.5 ± 2.9
NER recall: 76.8 ± 2.2
NER f1: 79.0 ± 1.6
RE precision: 33.5 ± 19.3
RE recall: 5.9 ± 3.7
RE f1: 9.9 ± 6.0
```


## Eval

### fold 0

#### NER

```
CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie_5fold_strat/fold_0/ --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file dev.jsonl --test_file test.jsonl --output_dir ./hp5fold_sciner-fold_0 --output_results
```

```
09/05/2023 09:12:32 - INFO - __main__ -   ***** Running evaluation  *****
09/05/2023 09:12:32 - INFO - __main__ -     Num examples = 505
09/05/2023 09:12:32 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|██████████████████████████████████████████████| 32/32 [00:09<00:00,  3.52it/s]
09/05/2023 09:12:41 - INFO - __main__ -     Evaluation done in total 9.103152 secs (55.475290 example per second)
09/05/2023 09:12:41 - INFO - __main__ -   Result: {"f1": 0.7908163265306123, "f1_overlap": 0.7868852459016392, "precision": 0.8469945355191257, "recall": 0.7416267942583732}
09/05/2023 09:12:41 - INFO - __main__ -   Result: {"dev_best_f1": 0.7988826815642458, "f1_": 0.7908163265306123, "f1_overlap_": 0.7868852459016392, "precision_": 0.8469945355191257, "recall_": 0.7416267942583732
```

#### copy NER results

```
cp hp5fold_sciner-fold_0/ent_pred_* hyperpie_5fold_strat/fold_0/
```

#### RE

```
CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie_5fold_strat/fold_0/ --learning_rate 2e-5  --num_train_epochs 10 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16 --train_file train.jsonl --test_file ent_pred_test.json --dev_file ent_pred_dev.json --use_ner_results --output_dir ./hp5fold_scire-hyperpie_fold_0 --use_typemarker
```

```
09/05/2023 10:01:37 - INFO - __main__ -   ***** Running evaluation  *****
09/05/2023 10:01:37 - INFO - __main__ -     Batch size = 16
09/05/2023 10:01:37 - INFO - __main__ -     Num examples = 362
Evaluating: 100%|██████████████████████████████████████████████| 23/23 [00:01<00:00, 18.77it/s]
09/05/2023 10:01:38 - INFO - __main__ -     Evaluation done in total 1.233794 secs (346.087056 example per second)
09/05/2023 10:01:38 - INFO - __main__ -   Result: {"f1": 0.0, "prec": 0, "rec": 0.0, "f1_with_ner": 0.0, "prec_w_ner": 0, "rec_w_ner": 0.0, "ner_f1": 0.793854033290653}
{'dev_best_f1': 0.05, 'f1_': 0.0, 'prec_': 0, 'rec_': 0.0, 'f1_with_ner_': 0.0, 'prec_w_ner_': 0, 'rec_w_ner_': 0.0, 'ner_f1_': 0.793854033290653}
```

### fold 1

#### NER

```
CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie_5fold_strat/fold_1/ --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file dev.jsonl --test_file test.jsonl --output_dir ./hp5fold_sciner-fold_1 --output_results
```

```
09/05/2023 11:01:55 - INFO - __main__ -   ***** Running evaluation  *****
09/05/2023 11:01:55 - INFO - __main__ -     Num examples = 444
09/05/2023 11:01:55 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|██████████████████████████████████████████████| 28/28 [00:07<00:00,  3.54it/s]
09/05/2023 11:02:03 - INFO - __main__ -     Evaluation done in total 7.919330 secs (56.065348 example per second)
09/05/2023 11:02:03 - INFO - __main__ -   Result: {"f1": 0.7745504840940526, "f1_overlap": 0.7747252747252747, "precision": 0.7734806629834254, "recall": 0.775623268698061}
09/05/2023 11:02:03 - INFO - __main__ -   Result: {"dev_best_f1": 0.8258706467661692, "f1_": 0.7745504840940526, "f1_overlap_": 0.7747252747252747, "precision_": 0.7734806629834254, "recall_": 0.775623268698061}
```

#### copy NER results

```
$ cp hp5fold_sciner-fold_1/ent_pred_* hyperpie_5fold_strat/fold_1
```

#### RE

```
$ CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie_5fold_strat/fold_1/ --learning_rate 2e-5  --num_train_epochs 10 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16 --train_file train.jsonl --test_file ent_pred_test.json --dev_file ent_pred_dev.json --use_ner_results --output_dir ./hp5fold_scire-hyperpie_fold_1 --use_typemarker
```

```
09/05/2023 11:43:07 - INFO - __main__ -   ***** Running evaluation  *****
09/05/2023 11:43:07 - INFO - __main__ -     Batch size = 16
09/05/2023 11:43:07 - INFO - __main__ -     Num examples = 362
Evaluating: 100%|██████████████████████████████████████████████| 23/23 [00:01<00:00, 19.16it/s]
09/05/2023 11:43:08 - INFO - __main__ -     Evaluation done in total 1.209518 secs (302.599823 example per second)
09/05/2023 11:43:08 - INFO - __main__ -   Result: {"f1": 0.10810810810810811, "prec": 0.4, "rec": 0.0625, "f1_with_ner": 0.10810810810810811, "prec_w_ner": 0.4, "rec_w_ner": 0.0625, "ner_f1": 0.7799442896935934}
{'dev_best_f1': 0.24390243902439027, 'f1_': 0.10810810810810811, 'prec_': 0.4, 'rec_': 0.0625, 'f1_with_ner_': 0.10810810810810811, 'prec_w_ner_': 0.4, 'rec_w_ner_': 0.0625, 'ner_f1_': 0.7799442896935934}
```

### fold 2

#### NER

```
CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie_5fold_strat/fold_2/ --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file dev.jsonl --test_file test.jsonl --output_dir ./hp5fold_sciner-fold_2 --output_results
```

```
09/05/2023 12:45:54 - INFO - __main__ -   ***** Running evaluation  *****
09/05/2023 12:45:54 - INFO - __main__ -     Num examples = 551
09/05/2023 12:45:54 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|██████████████████████████████████████████████| 35/35 [00:09<00:00,  3.62it/s]
09/05/2023 12:46:04 - INFO - __main__ -     Evaluation done in total 9.670518 secs (56.977297 example per second)
09/05/2023 12:46:04 - INFO - __main__ -   Result: {"f1": 0.8029197080291971, "f1_overlap": 0.8029020556227326, "precision": 0.8333333333333334, "recall": 0.7746478873239436}
09/05/2023 12:46:04 - INFO - __main__ -   Result: {"dev_best_f1": 0.7715877437325905, "f1_": 0.8029197080291971, "f1_overlap_": 0.8029020556227326, "precision_": 0.8333333333333334, "recall_": 0.7746478873239436}
```

#### copy NER results

```
$ cp hp5fold_sciner-fold_2/ent_pred_* hyperpie_5fold_strat/fold_2
```

#### RE

```
$ CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie_5fold_strat/fold_2/ --learning_rate 2e-5  --num_train_epochs 10 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16 --train_file train.jsonl --test_file ent_pred_test.json --dev_file ent_pred_dev.json --use_ner_results --output_dir ./hp5fold_scire-hyperpie_fold_2 --use_typemarker
```

```
09/05/2023 13:57:58 - INFO - __main__ -   ***** Running evaluation  *****
09/05/2023 13:57:58 - INFO - __main__ -     Batch size = 16
09/05/2023 13:57:58 - INFO - __main__ -     Num examples = 396
Evaluating: 100%|██████████████████████████████████████████████| 25/25 [00:01<00:00, 19.42it/s]
09/05/2023 13:57:59 - INFO - __main__ -     Evaluation done in total 1.296065 secs (346.433154 example per second)
09/05/2023 13:57:59 - INFO - __main__ -   Result: {"f1": 0.15, "prec": 0.375, "rec": 0.09375, "f1_with_ner": 0.15, "prec_w_ner": 0.375, "rec_w_ner": 0.09375, "ner_f1": 0.8048780487804879}
{'dev_best_f1': 0.15, 'f1_': 0.15, 'prec_': 0.375, 'rec_': 0.09375, 'f1_with_ner_': 0.15, 'prec_w_ner_': 0.375, 'rec_w_ner_': 0.09375, 'ner_f1_': 0.8048780487804879}
```

### fold 3

#### NER

```
$ CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie_5fold_strat/fold_3/ --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file dev.jsonl --test_file test.jsonl --output_dir ./hp5fold_sciner-fold_3 --output_results
```

```
09/05/2023 15:04:31 - INFO - __main__ -   ***** Running evaluation  *****
09/05/2023 15:04:31 - INFO - __main__ -     Num examples = 479
09/05/2023 15:04:31 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|██████████████████████████████████████████████| 30/30 [00:08<00:00,  3.50it/s]
09/05/2023 15:04:40 - INFO - __main__ -     Evaluation done in total 8.584703 secs (55.796922 example per second)
09/05/2023 15:04:40 - INFO - __main__ -   Result: {"f1": 0.7745358090185676, "f1_overlap": 0.7620286085825748, "precision": 0.8, "recall": 0.7506426735218509}
09/05/2023 15:04:40 - INFO - __main__ -   Result: {"dev_best_f1": 0.8284313725490196, "f1_": 0.7745358090185676, "f1_overlap_": 0.7620286085825748, "precision_": 0.8, "recall_": 0.7506426735218509}
```

#### copy NER results

```
$ cp hp5fold_sciner-fold_3/ent_pred_* hyperpie_5fold_strat/fold_3/
```

#### RE

```
$ CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie_5fold_strat/fold_3/ --learning_rate 2e-5  --num_train_epochs 10 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16 --train_file train.jsonl --test_file ent_pred_test.json --dev_file ent_pred_dev.json --use_ner_results --output_dir ./hp5fold_scire-hyperpie_fold_3 --use_typemarker
```

```
09/05/2023 15:15:48 - INFO - __main__ -   ***** Running evaluation  *****
09/05/2023 15:15:48 - INFO - __main__ -     Batch size = 16
09/05/2023 15:15:48 - INFO - __main__ -     Num examples = 365
Evaluating: 100%|██████████████████████████████████████████████| 23/23 [00:01<00:00, 19.06it/s]
09/05/2023 15:15:50 - INFO - __main__ -     Evaluation done in total 1.215844 secs (316.652580
example per second)
09/05/2023 15:15:50 - INFO - __main__ -   Result: {"f1": 0.14285714285714285, "prec": 0.4, "rec": 0.08695652173913043, "f1_with_ner": 0.14285714285714285, "prec_w_ner": 0.4, "rec_w_ner": 0.08695652173913043, "ner_f1": 0.7870619946091645}
{'dev_best_f1': 0.1111111111111111, 'f1_': 0.14285714285714285, 'prec_': 0.4, 'rec_': 0.08695652173913043, 'f1_with_ner_': 0.14285714285714285, 'prec_w_ner_': 0.4, 'rec_w_ner_': 0.08695652173913043, 'ner_f1_': 0.7870619946091645}
```


### fold 4

#### NER

```
$ CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie_5fold_strat/fold_4/ --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.jsonl --dev_file dev.jsonl --test_file test.jsonl --output_dir ./hp5fold_sciner-fold_4 --output_results
```

```
09/05/2023 16:32:51 - INFO - __main__ -   ***** Running evaluation  *****
09/05/2023 16:32:51 - INFO - __main__ -     Num examples = 521
09/05/2023 16:32:51 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|██████████████████████████████████████████████| 33/33 [00:09<00:00,  3.58it/s]
09/05/2023 16:33:00 - INFO - __main__ -     Evaluation done in total 9.218625 secs (56.516022 example per second)
09/05/2023 16:33:00 - INFO - __main__ -   Result: {"f1": 0.8086253369272237, "f1_overlap": 0.804289544235925, "precision": 0.819672131147541, "recall": 0.7978723404255319}
09/05/2023 16:33:00 - INFO - __main__ -   Result: {"dev_best_f1": 0.7724317295188556, "f1_": 0.8086253369272237, "f1_overlap_": 0.804289544235925, "precision_": 0.819672131147541, "recall_": 0.7978723404255319}
```

#### copy NER results

```
$ cp hp5fold_sciner-fold_4/ent_pred_* hyperpie_5fold_strat/fold_4/
```

#### RE

```
$ CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie_5fold_strat/fold_4/ --learning_rate 2e-5  --num_train_epochs 10 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16 --train_file train.jsonl --test_file ent_pred_test.json --dev_file ent_pred_dev.json --use_ner_results --output_dir ./hp5fold_scire-hyperpie_fold_4 --use_typemarker
```

```
09/05/2023 16:50:37 - INFO - __main__ -   ***** Running evaluation  *****
09/05/2023 16:50:37 - INFO - __main__ -     Batch size = 16
09/05/2023 16:50:37 - INFO - __main__ -     Num examples = 366
Evaluating: 100%|██████████████████████████████████████████████| 23/23 [00:01<00:00, 18.65it/s]
09/05/2023 16:50:39 - INFO - __main__ -     Evaluation done in total 1.243338 secs (349.060259 example per second)
09/05/2023 16:50:39 - INFO - __main__ -   Result: {"f1": 0.14285714285714285, "prec": 0.75, "rec": 0.07894736842105263, "f1_with_ner": 0.09523809523809525, "prec_w_ner": 0.5, "rec_w_ner": 0.05263157894736842, "ner_f1": 0.814111261872456}
{'dev_best_f1': 0.12, 'f1_': 0.14285714285714285, 'prec_': 0.75, 'rec_': 0.07894736842105263, 'f1_with_ner_': 0.09523809523809525, 'prec_w_ner_': 0.5, 'rec_w_ner_': 0.05263157894736842, 'ner_f1_': 0.814111261872456}
```

## Apply to unannotated data

* start w/ `transformed_pprs.jsonl`
    * take sample
    * `$ cat transformed_pprs.jsonl | shuf | head -n 15000 > transformed_pprs_15k_sample.jsonl`
* do dist sup (to get paras) ✔
    * `$ python3 distant_supervision.py` (w/ paths set accordinly)
* transform to PL-Marker format (also dues crutial cap at 200 tokens) ✔
    * `$ python3 hyperpie/data/convert_plmarker.py ../data/transformed_pprs_15k_sample_dist_annot.jsonl loose_matching` (Exact token matches: 4743951 (0.64), Mid token matches: 2703921 (0.36))
* apply model

```
$ CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --
model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file all_444.jsonl --dev_file all_444.jsonl --test_file transformed_pprs_15k_sample_dist_annot_plmarker.jsonl --output_dir ./sciner-hyperpie_predict_pprs_15_sample --output_results
```

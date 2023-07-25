# %%
import os
import json
import random
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
import wandb

# self defined
# models-->utils
from utils.base_model import Encoder
from utils.data import parse_file, Collate_Fn_Manager
from utils.util import set_seed

torch.cuda.empty_cache()
# check GPU
gpu_count = torch.cuda.device_count()
print("Number of GPU: ", gpu_count)
is_using_gpu = torch.cuda.is_available()
if is_using_gpu:  
    current_device = torch.cuda.current_device()
    print("using GPU is: ", current_device)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, 
                    help="random seed for data")
parser.add_argument("--num_epochs", type=int, default=5, 
                    help="number of epochs to train")
parser.add_argument("--batch_size", type=int, default=4, 
                    help="training batch size")
parser.add_argument("--warmup_epochs", type=int, default=1, 
                    help="warmup epochs")
parser.add_argument("--learning_rate", type=float, default=7e-5, 
                    help="learning rate")
parser.add_argument("--project", type=str, default="RE1st", 
                    help="project name for wandb")
# parser.add_argument("--model", type=str, default="base", 
#                     help="model")
parser.add_argument("--language_model", type=str, default="allenai/scibert_scivocab_cased", 
                    help="language model")

parser.add_argument("--candidate_downsampling", type=int, default=1000, 
                    help="number of candidate spans to use during training (-1 for no downsampling)")

parser.add_argument("--negative_prob", type=float, default=1.0, 
                    help="probability of showing negative relation examples")

parser.add_argument("--k_mentions", type=int, default=50, 
                    help="number of mention spans to perform relation extraction on")

parser.add_argument("--k_mentions_test", type=int, default=400, 
                    help="number of mention spans to perform relation extraction on at test time")

parser.add_argument("--pooling", type=str, default="max", 
                    help="mention pooling method: max, mean")

args = parser.parse_args()

current_datetime = datetime.now()
folder_name = current_datetime.strftime("%Y%m%d_%H:%M:%S")

# set the wandb project where this run will be logged
# wandb.init(
#             project=args.project,
#             config = {
#                 'learning_rate': args.learning_rate,
#                 'epoch': args.num_epochs
#             }
# )
# wandb.config.update(args)
# wandb.config.identifier = folder_name

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
g = torch.Generator()
g.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------- data init --------------
tokenizer = AutoTokenizer.from_pretrained(args.language_model)

data_path = './data/'
train_files = os.listdir(data_path)
parsed_files = []
for file in train_files:
    if file.endswith(".json"):
        fname = os.path.join(data_path, file)
        output = parse_file(fname, tokenizer=tokenizer)
        parsed_files.extend(output)

train_loader = DataLoader(parsed_files, 
                          batch_size=args.batch_size, 
                          shuffle=True, 
                          collate_fn=Collate_Fn_Manager(max_span_len=15).collate_fn,
                          generator=g)


dev_data = './data/'
dev_files = os.listdir(dev_data)
parsed_files_dev = []
for file in dev_files:
    if file.endswith(".json"):
        fname = os.path.join(data_path, file)
        output = parse_file(fname, tokenizer=tokenizer)
        parsed_files_dev.extend(output)

dev_loader = DataLoader(parsed_files_dev, 
                        batch_size=args.batch_size, 
                        shuffle=False, 
                        collate_fn=Collate_Fn_Manager(max_span_len=15).collate_fn)

# # -------------- model loading --------------
# start by loading language model
lm_config = AutoConfig.from_pretrained(args.language_model,
                                       num_labels=10)
lm_model = AutoModel.from_pretrained(args.language_model,
                                    from_tf=False,
                                    config=lm_config,)
model = Encoder(config=lm_config,
                model=lm_model, 
                cls_token_id=tokenizer.convert_tokens_to_ids(tokenizer.cls_token), 
                sep_token_id=tokenizer.convert_tokens_to_ids(tokenizer.sep_token),
                negative_prob=args.negative_prob,
                k_mentions=args.k_mentions,
                pooling=args.pooling)

model.to(device)


print('Model load finised ...\n', model)
# -------------- optimizer things --------------
optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-6)
total_steps = len(train_loader) * args.num_epochs
warmup_steps = len(train_loader) * args.warmup_epochs
lr_scheduler = get_linear_schedule_with_warmup(optimizer, 
                                               num_warmup_steps=warmup_steps, 
                                               num_training_steps=total_steps
                                               )

scaler = GradScaler()
# %% --------------- TRAIN LOOP ------------------

best_dev_f1 = 0
step_global = 0

for ep in tqdm(range(args.num_epochs)):
    avg_rloss = []
    avg_mloss = []
    avg_loss = []
    model.true_positives = 0
    model.false_positives = 0
    model.false_negatives = 0
    model.true_negatives = 0
    
    model.train()
    with tqdm(train_loader) as progress_bar:
        for b in progress_bar:
            step_global += 1
            input_ids, attention_masks, candidate_spans, relation_labels, offset_mapping, ids, sources = b

            # reduce candidate spans via random sampling
            if args.candidate_downsampling != -1:
                candidate_spans = [random.sample(x, min(args.candidate_downsampling, len(x))) for x in candidate_spans]
            # add back spans for which we have labels
            for i, b_i in enumerate(relation_labels):
                for label in b_i:
                    h_ent = label['h']
                    t_ent = label['t']
                    if h_ent not in candidate_spans[i]:
                        candidate_spans[i].append(h_ent)
                    if t_ent not in candidate_spans[i]:
                        candidate_spans[i].append(t_ent)
            # print('relation_labels,,,,',relation_labels)
            # pass data to model
            with autocast():
                relation_loss, mention_loss, loss, output = model(input_ids.to(device), 
                                                                  attention_masks.to(device), 
                                                                  candidate_spans, 
                                                                  relation_labels)

            # accumulate loss
            avg_rloss.append(relation_loss.item())
            avg_mloss.append(mention_loss.item())
            avg_loss.append(loss.item())
            
            # backpropagate & reset
            scaler.scale(loss).backward()
            
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            # optimizer.step()
            scaler.update()
            lr_scheduler.step()
            model.zero_grad()
            del loss

            acc, p, r, f1 = 0, 0, 0, 0
            if model.true_positives != 0:
                p = model.true_positives / (model.true_positives + model.false_positives)
                r = model.true_positives / (model.true_positives + model.false_negatives)
                f1 = (2 * p * r) / (p + r)
            acc = (model.true_positives + model.true_negatives) / (model.true_positives + model.true_negatives + model.false_negatives + model.false_positives)
            # progress_bar.set_postfix({"L":f"{sum(avg_loss)/len(avg_loss):.2f}", "TP":f"{model.true_positives}", "A":f"{100*acc:.2f}", "P":f"{100*p:.2f}", "R":f"{100*r:.2f}", "F1":f"{100*f1:.2f}"})

    #         wandb.log({"loss": avg_loss[-1]}, step=step_global)
    #         wandb.log({"loss_mention": avg_mloss[-1]}, step=step_global)
    #         wandb.log({"loss_relation": avg_rloss[-1]}, step=step_global)

    # wandb.log({"precision_train": p}, step=step_global)
    # wandb.log({"recall_train": r}, step=step_global)
    # wandb.log({"f1_micro_train": f1}, step=step_global)
    # wandb.log({"accuracy_train": acc}, step=step_global)
    print('precision_train: ', p)
    print("recall_train: ", r)
    print("f1_micro_train: ", f1)
    print("accuracy_train: ", acc)   
    

    # --------------- EVAL ON DEV ------------------
    avg_loss = []
    model.true_positives = 0
    model.false_positives = 0
    model.false_negatives = 0
    model.true_negatives = 0
    model.eval()
    with tqdm(dev_loader) as progress_bar:
        for b in progress_bar:

            input_ids, attention_masks, candidate_spans, relation_labels, offset_mapping, ids, sources = b
            
            # pass data to model
            relation_loss, mention_loss, loss, output = model(input_ids.to(device), 
                                                              attention_masks.to(device), 
                                                              candidate_spans, 
                                                              relation_labels)

            # accumulate loss
            avg_loss.append(loss.item())
            del loss

            acc, p, r, f1 = 0, 0, 0, 0
            # print('model.true_positives:   ',model.true_positives)
            if model.true_positives != 0:
                p = model.true_positives/(model.true_positives+model.false_positives)
                r = model.true_positives/(model.true_positives+model.false_negatives)
                f1 = (2*p*r)/(p+r)
            acc = (model.true_positives + model.true_negatives)/(model.true_positives + model.true_negatives + model.false_negatives + model.false_positives)
            # progress_bar.set_postfix({"L":f"{sum(avg_loss)/len(avg_loss):.2f}", "TP":f"{model.true_positives}", "A":f"{100*acc:.2f}", "P":f"{100*p:.2f}", "R":f"{100*r:.2f}", "F1":f"{100*f1:.2f}"})

        # if f1 >= best_dev_f1:
        # best_dev_f1 = f1
        # wandb.log({"best_precision_dev": p}, step=step_global)
        # wandb.log({"best_recall_dev": r}, step=step_global)
        # wandb.log({"Best_f1_micro_dev": f1}, step=step_global)
        # wandb.log({"Best_accuracy_dev": acc}, step=step_global)
        print('precision_dev: ', p)
        print("recall_dev: ", r)
        print("f1_micro_dev: ", f1)
        print("accuracy_dev: ", acc)
        
        model_fname = os.path.join('./checkpoints', folder_name)
        torch.save(model.state_dict(), f'{model_fname}.pt')

        # wandb.log({"precision_dev": p}, step=step_global)
        # wandb.log({"recall_dev": r}, step=step_global)
        # wandb.log({"f1_micro_dev": f1}, step=step_global)
        # wandb.log({"accuracy_dev": acc}, step=step_global)
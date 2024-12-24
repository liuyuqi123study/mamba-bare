#import argparse
import os
import csv
os.environ["WANDB_API_KEY"]='60b1229d1854c67a97513e2ed49d0ece0d0c43ab'
from typing import List, Optional
from utils.generation import InferenceParams
import sys
sys.path.append('/content/drive/MyDrive/mamba-bare-main/mamba_core')
sys.path.append('/content/drive/MyDrive/mamba-bare-main/train/metrics')
sys.path.append('/content/drive/MyDrive/mamba-bare-main/train/callbacks')
sys.path.append('/content/drive/MyDrive/mamba-bare-main/train/benchmarks')
import mambapp
from pathlib import Path
from datasets import load_dataset
#from adversarial_filtering.bert.dataloader import setup_bert, InputExample, PaddingInputExample, \
 #   file_based_convert_examples_to_features, file_based_input_fn_builder, _truncate_seq_pair, gcs_agnostic_open, _save_np, _softmax
from metrics.perplexity import Perplexity
from metrics.num_tokens import NumTokens
import torch
import jsonlines
import hydra
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import Logger

from utils import utils
log = utils.get_logger(__name__)
from categories_MMLU import subcategories, categories
from omegaconf import OmegaConf, DictConfig
import argparse
import datasets

import numpy as np
import pandas as pd
import time
import re

from crop import crop
# Copyright (c) 2023, Tri Dao, Albert Gu.

import json
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM

from mambapp import MambaLMHeadModel


data_dir='/content/drive/MyDrive/data'
save_dir='results'
genlen=100
temperature=1.0
topk=1
topp=1.0
minp=0.0
repetition_penalty=True
batch=1
choices=['A','B','C','D']
ntrain=5
torch.random.manual_seed(0)


max_seq_length=128
'''The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded'''
do_train=False 
'''Whether to run training.'''
predict_val=False

predict_test=False
'''Whether to run the model in inference mode on the test set.'''
train_batch_size=16
predict_batch_size=16 
'''Total batch size for eval.'''
learning_rate=2e-5
'''The initial learning rate for Adam.'''
num_train_epochs=3,
'''Total number of training epochs to perform.'''


num_labels=4,



endingonly=False,



def preprocess_hellaswag(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs_hellaswag(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess_hellaswag(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess_hellaswag(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)

def remove_prefix(text: str, prefix: str):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text  # or whatever

def load_checkpoint(path, device='cpu'):
    path = Path(path).expanduser()
    if path.is_dir():
        path /= 'last.ckpt'
    # dst = f'cuda:{torch.cuda.current_device()}'
    log.info(f'Loading checkpoint from {str(path)}')
    state_dict = torch.load(path, map_location=device)
    # T2T-ViT checkpoint is nested in the key 'state_dict_ema'
    if state_dict.keys() == {'state_dict_ema'}:
        state_dict = state_dict['state_dict_ema']
    # Swin checkpoint is nested in the key 'model'
    if state_dict.keys() == {'model'}:
        state_dict = state_dict['model']
    # Lightning checkpoint contains extra stuff, we only want the model state dict
    if 'pytorch-lightning_version' in state_dict:
        state_dict = {remove_prefix(k, 'model.'): v for k, v in state_dict['state_dict'].items()}
    return state_dict

def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

#也就是对prompt进行构造
#这是对选择题进行构造
def format_example_2(df, idx, include_answer=True):
    prompt = df.loc[idx, 0]
    if include_answer:
        prompt+=" {}\n\n".format(df.iloc[idx, -1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

@torch.no_grad
def eval_MMLU(subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []

    for i in range(test_df.shape[0]):#一行一行地读取
        # get prompt and make sure it fits
        k = ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)#是不是few-shot
        prompt = train_prompt + prompt_end
#对prompt进行读取
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        label = test_df.iloc[i, test_df.shape[1]-1]
        max_length = (input_ids.shape[1] + 1)
        model.to('cuda')
        fn = lambda: model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    output_scores=True,
                    return_dict_in_generate=True
        )
        result=fn()
        logits=result.scores[0].detach().cpu().numpy().flatten()
        
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[0]],
                        logits[tokenizer("B").input_ids[0]],
                        logits[tokenizer("C").input_ids[0]],
                        logits[tokenizer("D").input_ids[0]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print(all_probs.shape)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

def eval_lambada(model, tokenizer):
    cors = []
    all_probs = []
    dataset = load_dataset("lambada")
    test_dataset=pd.DataFrame(dataset['test'])
    test_df=pd.DataFrame()
    i=0
    for line in test_dataset['text']:
           
            test_df.loc[i,0]=' '.join(line.split()[:-1])
            test_df.loc[i,1]=' '+line.split()[-1]
            i=i+1

    for j in range(test_df.shape[0]):
        # get prompt and make sure it fits
        
        input_ids = tokenizer(test_df.loc[j,0], return_tensors="pt").input_ids.cuda()
        #print('shape:',input_ids.shape[1])
        max_length=input_ids.shape[1]+int(tokenizer(test_df.loc[j,1],return_tensors="pt").input_ids.cuda().shape[1])
        model.to('cuda')
        fn = lambda: model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    output_scores=True,
                    return_dict_in_generate=True)
        result=fn()
        pred=tokenizer.batch_decode(result.sequences.tolist())
        print(tokenizer.batch_decode(result.sequences.tolist()))
        #对输出的结果进行解码
        sentence=test_df.loc[j,0]+test_df.loc[j,1]
        #看一下应该的输出是什么
        cor = pred == sentence
        
        cors.append(cor)
       

    acc = np.mean(cors)
    cors = np.array(cors)
    #all_probs = np.array(all_probs)

  
    print("Average accuracy {:.3f}".format(acc))
    #print(np.mean(all_probs))

    return cors, acc

def eval_hellaswag(model, tokenizer):
    dataset = load_dataset("hellaswag",trust_remote_code=True)['validation']
    preprocessed=process_docs_hellaswag(dataset)
    for i in range(len(preprocessed)):#一行一行地读取
        doc=dataset[i]
        data=preprocessed[i]
        ctx=doc["ctx_a"] + " " + doc["ctx_b"]
        prompt=[choice for choice in data['choices']]
        
#对prompt进行读取
        input_ids = tokenizer(ctx,return_tensors="pt").input_ids.cuda()
        model.to('cuda')
        fn = lambda: model.generate(
                    input_ids=input_ids,
                    output_scores=True,
                    return_dict_in_generate=True,
                    
        )
        result=fn()
        logits=result.scores[0].detach().cpu().numpy().flatten()
        print(len(logits[0]))
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer(prompt[0][1]).input_ids[0]],
                        logits[tokenizer(prompt[1][1]).input_ids[0]],
                        logits[tokenizer(prompt[2][1]).input_ids[0]],
                        logits[tokenizer(prompt[3][1]).input_ids[0]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "1", 1: "2", 2: "3", 3: "4"}[np.argmax(probs)]
        cor = pred == doc['label']
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print(all_probs.shape)
    print("Average accuracy {:.3f} - {}".format(acc))
    

    
        

def eval_PIQA(model, tokenizer):
    dataset = load_dataset("piqa")
    val_dataset=dataset['validation']
    test_dataset=pd.DataFrame()
    i=0
    labels=[]
    for data in val_dataset:
        test_dataset.loc[i,0]=data['goal']+data['sol1']
        i=i+1
        test_dataset.loc[i,1]=data['goal']+data['sol2']
        i=i+1
        labels.append(data['label'])
    result=[]
    for i in range(len(test_dataset)):
        input_ids=tokenizer(test_dataset.loc[i,0], return_tensors="pt").input_ids.cuda()
   
        model.to('cuda')
        model.eval()
        output=model(
                    input_ids=input_ids,
                    num_last_tokens=1,
        )
        if i==0:
            result=output.logits.detach().cpu().numpy().flatten()
        else:
            result=np.vstack((result,output.logits.detach().cpu().numpy().flatten()))
    np.savetxt('result.csv',result)
    return 

def eval_arc(model,tokenizer):
    dataset = load_dataset('ai2_arc','ARC-Easy')
    test_dataset=dataset['test']
    test_df=pd.DataFrame()
    labels=pd.Dataframe()
    j=0
    for i in range(len(test_dataset)):
        for k in len(test_dataset[i]['choices']['text']):
            test_df.loc[j,'input']=test_dataset[i]['question']+test_dataset[i]['choices']['text'][k]
            test_df.loc[j,'id']=test_dataset[i]['id']
            test_df.loc[j,'label']=test_dataset[i]['choices']['labels'][k]
            j=j+1
        labels.loc[i,'id']=test_dataset[i]['id']
        labels.loc[i,'answerKey']=test_dataset[i]['answerkey']
    predictions={}
    for k in range(len(labels)):
        result=[]
        for i in range(4):
            input_ids = tokenizer(test_dataset.loc[4*k+i,0], return_tensors="pt").input_ids.cuda()
            
            model.to('cuda')
            model.eval()
            output=model.forward(
                        input_ids=input_ids,
                        num_last_tokens=1,
            )
            result.append(output.logits.squeeze().cpu().tolist())
        pred=np.argmax(np.array(result))
        predictions[labels.loc[k,'pred']]=pred
    with open('result_arc_easy.json','w') as f:
        json.dump(predictions,f)
    return 
def eval_arc_challenge(model,tokenizer):
    dataset = load_dataset('ai2_arc','ARC-Challenge')
    test_dataset=dataset['test']
    test_df=pd.DataFrame()
    labels=pd.Dataframe()
    j=0
    for i in range(len(test_dataset)):
        for k in len(test_dataset[i]['choices']['text']):
            test_df.loc[j,'input']=test_dataset[i]['question']+test_dataset[i]['choices']['text'][k]
            test_df.loc[j,'id']=test_dataset[i]['id']
            test_df.loc[j,'label']=test_dataset[i]['choices']['labels'][k]
            j=j+1
        labels.loc[i,'id']=test_dataset[i]['id']
        labels.loc[i,'answerKey']=test_dataset[i]['answerkey']
    predictions={}
    for k in range(len(labels)):
        result=[]
        for i in range(4):
            input_ids = tokenizer(test_dataset.loc[4*k+i,0], return_tensors="pt").input_ids.cuda()
            
            model.to('cuda')
            model.eval()
            output=model.forward(
                        input_ids=input_ids,
                        num_last_tokens=1,
            )
            result.append(output.logits.squeeze().cpu().tolist())
        pred=np.argmax(np.array(result))
        predictions[labels.loc[k,'pred']]=pred
    with open('result_arc_challenge.json','w') as f:
        json.dump(predictions,f)
    return 

def winogrande(model,tokenizer):
    dataset=load_dataset('winogrande')
    
    test_dataset=dataset['validation']

def benchmark(config: DictConfig) -> None:
     # load model
    OmegaConf.set_struct(config, False)

    checkpoint_type = 'pytorch'
    if checkpoint_type not in ['lightning', 'pytorch']:
        raise NotImplementedError(f'checkpoint_type ${checkpoint_type} not supported')

    if checkpoint_type == 'lightning':
        cls = hydra.utils.get_class(config.task._target_)
        model = cls.load_from_checkpoint(checkpoint_path=config.eval.ckpt)
    elif checkpoint_type == 'pytorch':
        model_cfg = config.model_pretrained if 'model_pretrained' in config else None
        trained_model: LightningModule = hydra.utils.instantiate(config.task, cfg=config,
                                                                 model_cfg=model_cfg,
                                                                 _recursive_=False)
        if 'ckpt' in config.eval:
            load_return = trained_model.model.load_state_dict(
                load_checkpoint(config.eval.ckpt, device=trained_model.device), strict=False
            )
            
            log.info(load_return)
        if 'model_pretrained' in config:
            ...
        else:
            model = trained_model.model
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    #model.eval()
    model.eval()
    '''subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir,'MMLU', "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, "results_{}".format('mambapp'))):
        os.makedirs(os.path.join(save_dir, "results_{}".format('mambapp')))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(data_dir, 'MMLU',"dev", subject + "_dev.csv"), header=None
        )[: ntrain]
        test_df = pd.read_csv(
            os.path.join(data_dir,'MMLU', "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval_MMLU(subject, model, tokenizer, dev_df,test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format('mambapp')] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df['{}_choice{}_probs'.format('mambapp', choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                save_dir, "results_{}".format('mambapp'), "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))'''
    print('evaluate base on lambada')
    path_lambada='/content/drive/MyDrive/data/LAMBADA/lambada_test_plain_text.txt'
    path_hellaswag='/content/drive/MyDrive/data/HellaSwag/hellaswag_val.jsonl'
    eval_lambada(model, tokenizer)
    #eval_hellaswag(model,tokenizer)


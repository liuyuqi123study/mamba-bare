

import json
import logging
import requests
import torch
from typing import Dict, List, Optional
import sys
sys.path.append('/content/drive/MyDrive/mamba-bare-main/train')
from utils.utils import get_logger
from pathlib import Path
log = get_logger(__name__)
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


class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

        if 'Yarn-Llama' in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}
        
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=name_or_path,
                tokenizer=self.tokenizer,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                model_kwargs=model_kwargs,
            )
        except:
            self.pipeline = None
            self.model = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16,)
            
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            generated_ids = self.model.generate(
                **inputs,
                **self.generation_kwargs
            )
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            output = self.pipeline(text_inputs=prompts, **self.generation_kwargs, )
            assert len(output) == len(prompts)
            # output in the form of a list of list of dictionaries
            # outer list len = batch size
            # inner list len = 1
            generated_texts = [llm_result[0]["generated_text"] for llm_result in output]

        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            if text.startswith(prompt):
                text = text[len(prompt):]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        return results


class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        from mambapp import MambaLMHeadModel,MambaConfig
        self.config = json.load(open('/content/drive/MyDrive/mamba-bare-main/train/logs/runs/2024-08-29/03-25-31/config.json'))
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model=MambaLMHeadModel(MambaConfig(**self.config),device='cuda')
        self.model.load_state_dict(state_dict=load_checkpoint('/content/drive/MyDrive/mamba-bare-main/train/mamba130M_thepile_step_20000.ckpt', device='cuda'), strict=False)
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.max_genlen = self.generation_kwargs.pop('max_new_tokens')
        self.minp = 0.0

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen
        
        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            #cg=True,
            return_dict_in_generate=True,
            output_scores=False,
            #top_k=1,
            #enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {'text': [self.tokenizer.decode(out.sequences[0][input_ids.shape[1]:])]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        # FIXME: naive implementation
        return [self.__call__(prompt, **kwargs) for prompt in prompts]

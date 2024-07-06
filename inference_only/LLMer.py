import os
import json
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModel 
import transformers
import torch


MODEL_MAP = {
    'google/flan-t5-large': ('pipeline-text2text', 'concat'),
    'google/flan-t5-xl': ('pipeline-text2text', 'concat'),
    'google/flan-t5-xxl': ('pipeline-text2text', 'concat'),
    
    'decapoda-research/llama-7b-hf': ('CausalLM', 'concat'),
    'decapoda-research/llama-13b-hf': ('CausalLM', 'concat'),
    
    'meta-llama/Llama-2-7b-hf': ('CausalLM', 'concat'),
    'meta-llama/Llama-2-13b-hf': ('CausalLM', 'concat'),
    'meta-llama/Llama-2-7b-chat-hf': ('CausalLM', 'lamma2-chat'),
    'meta-llama/Llama-2-13b-chat-hf': ('CausalLM', 'lamma2-chat'),
    'meta-llama/Llama-2-70b-chat-hf': ('CausalLM', 'lamma2-chat'),
    
    'tiiuae/falcon-7b': ('CausalLM', 'concat'),
    'tiiuae/falcon-7b-instruct': ('CausalLM', 'concat'),
    'tiiuae/falcon-40b': ('CausalLM', 'concat'),
    'tiiuae/falcon-40b-instruct': ('CausalLM', 'concat'),
    
    'baichuan-inc/Baichuan-7B': ('CausalLM', 'concat'),
    'baichuan-inc/Baichuan-13B-Base': ('CausalLM', 'concat'),
    'baichuan-inc/Baichuan2-7B-Base': ('CausalLM', 'concat'),
    'baichuan-inc/Baichuan2-13B-Base': ('CausalLM', 'concat'),
    'baichuan-inc/Baichuan-13B-Chat': ('CausalLM', 'baichuan2-chat'),
    'baichuan-inc/Baichuan2-7B-Chat': ('CausalLM', 'baichuan2-chat'),
    'baichuan-inc/Baichuan2-13B-Chat': ('CausalLM', 'baichuan2-chat'),
    
    'THUDM/chatglm-6b': ('AutoModel-half', 'chatglm2-chat'),
    'THUDM/chatglm2-6b': ('AutoModel-half', 'chatglm2-chat'),
    
    'Qwen/Qwen-14B': ('CausalLM', 'concat'),
    'Qwen/Qwen-7B-Chat': ('CausalLM', 'qwen-chat'),
    'Qwen/Qwen-14B-Chat': ('CausalLM', 'qwen-chat'),
    
    'internlm/internlm-7b': ('CausalLM', 'concat'),
    'internlm/internlm-20b': ('CausalLM', 'concat'),
    'internlm/internlm-chat-7b-v1_1': ('CausalLM', 'internlm-chat'),
    'internlm/internlm-chat-20b': ('CausalLM', 'internlm-chat'),
    
    'lmsys/vicuna-7b-v1.5': ('CausalLM', 'vicuna'),
    'lmsys/vicuna-13b-v1.5': ('CausalLM', 'vicuna'),
    
    'WizardLM/WizardLM-13B-V1.2': ('CausalLM', 'vicuna'),
}

class LLMer():
    def __init__(self, model_name, local_dir=None):
        self.model_name = model_name
        if local_dir:
            self.model_name_or_path = os.path.join(local_dir, model_name) # load from local
        else:
            self.model_name_or_path = model_name # load fron internet
            
        self.load_type, self.query_type = MODEL_MAP[model_name]
        
        self.build_model()
        self.case_test()
    
    def build_model(self):
        start_time = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False, trust_remote_code=True)
        
        if self.load_type == 'pipeline-text2text':
            model = transformers.pipeline("text2text-generation", 
                                          model=self.model_name_or_path, 
                                          tokenizer=tokenizer,
                                          device_map="auto",
                                          model_kwargs={'use_safetensors':False}) 
            
        elif self.load_type == 'pipeline':
            model = transformers.pipeline("text-generation", 
                                          model=self.model_name_or_path, 
                                          tokenizer=tokenizer,
                                          device_map="auto",
                                          trust_remote_code=True,
                                          model_kwargs={'use_safetensors':False})    
            
        elif self.load_type == 'CausalLM':
            use_safetensors = False if self.query_type not in ['qwen-chat'] else True
            model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path,
                                                         device_map="auto",
                                                         trust_remote_code=True,
                                                         use_safetensors=use_safetensors)
                                                        # torch_dtype=torch.float16, # only for 70B model
            model.eval()
            
            if self.query_type in ['baichuan2-chat', 'qwen-chat']:
                model.generation_config = GenerationConfig.from_pretrained(self.model_name_or_path,trust_remote_code=True)
                
        elif self.load_type == 'AutoModel-half':
            model = AutoModel.from_pretrained(self.model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=True,
                                             use_safetensors=False).half()
            model.eval()
            
        else:
            raise NotImplementedError
        
        self.tokenizer = tokenizer
        self.model = model
        
        end_time = time.time()
        print(f"Load {self.model_name}: {end_time-start_time:.2f}s")
    
    def make_query(self, input_seq, system_prompt, examples, force_quer_type=None):
        query_type = self.query_type if force_quer_type is None else force_quer_type
        
        if query_type == 'concat':
            query = system_prompt
            for example in examples:
                query += f"{example['user']} {example['assistant']}\n"
            query += input_seq  
            
        elif query_type == 'lamma2-chat':
            # ref: https://gpus.llm-utils.org/llama-2-prompt-template/
            query = ""
            query += f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n" # tokenizer add the first <s>
            for example in examples:
                query += f"{example['user']} [/INST] {example['assistant']}</s><s>[INST] "
            query += f"{input_seq} [/INST]"
            
        elif query_type == 'baichuan2-chat':
            # ref: https://github.com/baichuan-inc/Baichuan2
            # ref: https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/blob/main/modeling_baichuan.py
            query = []
            if len(examples) == 0:
                query.append({"role": "user", "content": f"{system_prompt}\n{input_seq}"})
            else:
                query += [{"role": "user", "content": f"{system_prompt}\n{examples[0]['user']}"},
                        {"role": "assistant", "content": f"{examples[0]['assistant']}"}]
                for example in examples[1:]:
                    query += [{"role": "user", "content": f"{example['user']}"},
                            {"role": "assistant", "content": f"{example['assistant']}"}]
                query.append({"role": "user", "content": f"{input_seq}"})
                                    
        elif query_type in ['chatglm2-chat', 'qwen-chat', 'internlm-chat']:
            if len(examples) == 0:
                query = {
                    'query': f"{system_prompt}\n{input_seq}",
                    'history': [],
                }
            else:
                history = [(f"{system_prompt}\n{examples[0]['user']}",f"{examples[0]['assistant']}")]
                for example in examples[1:]:
                    history.append((f"{example['user']}",f"{example['assistant']}"))
                query = {
                    'query': f"{input_seq}",
                    'history': history,
                }
                
        elif query_type in ['vicuna']:   
            # ref: https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#prompt-template
            # ref: https://huggingface.co/WizardLM/WizardLM-70B-V1.0
            fixed_prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
            if len(examples) == 0:
                query = fixed_prefix + f"USER: {system_prompt}\n{input_seq} ASSISTANT:"
            else:
                query = fixed_prefix + f"USER: {system_prompt}\n{examples[0]['user']} ASSISTANT: {examples[0]['assistant']} </s>"
                for example in examples[1:]:
                    query+= f"USER: {example['user']} ASSISTANT: {example['assistant']} </s>"
                query += f"USER: {input_seq} ASSISTANT:"
                
        else:
            raise NotImplementedError
    
        return query
    
    def get_response(self, query, **kwargs):
        if self.load_type in ['CausalLM', 'AutoModel-half']:
            if self.query_type in ['concat', 'lamma2-chat', 'vicuna']:
                inputs = self.tokenizer(query, return_tensors="pt").to(self.model.device)
                output = self.model.generate(
                    input_ids = inputs['input_ids'],
                    attention_mask = inputs['attention_mask'],
                    **kwargs,
                )
                len_of_inputs = len(inputs['input_ids'][0]) # batch size = 1; only one sample
                output_str = self.tokenizer.decode(output[0][len_of_inputs:], skip_special_tokens=True)
                return output_str
            
            elif self.query_type in ['baichuan2-chat']:
                output = self.model.chat(self.tokenizer, query)
                return output
            
            elif self.query_type in ['chatglm2-chat', 'qwen-chat', 'internlm-chat']:
                output = self.model.chat(self.tokenizer, query['query'], query['history'])
                return output[0]
            
        elif self.load_type in ['pipeline', 'pipeline-text2text']:
            output = self.model(query, **kwargs)
            return output[0]['generated_text']
        
        else:
            raise NotImplemented
    
    def case_test(self):
        system_prompt = "You are a robot.\n"
        sample = "Hello!"
        query = self.make_query(sample, system_prompt=system_prompt, examples=[])
        response = self.get_response(query, max_new_tokens=5)
        return True if len(response.strip()) > 0 else False
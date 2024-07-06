import json
import re
from tqdm import tqdm
import numpy as np
import argparse


# API and proxy setting
import os
import sys

from OpenaiAPI import OpenaiAPIMaster
from LLMer import LLMer


# os.environ['http_proxy'] = ""
# os.environ['https_proxy'] = ""


def get_params():
    parser = argparse.ArgumentParser(description='verify.py')

    # dateset
    parser.add_argument("--data_path", type=str, default='') 
    parser.add_argument("--save_path", type=str, default='') 
    
    # model
    parser.add_argument("--call_type", type=str, default='api', choices=['api', 'llm']) 

    parser.add_argument("--model_name", type=str, default='gpt-4-0613')  
    parser.add_argument("--api_key", type=str, default='')
    parser.add_argument("--local_dir", type=str, default='')  # for local llm


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    
    return args

  
if __name__ == '__main__':
    args = get_params()

    if args.call_type == 'api':
        # API
        model_name = args.model_name
        api_key = args.key
        api_master = OpenaiAPIMaster(model_name, api_key)
        api_master.handshake()

    elif args.call_type == 'llm':
        llm = LLMer(args.model_name, args.local_dir)

    else: 
        raise NotImplementedError
    
    # data
    datas = json.load(open(args.data_path))

    for item in tqdm(datas):

        if args.call_type == 'api':
            generate_args = {
                'temperature': 0.7,
                'n': 1,
            }

            query = [
                {"role": "system", "content": item['instruction']},
                {"role": "user", "content": item['input']},
            ]

            response = api_master.chat_query(query, retry = True, generate_args = generate_args)
            response = response["choices"][0]

            item['pred_info'] = {}
            item['pred_info'].update({'pred_str': response})

        elif args.call_type == 'llm':
            system_prompt = item['instruction'] 
            examples = []
            sample = item['input']

            query = llm.make_query(sample, system_prompt=system_prompt, examples=examples)
            response = llm.get_response(query)

            item['pred_info'] = {}
            item['pred_info'].update({'pred_str': response})

    save_path = args.save_path
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines([json.dumps(item, ensure_ascii=False)+'\n' for item in datas])

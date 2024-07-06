import json
from tqdm import tqdm
import pandas as pd
import argparse

from OpenaiAPI import OpenaiAPIMaster
from eval_prompts import *


def eval_func_decomposed(input_info, pred_info, qtype, id,
                         api_master):

    def _parse(s):
        if s.lower().strip().startswith("yes"):
            return 1
        elif s.lower().strip().startswith("no"):
            return 0
        else:
            return None

    def _get_response(query):
        response = api_master.chat_query(query, retry = True, generate_args = generate_args)
        a = [_parse(i["message"]["content"]) for i in response["choices"]]        
        return a
    

    generate_args = {
        'temperature': 1.0,
        'n': 1,
    }

    argument = input_info['P']
    argument_m = pred_info['pred_str'].split("\n")[0].strip()
    premise2 = input_info['Om']
    assert "Premise2:" in premise2
    premise2 = premise2.split("Premise2:")[1].strip()

    premise1 = input_info['O']
    assert "Premise1:" in premise1
    premise1 = premise1.split("Premise1:")[1].strip()

    metrics = {} 
    metrics['id_string'] = id

    # item 1 (1)
    input_item_1_1 = f"""
Argument: {argument}
Premise: {premise1}
"""

    # item 1 (2) / item 2 (2)
    input_item_12_2 = f""" 
Argument: {argument_m}
Premise: {premise2}
"""

    # item 2 (1)
    input_item_2_1 = f""" 
Argument: {argument}
Premise: {premise2}
"""

    system_prompts_12 = get_system_prompts(qtype)
    metrics['decomposed'] = {}
    metrics['ses'] = {}
    for k,v in system_prompts_12.items():
        query_11 = [
            {"role": "system", "content": v},
            {"role": "user", "content": input_item_1_1},
        ]
        query_12_22 = [
            {"role": "system", "content": v},
            {"role": "user", "content": input_item_12_2},
        ]
        query_21 = [
            {"role": "system", "content": v},
            {"role": "user", "content": input_item_2_1},
        ]

        c_11 = _get_response(query_11)
        c_12_22 = _get_response(query_12_22)
        c_21 = _get_response(query_21)  
        
        metrics['decomposed'][k] = {}
        metrics['decomposed'][k]['c_11'] = c_11
        metrics['decomposed'][k]['c_12_22'] = c_12_22
        metrics['decomposed'][k]['c_21'] = c_21

        metrics['ses'][k] = {}
        metrics['ses'][k] = _ses_score(id=id, metrics=metrics)

    return metrics

def _ses_score(id, metrics):
    try:
        scores = metrics['decomposed']['system_prompt_ses']
    except:
        print(f'WARNING: lacking {id}-th result!!')
        return 0.  
    c_11 = scores['c_11'][0]
    c_12_22 = scores['c_12_22'][0]
    c_21 = scores['c_21'][0]
    
    s31 = 1 if (c_11 == 1 and c_12_22 == 1) else 0
    s32 = 1 if (c_21 == 0 and c_12_22 == 1) else 0
    ses = s31 + s32    
    ses = 1 if ses >= 1 else 0
    
    return ses

def ses_score(batch):
    if len(batch) == 0:
        return None
    scores = [item['metrics']['ses']['system_prompt_ses'] for item in batch]
    scores = sum(scores)/len(scores)
    return scores

def write2excel(column_names:list[str],
                data_list:list[list], sheet_names:list[str],
                write_f:str):

    assert len(data_list) == len(column_names) == len(sheet_names), \
        "{}, {}, {}".format(len(data_list), len(column_names), len(sheet_names))
    df_names = locals()
    for i, data in enumerate(data_list):
        df_names["df" + str(i)] = pd.DataFrame(data, columns=column_names[i])
    with pd.ExcelWriter(write_f) as writer:
        for i in range(len(data_list)):
            _sheet_name = sheet_names[i] if sheet_names[i] is not None else "any"
            _sheet_name = _sheet_name.split("/")[-1][-30:]
            df_names["df" + str(i)].to_excel(writer, sheet_name=_sheet_name, index=False)
    print("Written to {}.\n\n".format(write_f))   


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_pred_file', type=str, help='path to a pred json file')
    parser.add_argument('--api_model', type=str, help='specify the LLM model for SES tasks')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--api_org', type=str, default=None)

    args = parser.parse_args()

    api_master = OpenaiAPIMaster(args.api_model, args.api_key, api_org=args.api_org)
    api_master.handshake()

    datas = json.load(open(args.model_pred_file))

    for item in tqdm(datas):
        score = eval_func_decomposed(input_info = item['input_info'], 
                    pred_info = item['pred_info'],
                    qtype = item['qtype'],
                    id = item['id_string'],
                    api_master = api_master)
    
        item['metrics'] = score
    
    # batch SES
    s = ses_score(datas)
    s_necessary = ses_score([d for d in datas if d['qtype'] == 0 ])
    s_sufficient = ses_score([d for d in datas if d['qtype'] == 1 ])
    s_strengthen = ses_score([d for d in datas if d['qtype'] == 2 ])
    s_weaken = ses_score([d for d in datas if d['qtype'] == 3 ])

    # save.
    save_path = args.model_pred_file.replace('.json','.ses.json')
    with open(save_path,'w') as f:
        json.dump(datas, f, indent=4)
    print(f'Saved to {save_path}')

    save_path_batch = save_path.replace('.json', '.xlsx')
    header = ['ses', 'ses_necessary', 'ses_sufficient', 'ses_strengthen', 'ses_weaken', 'file_name']
    output = [[s, s_necessary, s_sufficient, s_strengthen, s_weaken, args.model_pred_file]]
    write2excel(column_names=[header], data_list=[output], sheet_names=['results'], write_f=save_path_batch)

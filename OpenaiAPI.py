import json
import openai
from multiprocessing.pool import ThreadPool
import threading
import time
import os

# Ref: https://platform.openai.com/docs/api-reference/introduction

class OpenaiAPIMaster:
    def __init__(self, model_name = 'gpt-4', api_key = None, api_org = None):
        self.model_name = model_name
        
        if api_key is None:
            if openai.api_key is None:
                print("Please set the api_key")
        else:
            openai.api_key = api_key
            if api_org is not None:
                openai.org = api_org
        
        print(f"api_key: {openai.api_key}")
        print(f"model_name: {model_name}")
        print(f"http/https proxy: {os.environ.get('http_proxy', 'No proxy')} / {os.environ.get('https_proxy', 'No proxy')}")
        
        
    
    def chat_query(self, query, retry = True, generate_args = {}):
        
        done = False
        while not done:
            
            try:
                # check query
                # messages = [{"role": "system", "content": "You are a helpful AI assistant."},]
                messages = []
                if isinstance(query, str):
                    messages.append(
                        {"role": "user", "content": query},
                    )
                elif isinstance(query, list):
                    messages += query
                else:
                    raise ValueError("Unsupported query: {0}".format(query))

                # get response
                response = openai.ChatCompletion.create(
                                model = self.model_name,
                                messages = messages,
                                **generate_args
                )
                
                done = True
                
            except Exception as e:
                print(str(e))
                response = {'choices': [{'message': {'content': ""}}]}
                
                if 'Unsupported query' in str(e):
                    done = True
                    
                if 'Rate limit reached' in str(e):
                    if retry:
                        time.sleep(1)
                        print('Retrying......')
                        done = False
                    else:
                        done = True  
                
            
        return response
    
    def handshake(self):
        start_time = time.time()
        
        self.chat_query("hello!")
        
        run_time = time.time() - start_time
        print(f"{run_time:.2f} seconds")
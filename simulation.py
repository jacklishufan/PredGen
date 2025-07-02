# this is a script used to covert output from whipser to a simulated text input stream
folder = '/data1/jacklishufan/outputs_audio/*/*_ctx_transcribed.json'
from pathlib import Path
from glob import glob
import json
files = glob(folder)
# breakpoint()
def build_logs(files=files,offset=0.1):
    all_entries = {}
    for row in files:
        # load json
        with open(row, 'r') as f:
            data = json.load(f)
        
        segments = [x['segments'] for x in data['out']]
        texts = [''.join([x['text'] for x in s]) for s in segments]
        timesteps = [x['time']-data['out'][0]['time'] for x in data['out']]
        logs = [(text,timestep) for text,timestep in zip(texts,timesteps) ]
        cleanned_log = []
        for log in logs:
            past_log = ''
            if len(cleanned_log) > 0:
                past_log =  cleanned_log[-1][0]
            if log[0] != past_log:
                cleanned_log.append(log)
        row_raw = row.replace('_ctx_transcribed.json','_ctx.json')
        with open(row_raw, 'r') as f:
            data_raw = json.load(f)
        # breakpoint()
        if 'prompt' in data_raw:
            prompt = data_raw['prompt']
        else:
            prompt = data_raw['question']
            
        all_entries[prompt] = cleanned_log
        
    return all_entries

# ALL_LOGS = build_logs()
# breakpoint()
with open ('simulated_inputs.json','w') as f:
    f.write(json.dumps(build_logs(),indent=4,ensure_ascii=False))
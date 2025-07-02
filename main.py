import datasets
from dataset import MMLUProDataset, BaseDataset, MMLUDataset,AccMeter
from prompt_speculation_v2 import streamer_generate,baseline_generate,CustomHFStreamer
from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
import torch
import time 
from tqdm.auto import tqdm
import torch
import functools
import os
import re

def preprocessor_mmlu_pro_cot(prompt):
    messages = [
        {"role": "system", "content": 'You are a helpful assistant. Solve the problem by thinking step by step. The instruction given by user may be truncated. In this case, you should give answers based on your best guesses of the incomplete instruction. Do not complain about incomplete prompts. '},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def preprocessor_gsm8k_pro_cot(prompt):
    messages = [
        {"role": "system", "content": 'You are a helpful assistant. Solve the problem by thinking step by step. The instruction given by user may be truncated. In this case, you should give answers based on your best guesses of the incomplete instruction. Do not complain about incomplete prompts. '},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text
class CONFIG:
    HAS_SYSTEM = True
def preprocessor_mt_bench(prompt,history=[]):
    messages = [
        {"role": "system", "content": 'You are a helpful assistant. The instruction given by user may be truncated. In this case, you should give answers based on your best guesses of the incomplete instruction. Do not complain about incomplete prompts. '},
        *history,
        {"role": "user", "content": prompt}
    ]
    if not CONFIG.HAS_SYSTEM:
        messages[0]={"role": "user", "content": 'You are a helpful assistant. The instruction given by user may be truncated. In this case, you should give answers based on your best guesses of the incomplete instruction. Do not complain about incomplete prompts. The conversation starts now:'}
        messages.insert(1, {"role": "assistant", "content": 'OK. I will provide answer based on partial inputs.'})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def preprocessor_humaneval(instruction):
    lang_name = 'python'
    system = 'You are an AI programming assistant, and answer questions related to computer science. The instruction given by user may be truncated. In this case, you should give answers based on your best guesses of the incomplete instruction. Do not complain about incomplete prompts. '
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text
sentence_breaks = ['.','?','!']
def inference(prompt,model,tokenizer,preprocessor,max_length=2048,use_hf=False):
    torch.cuda.synchronize()
    t0 = time.time()
    current_prompt_processed = preprocessor(prompt)
    inputs_ids =  tokenizer([current_prompt_processed],return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')# model.device
    #outputs = model.generate(inputs_ids,use_cache=True,repetition_penalty=1,do_sample=False,max_new_tokens=max_length)
    nfe = 0
    timestampe_first_stentence = None
    nfe_to_first_sentence =None
    if use_hf:
        # simulate streaming
        streamer = CustomHFStreamer(tokenizer)
        outputs = model.generate(inputs_ids,use_cache=True,repetition_penalty=1,do_sample=False,max_new_tokens=max_length,streamer=streamer)
        timestampe_first_stentence = streamer.timestemp_first_sentence
        nfe_to_first_sentence = streamer.nfe_to_first_sentence
        outputs = outputs[:,inputs_ids.shape[-1]:]
        gen_text = tokenizer.batch_decode(outputs)[0].replace(tokenizer.eos_token,'')
    else:
        outputs,_,timestampe_first_stentence,nfe_to_first_sentence = baseline_generate(inputs_ids,model,tokenizer, DynamicCache(),512)
        gen_text = tokenizer.batch_decode(outputs)[0].replace(tokenizer.eos_token,'')
    torch.cuda.synchronize()
    t1 = time.time()
    latency = t1-t0
    if timestampe_first_stentence is None:
        ttfs = None
    else:
        ttfs = timestampe_first_stentence - t0
    return dict(
        gen_text=gen_text,
        ttfs=ttfs,
        nfe_to_first_sentence=nfe_to_first_sentence,
        latency=latency,
    )

import json
import argparse
import datasets

def do_inference_multi(args,prompt,model,tokenizer,preprocessor,max_length):
    if args.decoding == 'hf':
        extra_data = inference(prompt,model,tokenizer,preprocessor,max_length=max_length,use_hf=True)
    elif args.decoding == 'greedy':
        _,_,extra_data =  streamer_generate(prompt,model,tokenizer,speed=args.speed,preprocessor=preprocessor,verbose=args.verbose,final_text='',max_len=max_length,use_logs=args.simulate)
    elif args.decoding == 'model':
        _,_,extra_data =  streamer_generate(prompt,model,tokenizer,speed=args.speed,preprocessor=preprocessor,verbose=args.verbose,final_text='',max_len=max_length,use_logs=args.simulate,
                                            acceptance='model_query')
    elif args.decoding == 'sd':
        _,_,extra_data =  streamer_generate(prompt,model,tokenizer,speed=args.speed,preprocessor=preprocessor,verbose=args.verbose,final_text='',max_len=max_length,
                                            acceptance='topk',top_k=args.topk,use_logs=args.simulate)
    elif args.decoding == 'cllm':
         _,_,extra_data =  streamer_generate(prompt,model,tokenizer,speed=args.speed,preprocessor=preprocessor,verbose=args.verbose,final_text='',max_len=max_length,generation_mode='jacobi',use_logs=args.simulate)
    else:
        raise NotImplementedError()
    return extra_data


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--num_questions',type=int,default=100)
    parser.add_argument('-j','--task',type=str,choices=['mt-bench','mmlu-pro','humaneval','gsm8k','lmsys'],default='mmlu-pro')
    parser.add_argument('-s','--seed',type=int,default=0)
    parser.add_argument('-l','--max_length',type=int,default=2048)
    parser.add_argument('-m','--model_p',type=str,default=None)
    parser.add_argument('-lp','--lora_p',type=str,default=None)
    parser.add_argument('-d','--decoding',type=str,choices=['greedy','cllm','sd','hf','model','cllm'],default='hf')
    parser.add_argument('-v','--verbose',action='store_true')
    parser.add_argument('-sp','--speed',type=int,default=600)
    parser.add_argument('-o','--output',type=str,default='output_new_sd')
    parser.add_argument('--topk',type=int,default=5)
    parser.add_argument('-f','--overwrite',action='store_true')
    parser.add_argument('--skip',action='store_true')
    parser.add_argument('--simulate',action='store_true')
    
    #parser.add_argument('-d','--max_length',type=int,default=2048)
    args = parser.parse_args()
    model_p = args.model_p
    lora_path = args.lora_p
    if 'gemma' in model_p:
        CONFIG.HAS_SYSTEM = False
    num_questions = args.num_questions
    seed = args.seed
    max_length = args.max_length
    output_name = args.output
    # breakpoint()
    
    if args.task == 'mmlu-pro':
        dataset = MMLUProDataset('TIGER-Lab/MMLU-Pro')
        preprocessor = preprocessor_mmlu_pro_cot
    elif args.task == 'mt-bench':
        dataset = datasets.load_dataset('philschmid/mt-bench')['train']
    elif args.task == 'humaneval':
        # cllm_dir = Path(__file__)
        data_abs_dir = os.path.join('train/original_eval/evaluation/humaneval/data')
        lang = 'python'
        raw_data = [json.loads(x) for x in open(
                os.path.join(data_abs_dir,f'humaneval-{lang}.jsonl')
                ) if x.strip()]
        #raw_data = {x['task_id']:x for x in raw_data}
        dataset = raw_data[:args.num_questions]
        preprocessor = preprocessor_humaneval
        #preprocessor_humaneval(question)
    elif args.task == 'gsm8k':
        from live_mind.utils.dataset.gsm8k import GSM8KTemplate
        dataset = datasets.load_dataset('openai/gsm8k','main',split='test')
        shot_dataset  = datasets.load_dataset('openai/gsm8k','main',split='train')
        shots_set = []
        for data in shot_dataset:
                shots_set.append(data)
        preprocessor = preprocessor_gsm8k_pro_cot
    elif args.task == 'lmsys':
        dataset = datasets.load_dataset('jacklishufan/lmsys-processed-filtered-eng')['train']
        preprocessor = preprocessor_mt_bench
    else:
        raise NotImplementedError()
    
    output_name = output_name +f'_task_{args.task}'
    if args.decoding == 'hf':
        output_name = output_name + f'_hf_seed_{seed}_len_{max_length}_n_{num_questions}.json'
    elif args.decoding == 'greedy':
        output_name = output_name + f'_greedy_seed_{seed}_len_{max_length}_n_{num_questions}_speed_{args.speed}.json'
    elif args.decoding == 'model':
        output_name = output_name + f'_model_seed_{seed}_len_{max_length}_n_{num_questions}_speed_{args.speed}.json'
    elif args.decoding == 'sd':
        output_name = output_name + f'_sd_topk_seed_{seed}_len_{max_length}_n_{num_questions}_speed_{args.speed}_topk_{args.topk}.json'
    elif  args.decoding == 'cllm':
        output_name = output_name + f'_greedy_cllm_seed_{seed}_len_{max_length}_n_{num_questions}_speed_{args.speed}.json'
    if args.simulate:
        output_name = output_name.replace('.json','_simulated.json')
    if os.path.exists(output_name) and not args.overwrite:
        print("File Exist!",output_name)
        exit(0)
    if args.task == 'mmlu-pro':
        dataset.select(num_questions, randomize=True, seed=seed, split='test')
        tqdm_bar = tqdm(dataset.selected_questions)
    elif args.task == 'gsm8k':
        if len(dataset) > num_questions:
            seed = 42
            dataset = dataset.shuffle(seed=seed).select(range(num_questions))
        tqdm_bar = tqdm(dataset)
        meter = AccMeter()
    elif args.task == 'lmsys':
        if len(dataset) > num_questions:
            seed = 42
            dataset = dataset.shuffle(seed=seed).select(range(num_questions))
        tqdm_bar = tqdm(dataset)
        meter = AccMeter()
    else:
        tqdm_bar = tqdm(dataset)
    
    all_responses = []
    num_correct = 0
    
    num_total = 0
 
    model = AutoModelForCausalLM.from_pretrained(model_p,device_map=device,torch_dtype=torch.float16)
    if 'vicuna' in model_p.lower():
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_p)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if lora_path: 
        from peft import PeftConfig, PeftModel,get_peft_model
        tokenizer = AutoTokenizer.from_pretrained(lora_path)
        model.resize_token_embeddings(len(tokenizer))
        lora_config = PeftConfig.from_pretrained(lora_path)
        model = get_peft_model(model,lora_config)
        adaptor_state = torch.load(os.path.join(lora_path,'pytorch_model.bin'),map_location='cpu')
        result = model.load_state_dict(adaptor_state,strict=False)
        assert len(result.unexpected_keys) == 0 
        del adaptor_state
        model = model.merge_and_unload()
    # breakpoint()
    for entry in tqdm_bar:
        # breakpoint()
        if args.task == 'mmlu-pro':
            question = entry["question"]
            final_text = dataset.add_str(entry)
            question = question + ' Your answer should conclude with: The anwer is _, where _ is a letter. '
            prompt = question + final_text
        elif args.task == 'mt-bench':
            #questions = 
            question,question2 = entry['turns']
            final_text = ''
            prompt = question + final_text
            preprocessor = preprocessor_mt_bench 
        elif args.task == 'lmsys':
            # breakpoint()
            conv = entry['conversation']
            if conv[0]['role'] != 'user':
                continue
            question = conv[0]['content']
            final_text = ''
            prompt = question + final_text
        elif args.task == 'humaneval':
            question =entry['prompt']
            
            insturction = f'''Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{lang}
{question}```'''
            prompt = insturction    
        elif args.task == 'gsm8k':
            question =entry['question']
            prompt = question  
            golden = GSM8KTemplate.format_answer(entry) 
            prompt: dict = GSM8KTemplate.generate_output(
                train_set=shots_set,
                input=question,
                n_shots=4,
                enable_cot=True,
            )
            prompt = prompt + ' Your answer should conclude with: The anwer is _, where _ is the numerical answer. '
        extra_data = do_inference_multi(args,prompt,model,tokenizer,preprocessor,max_length)
        if args.task == 'mmlu-pro':
            is_correct: bool = dataset.verify_answer(extra_data['gen_text'], entry["answer"])
            if is_correct:
                num_correct += 1
            num_total += 1
            acc = num_correct / num_total
            extra_data.update(dict(
                question=question,
                final_text=final_text,
                is_correct=is_correct
            ))
            all_responses.append(extra_data)
            tqdm_bar.set_description(f'ACC {acc:.4f} ({num_correct}/{num_total})')
        elif args.task == 'mt-bench':
            extra_info = dict(
                question_id=entry['question_id'],
                category=entry['category']
            )
            extra_data.update(dict(question=question,final_text=final_text,round=1,))
            history = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": extra_data['gen_text']}
            ]
            preprocessor = functools.partial(preprocessor_mt_bench,history=history)
            extra_data_2 = do_inference_multi(args,question2,model,tokenizer,preprocessor,max_length)
            extra_data_2.update(dict(question=question2,final_text=final_text,round=2))
            extra_data.update(extra_info)
            extra_data_2.update(extra_info)
            all_responses.append(extra_data)
            all_responses.append(extra_data_2)
        elif args.task == 'lmsys':
            # breakpoint()
            extra_data.update(dict(
                prompt=question,
                completion=extra_data['gen_text']
            ))
            all_responses.append(extra_data)
        elif args.task == 'humaneval':
            extra_data.update(dict(
                task_id=entry['task_id'],
                prompt=entry['prompt'],
                completion=extra_data['gen_text']
            ))
            all_responses.append(extra_data)
            # breakpoint()
        elif args.task == 'gsm8k':
            extra_data['gen_text'] = extra_data['gen_text'].replace(tokenizer.eos_token,'')
            preds = re.findall("answer is ([0-9]*)\.?$", extra_data['gen_text'].lower())
            target = entry['answer'].split('####')[-1].strip().replace(',','')
            if len(preds) > 0:
                result = int(preds[0])
            else:
                result = None
            correct = False
            if int(target) == result:
                correct = True
            meter.update(correct)
            # breakpoint()
            tqdm_bar.set_description(f'ACC {meter.acc:.4f} ({meter.num_correct}/{meter.num_total})')
            extra_data.update(dict(
                golf=entry['answer'],
                question=entry['question'],
                target=str(target),
                result=str(result),
                correct=correct,
                # prompt=entry['prompt'],
                # completion=extra_data['gen_text']
            ))
            all_responses.append(extra_data)
    meta = {}
    if args.task == 'mmlu-pro':
        meta = dict(
            num_correct=num_correct,
            num_total=num_total,
            acc=acc
        )
    elif args.task == 'humaneval':
        from train.eval.human_eval.inference_hack import evaluation_only
        _,pass_rate = evaluation_only('python',all_responses)
        meta = dict(
            lang=lang,
            pass_at_1=pass_rate
        )
    elif args.task == 'gsm8k':
        meta = dict(
            acc=meter.acc 
        )
        
    final_data = dict(
        meta = meta,
        all_responses=all_responses
    )
    output_dir = os.path.dirname(output_name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not args.skip:
        with open(output_name,'w') as f:
            f.write(json.dumps(final_data,indent=2))
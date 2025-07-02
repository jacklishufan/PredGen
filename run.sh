


BASE_MODEL=/data1/jacklishufan/Qwen2.5-7B-Instruct 

# AR Decoding
DECODING=hf
CUDA_VISIBLE_DEVICES=0 python main.py -f --model_p $BASE_MODEL -j lmsys -n 100 --decoding $DECODING -o outputs/lmsys-sentence 
# Greedy Acc
DECODING=greedy
CUDA_VISIBLE_DEVICES=0 python main.py -f --model_p $BASE_MODEL -j lmsys -n 100 --decoding $DECODING -o outputs/lmsys-sentence
# SD 
DECODING=sd
CUDA_VISIBLE_DEVICES=0 python main.py -f --model_p $BASE_MODEL -j lmsys -n 100 --decoding $DECODING -o outputs/lmsys-sentence --topk 3
# CLLM
DECODING=sd
CUDA_VISIBLE_DEVICES=0 python main.py -f --model_p $BASE_MODEL -j lmsys -n 100 --decoding $DECODING -o outputs/lmsys-sentence --topk 3

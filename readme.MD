# PredGen: Accelerated Inference of Large Language Models through Input-Time Speculation for Real-Time Speech Interaction
[Arxiv](https://arxiv.org/abs/2506.15556)

### Reproduce Results

To reproduce main experiments

```
bash run.sh
```

To reproduce a specific setup, use the following command

```
CUDA_VISIBLE_DEVICES=0 python main.py -f --model_p [model_path] -j lmsys -n 100 --decoding [hf/greedy/model/cllm/sd] -o outputs/lmsys-sentence [--simulate]
```

If `--simulate` flag is enabled, it will load recorded inputs from `simulated_inputs.json` which are streamed transcriptions of TTS outputs with timestamps. Outherwise, it will emulate input stream by streaming words from the ground truth text.

If `--decoding` is `sd`, (i.e. speculative top-k decoding), there is an additional flag `--topk [k]` that can be passed to specify K in top-K sampling.

If `--decoding` is `cllm` enabled, please download the Lora adaptor from [Huggingface](https://huggingface.co/jacklishufan/predgen-qwen-cllm) and pass in additional argument `--lora_p [path/to/checkpoint]`
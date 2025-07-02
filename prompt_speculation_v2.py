import torch
from typing import Tuple
import time
import prettytable
import os
import random
from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
import json
try:
    with open ('simulated_inputs.json') as f:
        ALL_LOGS = json.loads(f.read())
except:
    print('simulated_inputs.json not found, using empty ALL_LOGS')
    ALL_LOGS = {}
def get_common_prefix_length(tensor1, tensor2):
    # return [0]
    # Ensure the tensors have the same shape
    if tensor1.shape != tensor2.shape:
        min_len = min(tensor1.shape[-1],tensor2.shape[-1])
        tensor1 = tensor1[...,:min_len]
        tensor2 = tensor2[...,:min_len]
        
    # Compare elements along the last axis
    match = tensor1 == tensor2
    
    # Find the first non-match index along the last axis
    # Using cumulative product to stop counting after the first False
    cumsum = match.cumprod(dim=-1)
    
    # Sum along the last axis to get the length of common prefixes
    prefix_lengths = cumsum.sum(dim=-1)
    return prefix_lengths


def get_sd_acceptance(draft,prediction,prediction_probs,k=1):
    if draft.shape != prediction.shape:
        min_len = min(draft.shape[-1],prediction.shape[-1])
        draft = draft[...,:min_len] # N X L 
        prediction = prediction[...,:min_len] # N X L
        prediction_probs = prediction_probs[:,:min_len,:] # N X L X D
    # implement top k acceptance 
    # Get the indices of the top k predictions for each timestep
    top_k_indices = torch.topk(prediction_probs, k, dim=-1).indices  # Shape: (N, L, k)

    # Compare prediction with top-k indices
    # We want to check if the predicted class is in the top k predictions
    # `prediction.unsqueeze(-1)` reshapes the prediction to match top_k_indices
    acceptance = (top_k_indices == draft.unsqueeze(-1)).any(dim=-1)
    rejection_mask = torch.cumprod(acceptance, dim=-1)  # Cumulative product along the sequence length axis
    
    # Convert the rejection_mask back to boolean
    # acceptance = rejection_mask.bool()  # After rejection propagation, convert back to boolean
    
    # Return the number of accepted tokens
    accepted_count = rejection_mask.sum(-1).item()  # Sum the accepted tokens and convert to a Python scalar
    
    return accepted_count





    
class Buffer:
    texts = []
    
def clear():
    os.system('clear')
    for row in Buffer.texts:
        print(*row)
    Buffer.texts = []
    
    
def print_buf(*args):
    Buffer.texts.append(args)
from prettytable import PrettyTable
table = PrettyTable()

def print_table(**kwargs):
    table = PrettyTable()
    table.field_names = list(kwargs.keys())
    table.add_row(list(kwargs.values()))
    return table.get_string()
    
    
def truncate_key_value(
        self,
        num_of_accepted_tokens,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :num_of_accepted_tokens, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :num_of_accepted_tokens, :]
        return self


def remove_last_word(s):
    words = s.split()  # Split the string into words
    if len(words) > 1:
        return ' '.join(words[:-1])  # Join all words except the last one
    return ''  # Return empty if there's only one word
def two_color(s,l,a,b):
    return colored(s[:l],a) + colored(s[l:],b)
    
from transformers.generation import TextStreamer
import  time
class CustomHFStreamer(TextStreamer):
    
    def __init__(self, tokenizer, skip_prompt = False,print_rows=[],n_str_accpeted=0, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.print_rows = print_rows
        self.n_str_accpeted = n_str_accpeted
        self.sentence_breaks = get_sentence_break(tokenizer)
        self.sentence_breaks_symbols = ['.','?','!']
        self.nfe = -1 # hack
        self.timestemp_first_sentence = None
        self.nfe_to_first_sentence = None
        self.timestemp_first_token = None
        
    def put(self, value):
        super().put(value)
        self.nfe += 1
        now = time.time()
        if self.nfe <=0:
            # skip first nfe for prefill
            return
        if self.timestemp_first_token  is None:
            self.timestemp_first_token = now
        #if value.item() in self.sentence_breaks and self.timestemp_first_sentence  is None:
        text = self.tokenizer.decode([value.item()])
        if any([x in text for x in self.sentence_breaks_symbols]) and self.timestemp_first_sentence  is None: 
            self.nfe_to_first_sentence = self.nfe
            self.timestemp_first_sentence  = now
    def on_finalized_text(self, text: str, stream_end: bool = False):
        pass
        # print(text, flush=True, end="" if not stream_end else None)
        # new_rows = []
        # new_rows.append('---------New Generation------------')
        # new_rows.append(two_color(text,self.n_str_accpeted,'blue','red'))
        # clear()
        # print('\n'.join(self.print_rows))
        # print('\n'.join(new_rows))
        #print(text, flush=True, end="" if not stream_end else None)

def _split_into_sentence_end_positions(ids, break_set):
    """
    Given a 1D tensor (or list) of token IDs and a set of break token IDs,
    return a list of *positions* (end indices) where any break token appears.

    Example:
      ids = [101, 2023, 13, 999, 2024, 13, 102]
      break_set = {13,999}
      returns [2, 3, 5]  # positions (0-based) where break tokens appear
    """
    end_positions = []
    for i, token_id in enumerate(ids):
        if token_id in break_set:
            end_positions.append(i)
    return end_positions

####################################
# 2. Build Qwen-Style Query
####################################
def build_query_for_acceptance_qwen(partial_prompt, partial_answer):
    """
    For Qwen, you typically have the format:

    <|im_start|>system
    You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
    <|im_end|>
    <|im_start|>user
    You are given an incomplete prompt ...
    Partial Prompt: ...
    Partial Answer: ...
    <|im_end|>
    <|im_start|>assistant
    The answer is
    """
    system_block = (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
        "<|im_end|>\n"
    )
    user_block = (
        "<|im_start|>user\n"
        "You are given an incomplete prompt and model's speculative partial answer. "
        "Please judge if the partial prompt is consistent with the model's answer.\n"
        f"Partial Prompt: {partial_prompt}\n"
        f"Partial Answer: {partial_answer}\n"
        "<|im_end|>\n"
    )
    assistant_block = "<|im_start|>assistant\nThe answer is"  # We'll handle yes/no after this
    return system_block + user_block + assistant_block

####################################
# 3. Compute Probability(yes) vs. Probability(no)
####################################
def _compute_yes_no_probabilities(model, tokenizer, batch_input_ids, device="cuda"):
    """
    Given a *batch* of tokenized queries (each ends with 'The answer is'),
    compute the probability of the next token being 'yes' vs. 'no'.
    
    We do something simple: get the logits at the final position for each query
    and compare logits for the "yes" token ID vs. the "no" token ID.

    If your model doesn't split "yes"/"no" into single tokens, you'll need to
    measure the total log-likelihood of multiple tokens (e.g. "yes" -> "y", "##es").
    For demonstration, we assume "yes" and "no" are single tokens in the vocabulary.
    """

    # Suppose your tokenizer has single IDs for "yes" and "no". If not, you'd define them differently.
    yes_id = tokenizer.encode("yes", add_special_tokens=False)
    no_id = tokenizer.encode("no", add_special_tokens=False)

    # If "yes" or "no" are more than one token, you'd compute sum of log-probs over the next few tokens.
    # For simplicity, let's assume each is 1 token:
    assert len(yes_id) == 1, "For demonstration, we assume 'yes' is 1 token"
    assert len(no_id) == 1, "For demonstration, we assume 'no' is 1 token"
    yes_id = yes_id[0]
    no_id = no_id[0]

    # Pad to same length
    max_len = max(len(x) for x in batch_input_ids)
    input_ids_padded = []
    attention_masks = []
    for seq in batch_input_ids:
        pad_len = max_len - len(seq)
        input_ids_padded.append(seq + [tokenizer.pad_token_id]*pad_len)
        attn_mask = [1]*len(seq) + [0]*pad_len
        attention_masks.append(attn_mask)

    input_ids_padded = torch.tensor(input_ids_padded, dtype=torch.long, device=device)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long, device=device)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_padded,
            attention_mask=attention_masks
        )
    # logits: [batch_size, seq_len, vocab_size]
    logits = outputs.logits

    # We want the last non-padding position for each example
    # That is: len(seq)-1 in each example's row
    yes_probs = []
    no_probs = []

    for i, seq in enumerate(batch_input_ids):
        last_idx = len(seq) - 1  # the final token's position
        last_logit = logits[i, last_idx, :]  # [vocab_size]
        # Probability distribution for the *next* token
        next_token_logit = last_logit  # Because HF models do SHIFT to the right
        next_token_probs = torch.log_softmax(next_token_logit, dim=-1)

        yes_probs.append(next_token_probs[yes_id].item())
        no_probs.append(next_token_probs[no_id].item())

    # Return parallel lists of log-probs
    return yes_probs, no_probs

####################################
# 4. Main acceptance logic for "model_query"
####################################

def build_flex_mask_model_query_qwen(
    system_text,
    user_text,             # includes partial prompt
    partial_answer_ids,    # the entire partial answer as a list of token IDs
    sentence_end_positions, 
    tokenizer
):
    """
    Build a single flattened input that includes:
      - [SYSTEM BLOCK]
      - [USER BLOCK with partial prompt + entire partial answer as text]
      - K "assistant blocks", each says "The answer is"

    Then build a custom `attention_mask` so that block #k can only see the user block
    plus partial_answer_ids[: end_of_sentence_k], not beyond.

    Returns:
      input_ids [1 x T]  (or just a 1D list)
      attention_mask [1 x T x T] that enforces the blockwise restriction
      block_positions: list of positions in the flatten seq (one for each block)
                       where "The answer is" block ends (we’ll pick next logit).
    """
    # 1. Turn the system+user text into IDs
    system_ids = tokenizer.encode(system_text, add_special_tokens=False)
    user_ids   = tokenizer.encode(user_text, add_special_tokens=False)

    # 2. For each sentence, we create an assistant block: "<|im_start|>assistant\nThe answer is"
    #    We store the positions after "The answer is" to measure next token log-probs.
    assistant_prefix_text = "<|im_start|>assistant\nThe answer is"
    assistant_prefix_ids  = tokenizer.encode(assistant_prefix_text, add_special_tokens=False)

    # We'll build the flatten sequence in parts
    # Part A: system + user (this is "shared")
    flatten_ids = system_ids + user_ids
    # We'll remember the boundary: how many tokens so far
    shared_end = len(flatten_ids)

    block_positions = []
    # Then for each sentence chunk, we replicate the assistant prefix
    for i, end_pos in enumerate(sentence_end_positions):
        # We want the partial answer up to this end_pos to be "visible" in attention
        # But we won't literally "append" that partial answer again as text,
        # because the user block *already contains the entire partial_answer_ids*.
        # Instead, we rely on the attention mask to hide tokens beyond end_pos from this block.

        # So just add the assistant prefix
        block_start = len(flatten_ids)
        flatten_ids += assistant_prefix_ids
        block_end = len(flatten_ids)
        block_positions.append(block_end - 1)  # the last token is "is"

    # Convert to a single 1D tensor for input_ids
    input_ids = torch.tensor(flatten_ids, dtype=torch.long).unsqueeze(0)  # [1, T]

    # 3. Build the attention mask [1, T, T]
    #    We'll do a lower-triangular by default, then apply additional masking.
    T = input_ids.size(1)
    base_mask = torch.ones(T, T, dtype=torch.bool)
    base_mask = torch.tril(base_mask)  # causal from left to right

    # Next, for each block #k, we only want it to see:
    #  - All tokens up to 'shared_end' (the entire system+user context).
    #  - Among partial_answer_ids, only up to end_pos for that sentence.
    #  - Its own assistant prefix tokens.
    #
    # So we need to figure out the range of partial_answer_ids within user_ids.
    # The user block likely ends in "... Partial Prompt: ... Partial Answer: <the entire partial answer text> ... <|im_end|>"
    # If your user text is exactly "You are given an incomplete prompt... Partial Prompt: <X> Partial Answer: <the ENTIRE partial answer>", 
    # we need to find where partial_answer_ids starts inside user_ids. 
    # For simplicity, let's assume that the entire partial_answer_ids is appended at the end of user_text.

    # We'll attempt a naive approach:
    partial_answer_start = len(system_ids) + (len(user_ids) - len(partial_answer_ids))
    # partial_answer_end = partial_answer_start + len(partial_answer_ids) - 1

    # We also track each block's (start, end) in flatten_ids. We'll reconstruct them:
    blocks_ranges = []
    idx = shared_end
    for i in range(len(sentence_end_positions)):
        block_len = len(assistant_prefix_ids)
        blocks_ranges.append( (idx, idx + block_len - 1) )  # inclusive
        idx += block_len

    # Create a final mask we can fill in
    attn_mask = base_mask.clone()

    # For each block i, allow attention from:
    #   [0 : shared_end-1]  (the system+user block) 
    # plus
    #   partial_answer_start : partial_answer_start + sentence_end_positions[i] + 1
    # plus
    #   the block itself

    for i, end_pos in enumerate(sentence_end_positions):
        block_start_i, block_end_i = blocks_ranges[i]  # inclusive range for that block
        # This block can attend from: 
        #   0..(partial_answer_start + end_pos)  plus 
        #   block_start_i..block_end_i
        limit = partial_answer_start + end_pos + 1  # +1 because end_pos is inclusive

        # Mask out anything beyond 'limit' in partial answer region for tokens in block i
        # The row indices we want to fix are block_start_i..block_end_i (the tokens in the block).
        for pos in range(block_start_i, block_end_i + 1):
            # Allowed to attend up to 'limit - 1' (the token index is 0-based).
            for col in range(limit, T):
                if col < block_start_i or col > block_end_i:
                    attn_mask[pos, col] = False

        # Also, the tokens in block i shouldn't attend to block i+1 or i+2, etc.
        # Our base_mask is lower-tri, so that might already handle it,
        # but let's be explicit: any columns after block_end_i are set to False.
        for pos in range(block_start_i, block_end_i + 1):
            for col in range(block_end_i + 1, T):
                attn_mask[pos, col] = False

    # We'll keep the mask as 0/1 rather than bool for HF. 
    # Typically, HF expects 0 = no attend, 1 = attend.
    attn_mask = attn_mask.long().unsqueeze(0)  # [1, T, T]

    return input_ids, attn_mask, block_positions


def compute_yes_no_probabilities_flex(
    model, 
    tokenizer, 
    input_ids, 
    attention_mask, 
    block_positions, 
    device='cuda'
):
    """
    1) Do a single forward pass with (input_ids, attention_mask).
    2) For each block i, get the token's logits at block_positions[i].
       That is the logit for the NEXT token after "The answer is".
    3) Compare log-prob("yes") vs. log-prob("no").
    """
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    yes_id = tokenizer.encode("yes", add_special_tokens=False)
    no_id  = tokenizer.encode("no", add_special_tokens=False)
    # For simplicity, assume these are single tokens
    yes_id, no_id = yes_id[0], no_id[0]

    with torch.no_grad():
        position_ids = attention_mask.sum(-1) - 1  # last non-padding position
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,position_ids=position_ids)
    logits = outputs.logits  # [1, T, V]

    # We want the distribution *after* block_positions[i]. 
    # In standard HF, the logits at index X is the score for the token *generated at position X*,
    # so we actually look at logits[0, block_positions[i], :].
    # That distribution is for the token that occurs at block_positions[i].
    # If you want the next token (depending on how your model shifts), 
    # often you look at logits[0, block_positions[i]-1, :] 
    # Check your model’s documentation carefully.
    #
    # Let's assume the standard GPT-like indexing:
    #  "The logits at position p represent the distribution of the *p-th generated token*."
    # So if we want the distribution for the token AFTER block_positions[i],
    # we look at logits[0, block_positions[i], :].
    yes_logps = []
    no_logps  = []
    for pos in block_positions:
        next_logit = logits[0, pos, :]  # distribution for next token
        next_probs = torch.log_softmax(next_logit, dim=-1)
        yes_logps.append(next_probs[yes_id].item())
        no_logps.append(next_probs[no_id].item())

    return yes_logps, no_logps


def flex_attention_model_query_acceptance(
    system_text,
    user_text,            # includes entire partial answer in text form
    partial_answer_ids,
    sentence_end_positions, 
    model,
    tokenizer,
    device='cuda'
):
    """
    1) Build the single flatten input + attention mask.
    2) Single forward pass.
    3) Return a list [accept/reject bool for each sentence].
    4) Then you can transform that to number of tokens accepted, etc.
    """
    input_ids, attn_mask, block_positions = build_flex_mask_model_query_qwen(
        system_text=system_text,
        user_text=user_text,
        partial_answer_ids=partial_answer_ids,
        sentence_end_positions=sentence_end_positions,
        tokenizer=tokenizer
    )

    yes_logps, no_logps = compute_yes_no_probabilities_flex(
        model=model, 
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attn_mask,
        block_positions=block_positions,
        device=device
    )

    # Decide acceptance from left to right
    accepted_sentence_count = 0
    for i, (ylp, nlp) in enumerate(zip(yes_logps, no_logps)):
        if ylp >= nlp:
            accepted_sentence_count += 1
        else:
            break

    return accepted_sentence_count


def _get_num_accepted_tokens_model_query(
    partial_prompt_str,
    prev_generation_ids,
    tokenizer,
    model,
    device="cuda"
):
    """
    1. Identify all sentence breaks in prev_generation_ids.
    2. For each sentence, build a batch input that ends with 'The answer is' and
       measure P(yes) vs. P(no).
    3. Accept consecutive sentences from the start until the model says 'no'.
    4. Return the number of tokens accepted.
    """
    # 1. Find sentence boundaries
    break_set = get_sentence_break(tokenizer)
    sentence_end_positions = _split_into_sentence_end_positions(prev_generation_ids, break_set)

    # If there are no breaks, treat the entire sequence as one "sentence"
    if len(sentence_end_positions) == 0:
        sentence_end_positions = [len(prev_generation_ids)-1]
    if len(sentence_end_positions) >= 4:
        sentence_end_positions = sentence_end_positions[:4]
    # 2. Build the batch: we want to check each prefix up to the k-th break
    # For example, if we have 3 breaks at positions [5, 10, 15], we will build 3 queries:
    #  prefix up to index 5, prefix up to index 10, prefix up to index 15
    batch_input_ids = []
    for end_pos in sentence_end_positions:
        # partial answer up to (and including) end_pos
        truncated_answer_ids = prev_generation_ids[: (end_pos+1)]
        truncated_answer_str = tokenizer.decode(truncated_answer_ids, skip_special_tokens=True)

        # Build Qwen prompt
        query_str = build_query_for_acceptance_qwen(partial_prompt_str, truncated_answer_str)
        
        query_ids = tokenizer.encode(query_str, add_special_tokens=False)
        batch_input_ids.append(query_ids)

    # 3. Single forward pass in batch: get P(yes), P(no) for each
    yes_logps, no_logps = _compute_yes_no_probabilities(
        model=model,
        tokenizer=tokenizer,
        batch_input_ids=batch_input_ids,
        device=device
    )

    # 4. Accept or reject each sentence from left to right
    # We'll keep a pointer to the "last accepted token index"
    last_accepted_token_idx = -1
    for (end_pos, ylp, nlp) in zip(sentence_end_positions, yes_logps, no_logps):
        # Decide accept or reject
        # "Accept" if p(yes) > p(no), i.e. logp(yes) > logp(no)
        #   (You may want a ratio or threshold approach instead)
        if ylp >= nlp:
            # accept this chunk
            last_accepted_token_idx = end_pos
        else:
            # reject => break
            break

    # If last_accepted_token_idx == -1, that means we accepted none
    num_accepted = last_accepted_token_idx + 1  # +1 because index-based
    return num_accepted

####################################
# 5. Integrate into your speculative_step
####################################
def try_to_to_tensor(x):
    if isinstance(x,torch.Tensor):
        x = x.item()
    return x

def find_sentence_ends(input_ids, break_set):
    """
    Return list of positions i in input_ids where input_ids[i] is in break_set.
    """
    return [i for i,tok in enumerate(input_ids) if tok in break_set]
torch.inference_mode()
def jacobi_forward_profiling(
    self,
    input_ids: torch.LongTensor = None,
    past_key_values= None,
    use_cache = None,
    max_new_tokens= None,
    prefill_phase= False,
    check_ttfs = False,
    sentence_breaks = []
    # temperature: float = 1,
    # top_p:float = 0,
):
    
    assert use_cache == True

    if input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    
    if prefill_phase: # prefill phase, just compute the keys & values of prompt
        y = self(input_ids=input_ids,past_key_values=past_key_values,use_cache=use_cache)
        predict_next_tokens = torch.argmax(torch.nn.functional.softmax(y.logits, dim=-1), dim=-1)
        first_correct_token = predict_next_tokens[:, -1]
        return y.past_key_values, first_correct_token
    else: # generation phase, input as random_initilized point and output as fixed point
        jacobian_trajectory = []
        accurate_n_gram = torch.zeros_like(input_ids).to(input_ids.device)
        accurate_length = 0
        next_point = input_ids
        jacobian_trajectory.append(next_point)

        iter_counter = 0
        if type(past_key_values) != tuple:
            base_cache = past_key_values.to_legacy_cache()
        else:
            base_cache = past_key_values
        i = torch.tensor([1]).long().cuda()
        timesteps = 1000
        n_converged = 1
        ttfs = None
        nfe_to_fs = None
        while True:
            past_key_values = base_cache
            current_point = next_point
            torch.cuda.synchronize()
            t00 = time.time()
            #timesteps = i  #* 1 / max_new_tokens * 1000
            # breakpoint()
            y = self(input_ids=current_point,past_key_values=past_key_values,use_cache=True,block_size=max_new_tokens,timesteps=timesteps,temb_factor=1.0
                     )
            # breakpoint()
            i += 1        
            iter_counter += 1        
            # probs = (y.logits/temperature).softmax(-1)[0, n_converged-1:seq_length-1] # unconverged # 
            # indices_to_verify = current_point[0, n_converged:].view(1,-1)
            # xx = torch.arange(indices_to_verify.shape[-1]).view(1,-1)
            # bb = 0
            # probs_preious = probs[xx,indices_to_verify][0].view(-1,1)
            # cum_prob_so_far = ((probs>probs_preious) * probs_preious).sum(-1) 
            # has_converged = cum_prob_so_far <= top_p
            # converged_delta = torch.cumprod(has_converged,-1).sum().long().item()
            # #probs[indices_to_verify]
            # n_converged += converged_delta
            # un_converged = seq_length-n_converged 
            # #probs = probs[converged_delta:]
            # if top_p == 0 or True:
            #     output_ids = y.logits.argmax(-1)[0, n_converged-1:seq_length-1].view(1,-1) # self.lm_head(y.last_hidden_state).argmax(-1)
            # else:
            #     # broken 
            #     probs_sorted,probs_indices = probs.sort(-1,descending=True)
            #     probs_sorted_cum = probs_sorted.cumsum(-1)
                
            #     probs_sorted_cum = torch.cat(
            #         [torch.zeros(probs_sorted_cum.shape[0],1).to(probs_sorted_cum),probs_sorted_cum],dim=-1
            #     )[...,:-1]
            #     probs[(probs_sorted_cum > top_p)] = 0
            #     _probs = probs / probs.sum(-1,keepdims=True)
            #     output_ids = torch.multinomial(_probs,1).view(1,-1)
            #     output_ids = output_ids[...,converged_delta:]
            output_ids = y.logits.argmax(-1)
            next_point= torch.cat((current_point[0, 0].view(1,-1), output_ids[0, :seq_length-1].view(1,-1)), dim=-1)
            #next_point= torch.cat((current_point[0, :n_converged].view(1,-1), output_ids), dim=-1)
            # breakpoint()
            prev_converged = n_converged-1
            n_converged = torch.eq(current_point, next_point).cumprod(-1).sum()
            if check_ttfs:
                newly_converged = current_point[0,prev_converged:n_converged].tolist()
                for token in newly_converged:
                    if token in sentence_breaks:
                        ttfs = time.time()
                        nfe_to_fs = iter_counter
                        check_ttfs = False
                        break
            un_converged = seq_length-n_converged 
            # breakpoint()
            torch.cuda.synchronize()
            t1 = time.time()
            torch.cuda.synchronize()
            t2 = time.time()
            # converged_token = n_converged()
            # MAYBE ADD IF BLOCK
            # convergence_rate = torch.eq(current_point, next_point).sum() / current_point.numel()
            # timesteps = (1 -convergence_rate.item()) * 1000
            # timesteps = int(timesteps)
            if un_converged <= 0 or iter_counter >=max_new_tokens:    
            #if iter_counter == 50:
                #print('Successfully break!')
                #print(next_point)
                past_key_values = base_cache
                #y = self(input_ids=current_point,past_key_values=past_key_values,use_cache=True,block_size=max_new_tokens)
                first_correct_token = y.logits[:,-1].argmax(-1) #output_ids[:,-1]
                if check_ttfs and first_correct_token.item() in sentence_breaks:
                    ttfs = time.time()
                    nfe_to_fs = iter_counter
                    check_ttfs = False
                break
            #breakpoint()
            past_key_values = base_cache
            #print(base_cache[0][0].shape)
            #delete_false_key_value(past_key_values,seq_length)
            torch.cuda.synchronize()
            t3 = time.time()
            #print(t3-t00)
            #print(t1-t0,t2-t1,t3-t2,t0-t00)
            

        return jacobian_trajectory[:-1], next_point, first_correct_token, iter_counter,y.past_key_values,ttfs,nfe_to_fs


def speculative_step(
    prev_ids,
    new_ids,
    prev_generation,
    model,
    tokenizer,
    past_key_values,
    max_new_token=512,
    prompt_text='',
    sentence_breaks=[],
    verbose=False,
    acceptance='greedy',
    generation_mode='ar',
    top_k=10,
    device='cuda'
):
    # breakpoint()
    """
    Modified to add acceptance='model_query' logic.

    For acceptance='model_query':
      1) We'll call _get_num_accepted_tokens_model_query(...) on prev_generation
      2) Then we do the normal prefix acceptance from the code
      3) The final accepted tokens is the min of "model_query" acceptance vs. prefix acceptance
    """

    # Move your data to the same device
    prev_generation = prev_generation.to(device)
    new_ids = new_ids.to(device)

    # Standard logic from your original code ...
    new_generation = torch.cat([new_ids, prev_generation], dim=-1)
    raw_new_generation = new_generation

    comm_prefix_len = get_common_prefix_length(new_ids, prev_ids)[0]  # N
    if past_key_values is not None:
        if type(past_key_values) == tuple:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values = truncate_key_value(past_key_values, comm_prefix_len)
        new_generation = new_generation[:, comm_prefix_len:]

    remaining_prompt_len = new_ids.shape[-1] - comm_prefix_len

    # Forward pass
    y = model(input_ids=new_generation, past_key_values=past_key_values, use_cache=True)
    predict_next_tokens = torch.argmax(y.logits, dim=-1)  # N x L
    num_accepted_tokens_mq = -1
    if remaining_prompt_len > 0:
        predict_next_tokens_ans = predict_next_tokens[:, remaining_prompt_len-1:]
    else:
        # Edge case if remaining_prompt_len == 0
        predict_next_tokens_ans = torch.cat([prev_generation[:, :1], predict_next_tokens], dim=-1)

    if remaining_prompt_len == 0:
        #  accept all_tokens
        num_accepted_tokens_prefix = min(predict_next_tokens_ans.shape[-1], prev_generation.shape[-1])
        num_accepted_tokens_model_query = num_accepted_tokens_prefix  # no change
    if acceptance == 'greedy':
        # Original prefix acceptance
        num_accepted_tokens_prefix = get_common_prefix_length(predict_next_tokens_ans, prev_generation)[0]  # N
        num_accepted_tokens_model_query = num_accepted_tokens_prefix  # no change
    elif acceptance == 'topk':
        # Your topk acceptance
        # predict_next_tokens_probs = torch.softmax(y.logits[:, remaining_prompt_len-1:], dim=-1)  # N x L x vocab
        predict_next_tokens_probs = y.logits[:,remaining_prompt_len-1:]
        num_accepted_tokens_prefix = get_sd_acceptance(
            prev_generation, predict_next_tokens_ans, predict_next_tokens_probs, k=top_k
        )
        num_accepted_tokens_model_query = num_accepted_tokens_prefix
    elif acceptance == 'model_query':
        # ================================
        # NEW LOGIC
        # ================================
        # 1) Standard prefix acceptance
        num_accepted_tokens_prefix = get_common_prefix_length(
            predict_next_tokens_ans, prev_generation
        )[0]

        # 2) Model-query acceptance
        # We'll decode the entire prev_generation to check if it has multiple sentences
        # Because shape is (1, L), we do [0] to get the list of IDs
        prev_generation_ids = prev_generation[0].tolist()
        # We'll pass in the *prompt_text* as partial prompt
        partial_prompt_str = prompt_text  # or maybe decode(new_ids) if you want?

        num_accepted_tokens_mq = _get_num_accepted_tokens_model_query(
            partial_prompt_str=partial_prompt_str,
            prev_generation_ids=prev_generation_ids,
            tokenizer=tokenizer,
            model=model,
            device=device
        )

        # final is the min of the two acceptances:
        # "If the prefix check is bigger than the model query acceptance, we only accept up to model query acceptance."
        # "If the prefix check is smaller, we accept that."
        num_accepted_tokens_model_query = max(num_accepted_tokens_prefix.item(), num_accepted_tokens_mq)
    elif   acceptance == 'model_query_flex':
        # =================================================
        # NEW single-pass "flex attention" approach
        # =================================================
        # 1) Identify sentence breaks in prev_generation
        num_accepted_tokens_prefix = get_common_prefix_length(
            predict_next_tokens_ans, prev_generation
        )[0]
        break_set = get_sentence_break(tokenizer)
        prev_gen_ids = prev_generation[0].tolist()
        sentence_ends = find_sentence_ends(prev_gen_ids, break_set)
        # If no breaks, treat entire thing as one sentence
        if len(sentence_ends) == 0:
            sentence_ends = [len(prev_gen_ids)-1]

        # 2) Build the system + user text for Qwen
        system_text = (
            "<|im_start|>system\n"
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
            "<|im_end|>\n"
        )
        # user_text might contain the entire partial prompt plus the entire partial answer as text
        # For example:
        user_text = (
            f"<|im_start|>user\n"
            "You are given an incomplete prompt and model's speculative partial answer. "
            "Please judge if the partial prompt is consistent.\n"
            f"Partial Prompt: {prompt_text}\n"
            f"Partial Answer: {tokenizer.decode(prev_gen_ids, skip_special_tokens=True)}\n"
            "<|im_end|>\n"
        )

        # 3) Single forward pass with flex attention
        accepted_sentence_count = flex_attention_model_query_acceptance(
            system_text=system_text,
            user_text=user_text,
            partial_answer_ids=prev_gen_ids,
            sentence_end_positions=sentence_ends,
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        # If we accepted `k` sentences, that means we accept up to sentence_ends[k-1]
        if accepted_sentence_count == 0:
            mq_accepted_tokens = 0
        else:
            last_accepted_end_pos = sentence_ends[accepted_sentence_count - 1]
            mq_accepted_tokens = last_accepted_end_pos + 1

        # 4) Combine with prefix acceptance
        final_accepted_tokens = min(try_to_to_tensor(num_accepted_tokens_prefix), mq_accepted_tokens)
        num_accepted_tokens_model_query = final_accepted_tokens

    else:
        raise NotImplementedError(f"Unknown acceptance={acceptance}")

    num_accepted_tokens = num_accepted_tokens_model_query  # The final decision
    remaining_tokens = prev_generation.shape[-1] - num_accepted_tokens

    # Now proceed with the original code’s logic to produce the next token
    first_correct_token = predict_next_tokens_ans[:, num_accepted_tokens : num_accepted_tokens+1]

    final_ids = torch.cat([
        new_ids,
        prev_generation[:, :num_accepted_tokens],
        first_correct_token
    ], dim=-1)

    nfe = 1
    extra_data = {}
    sentence_breaks_symbols = ['.','?','!']
    if remaining_tokens == 0 and first_correct_token[0][-1] == tokenizer.eos_token_id:
        # If we've consumed everything + got an eos
        generation = final_ids
        past_key_values = truncate_key_value(y.past_key_values, new_ids.shape[-1] + num_accepted_tokens)
        timestampe_first_stentence = None
        nfe_to_first_sentence = nfe
    else:
        past_key_values = truncate_key_value(y.past_key_values, new_ids.shape[-1] + num_accepted_tokens)
        timestampe_first_stentence = None
        nfe_to_first_sentence = None
        accepted_tokens = raw_new_generation[:,new_ids.shape[-1]:new_ids.shape[-1]+num_accepted_tokens+1]
        # if any([x in accepted_tokens for x in sentence_breaks ]):
        #     timestampe_first_stentence = time.time()
        #     nfe_to_first_sentence = nfe
        _text = tokenizer.decode(accepted_tokens[0])
        if any([x in _text for x in sentence_breaks_symbols ]):
            timestampe_first_stentence = time.time()
            nfe_to_first_sentence = nfe
        if verbose:
            # If you want to do the token-by-token stepping for the next max_new_token
            # the original code had a debug loop ...
            accepted_ans = tokenizer.batch_decode(accepted_tokens)
            n_str_accpeted = len(accepted_ans[0])
            
            all_tokens = predict_next_tokens_ans[:, :num_accepted_tokens+1][0].tolist()
            for _ in range(max_new_token):
                y2 = model(input_ids=first_correct_token, use_cache=True, past_key_values=past_key_values)
                nfe += 1
                do_break = 0
                next_token = torch.argmax(y2.logits[:, -1:], dim=-1).item()
                _text = tokenizer.decode([next_token])
                # if next_token in sentence_breaks and timestampe_first_stentence is None:
                #     timestampe_first_stentence = time.time()
                #     nfe_to_first_sentence = nfe
                #print(_text)
                if any([x in _text for x in sentence_breaks_symbols ]) and timestampe_first_stentence is None:
                    timestampe_first_stentence = time.time()
                    nfe_to_first_sentence = nfe
                if next_token == tokenizer.eos_token_id:
                    do_break = 1
                else:
                    all_tokens.append(next_token)
                    first_correct_token[0] = next_token
                if do_break or next_token in sentence_breaks:
                    generation = torch.tensor(all_tokens)[None]
                    print_rows = []
                    print_rows.append(colored("Current Input:"+prompt_text,"green"))
                    print_rows.append(colored('-----------------------------------',"green"))
                    if acceptance in ['model_query','model_query_flex']:
                        num_accepted_tokens = try_to_to_tensor(num_accepted_tokens)
                        remaining_tokens = try_to_to_tensor(remaining_tokens)
                        model_acc = try_to_to_tensor(num_accepted_tokens_mq)
                        prefix_acc = try_to_to_tensor(num_accepted_tokens_prefix)
                        acc_delta = -1 if model_acc == -1 else model_acc - prefix_acc
                        print_rows.append(print_table(Prefix=comm_prefix_len.item(),
                                    New__Input=remaining_prompt_len.item(),
                                    Accepted=num_accepted_tokens,Rejected=remaining_tokens,
                                    Model_Acc=model_acc,
                                    Prefix_ACc=prefix_acc,
                                    Acc_Delta=acc_delta
                                    ))
                    else:
                        print_rows.append(print_table(Prefix=comm_prefix_len.item(),
                                    New__Input=remaining_prompt_len.item(),
                                    Accepted=num_accepted_tokens.item(),Rejected=remaining_tokens.item()))
                    print_rows.append('---------prev------------')
                    print_rows.append(two_color(tokenizer.batch_decode(prev_generation)[0],n_str_accpeted,'blue','red'))
                    print_rows.append('---------ArgMax of Prev Gen------------')
                    print_rows.append(two_color(tokenizer.batch_decode(predict_next_tokens_ans)[0],n_str_accpeted,'blue','red'))
                    if generation is not None:
                        print_rows.append('---------New Generation------------')
                        print_rows.append(two_color(tokenizer.batch_decode(generation)[0],n_str_accpeted,'blue','red'))
                    clear()
                    print('\n'.join(print_rows))
                if do_break:
                    break
            generation = torch.tensor(all_tokens)[None]
        else:
            # If you're not printing intermediate progress
            all_tokens = predict_next_tokens_ans[:, :num_accepted_tokens+1][0].tolist()
            if generation_mode == 'ar':
                for _ in range(max_new_token):
                    y2 = model(input_ids=first_correct_token, use_cache=True, past_key_values=past_key_values)
                    nfe += 1
                    next_token = torch.argmax(y2.logits[:, -1:], dim=-1).item()
                    _text = tokenizer.decode([next_token])
                    
                    # if next_token in sentence_breaks and timestampe_first_stentence is None:
                    #     timestampe_first_stentence = time.time()
                    #     nfe_to_first_sentence = nfe
                    if any([x in _text for x in sentence_breaks_symbols ]) and timestampe_first_stentence is None:
                        timestampe_first_stentence = time.time()
                        nfe_to_first_sentence = nfe
                    if next_token == tokenizer.eos_token_id:
                        break
                    all_tokens.append(next_token)
                    first_correct_token[0] = next_token
            elif generation_mode == 'jacobi':
                all_tokens = predict_next_tokens_ans[:, :num_accepted_tokens][0].tolist()
                blk_size = 10
                new_id_lists =  new_ids[0].tolist()
                all_tokens =   new_id_lists + all_tokens
                generated_tokens = 0
                while True:
                    if generated_tokens > max_new_token:
                        break
                    #random_choices = all_tokens +
                    # print(generated_tokens)
                    random_point = torch.tensor(random.choices(all_tokens, k=(blk_size-1)), device=model.device).view(1,-1)
                    input_ids = torch.cat((first_correct_token.view(1,-1), random_point),dim=-1)
                    if timestampe_first_stentence is None and len(sentence_breaks) >0:
                        check_ttfs = True
                    else:
                        check_ttfs = False
                    jacobian_trajectory, n_gram_generation, first_correct_token, iter_steps,past_key_values,ttfs_local,nfe_to_fs_local = jacobi_forward_profiling(model,input_ids=input_ids, max_new_tokens=blk_size, past_key_values=past_key_values, use_cache = True, prefill_phase = False,
                                                                                                                          check_ttfs = True,
                                                                                                                          sentence_breaks=sentence_breaks
                                                                                                                          )
                    if check_ttfs and ttfs_local is not None:
                        timestampe_first_stentence = ttfs_local
                        nfe_to_first_sentence = nfe + nfe_to_fs_local
                    nfe += iter_steps 
                    # print(iter_steps)
                    new_tokens = n_gram_generation#[0].tolist()
                    # if timestampe_first_stentence is None and len(sentence_breaks) >0:
                    #     for next_token in n_gram_generation[0].tolist():
                    #         if next_token in sentence_breaks:
                    #             timestampe_first_stentence = time.time()
                    #             nfe_to_first_sentence = nfe
                    #             break
                    generated_tokens += blk_size
                    eos_positions = torch.where(n_gram_generation[0]==tokenizer.eos_token_id)[0]
                    if len(eos_positions)>0:
                        eos_reached = True
                        first_eos = eos_positions[0]
                        all_tokens.extend(n_gram_generation[0,:first_eos+1].tolist())
                        break
                    all_tokens.extend(n_gram_generation[0].tolist())
                
                    # print(generated_tokens)
                all_tokens = all_tokens[len(new_id_lists):]
                # print(tokenizer.decode(all_tokens))
            else:
                # breakpoint()
                raise NotImplementedError
            generation = torch.tensor(all_tokens)[None]

    extra_data.update(
        dict(
            num_accepted_tokens=try_to_to_tensor(num_accepted_tokens),
            nfe=nfe,
            nfe_to_first_sentence=nfe_to_first_sentence
        )
    )

    return generation, past_key_values, timestampe_first_stentence, extra_data
@torch.no_grad()
def baseline_generate(new_ids,model,tokenizer,past_key_values,max_new_token):
    # generation = model.generate(
    #     generation,
    #     do_sample=False,
    #     max_new_tokens=max_new_token,
    #     use_cache=True,
    #     repetition_penalty=1,
    #     past_key_values=past_key_values,
    # ) 
    sentence_breaks = get_sentence_break(tokenizer)
    all_tokens_prefix=new_ids[0].tolist()
    all_tokens = []
    first_correct_token = new_ids
    timestampe_first_stentence = None
    nfe = 0
    nfe_to_first_sentence = None
    for _ in range(max_new_token):
        y2 = model(input_ids=first_correct_token,use_cache=True,past_key_values=past_key_values)
        nfe += 1
        first_correct_token = torch.argmax(y2.logits[:,-1:], dim=-1)
        next_token = first_correct_token.item()
        if next_token in sentence_breaks and timestampe_first_stentence is None:
            timestampe_first_stentence = time.time()
            nfe_to_first_sentence = nfe
        if next_token == tokenizer.eos_token_id:
            break
        all_tokens.append(next_token)
    # if nfe_to_first_sentence is None:
    #     nfe_to_first_sentence = nfe
    generation = torch.tensor(all_tokens_prefix+all_tokens)[None]
    #print(tokenizer.batch_decode(generation[:,new_ids.shape[-1]:]))
    return generation[:,new_ids.shape[-1]:],past_key_values,timestampe_first_stentence,nfe_to_first_sentence
    
from termcolor import colored

# print(colored('hello', 'red'), colored('world', 'green'))
class InputTextStreamer:
    
    def __init__(self,prompt,tokenizer,speed=240,preprocessor=lambda x:x,final_text=''):
        self.speed = speed / 60
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.dt = 0
        self.max_time = len(self.prompt) / self.speed
        self.final_text = final_text
        
    def get_prompt(self,device='cuda'):
        num_characters =int(self.speed * self.dt) 
        current_prompt = self.prompt[:num_characters]
        if not self.is_done():
            current_prompt = remove_last_word(current_prompt)
        else:
            current_prompt = current_prompt + self.final_text
        current_prompt_processed = self.preprocessor(current_prompt)
        return self.tokenizer([current_prompt_processed],return_tensors='pt',add_special_tokens=False).input_ids.to(device),current_prompt
    
    def advance(self,dt):
        self.dt += dt
        
    def is_done(self):
        return self.dt > self.max_time
    
    @property
    def latency(self):
        return self.dt - self.max_time
    
class SimulatedTextStreamer:
    
    def __init__(self,logs,tokenizer,speed=240,preprocessor=lambda x:x,final_text=''):
        self.speed = speed / 60
        assert len(logs) > 0
        self.logs = logs # logs is in [(text,timestep),...]
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.dt = 0
        self.offset = 0.1 # 0.1 s warmup
        self.max_time = logs[-1][1] + self.offset #len(self.prompt) / self.speed
        self.final_text = logs[-1][0]
        
    def get_prompt(self,device='cuda'):
        #num_characters =int(self.speed * self.dt) 
        if self.dt < self.offset:
            current_prompt = self.logs[0][0]
        elif self.dt >= self.max_time:
            current_prompt = self.final_text
        else:
            dt_compute = self.dt - self.offset
            seen_logs = list([x for x in self.logs if x[1] <= dt_compute])
            current_prompt = seen_logs[-1][0]
            
        if not self.is_done():
            current_prompt = remove_last_word(current_prompt)
        else:
            current_prompt = current_prompt + self.final_text
        current_prompt_processed = self.preprocessor(current_prompt)
        return self.tokenizer([current_prompt_processed],return_tensors='pt',add_special_tokens=False).input_ids.to(device),current_prompt
    
    def advance(self,dt):
        self.dt += dt
        
    def is_done(self):
        return self.dt > self.max_time
    
    @property
    def latency(self):
        return self.dt - self.max_time
    
@torch.no_grad()
def build_lookup(input_ids,max_ngram_size):
    lookup_tables = {}
    lookup_tables['input_ids'] = input_ids
    for ngram_size in range(max_ngram_size, 0, -1):
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)
        lookup_table = {tuple(x):i for i,x in enumerate(windows[0].tolist())}
        lookup_tables[ngram_size] = lookup_table
    return lookup_tables

def get_sentence_break(tokenizer,keys = ['.','?','!']):
    candidates = list([tokenizer.encode(x,add_special_tokens=False)[0] for x in keys])
    return set(candidates)


def find_candidate_pred_tokens_with_table(promt,max_ngram_size,lookup_tables,num_pred_tokens):
    for ngram_size in range(max_ngram_size, 0, -1):
        ngram = tuple(promt[0, -ngram_size:].tolist())
        match = lookup_tables[ngram_size].get(ngram,None)
        if match is not None:
            start_idx = match + ngram_size
            end_idx = start_idx + num_pred_tokens
            return lookup_tables['input_ids'][0, start_idx:end_idx]
    return torch.tensor([], dtype=torch.long, device=lookup_tables['input_ids'].device)

@torch.no_grad()
def find_candidate_pred_tokens(input_ids,promt, max_ngram_size=2, num_pred_tokens=10):
    # return  torch.tensor([], dtype=torch.long, device=input_ids.device)
    input_length = input_ids.size(1)

    # Ensure max_ngram_size and num_pred_tokens are valid
    if max_ngram_size <= 0 or num_pred_tokens <= 0 or max_ngram_size > input_length:
        raise ValueError("Invalid max_ngram_size or num_pred_tokens")

    for ngram_size in range(max_ngram_size, 0, -1):
        # Extract the last n tokens as our search ngram
        ngram = promt[0, -ngram_size:].tolist()
        # Create sliding windows of size ngram_size
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)
        # Convert ngram to a tensor for comparison
        ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)

        # Find where the windows match the ngram
        matches = (windows == ngram_tensor).all(dim=2)

        # Get the indices of matches
        match_indices = matches.nonzero(as_tuple=True)[1]

        # Iterate through match indices to find a valid continuation
        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens
            # Ensure we don't go beyond the length of input_ids and avoid self-match
            if end_idx <= input_length and start_idx < input_length - ngram_size:
                return input_ids[0, start_idx:end_idx]

    # If no match is found, return an empty tensor
    return torch.tensor([], dtype=torch.long, device=input_ids.device)

COLORS = ["\x1b[31m", "\x1b[32m", "\x1b[34m", "\x1b[35m"]  # Red, Green, Blue, Magenta
UNDERLINE = "\x1b[4m"
RESET = "\x1b[0m"

      
@torch.no_grad()
def streamer_generate(prompt,model,tokenizer,speed=240,preprocessor=lambda x:x,verbose=False,final_text='',max_len=2048,
                      acceptance='greedy',top_k=10,generation_mode='ar',use_logs=True):
    if use_logs:
        assert prompt in ALL_LOGS, f"prompt not found"
        logs = ALL_LOGS[prompt]
        if len(logs) == 0:
            print("Using conventional")
            streamer = InputTextStreamer(prompt,tokenizer,speed,preprocessor=preprocessor,final_text=final_text)
        else:
            streamer = SimulatedTextStreamer(logs,tokenizer,speed,preprocessor=preprocessor,final_text=final_text)
    else:
        streamer = InputTextStreamer(prompt,tokenizer,speed,preprocessor=preprocessor,final_text=final_text)
    sentence_breaks = get_sentence_break(tokenizer)
    streamer.advance(5)
    prompt,prompt_text = streamer.get_prompt('cuda') # ids
    t0 = time.time()
    past_key_values = DynamicCache()
    output = model.generate(prompt.cuda(),do_sample=False,use_cache=True,return_dict_in_generate=True,past_key_values=past_key_values,max_new_tokens=30,top_k=None,num_beams=1,repetition_penalty=1)
    prev_generation = output.sequences[:,prompt.shape[1]:]
    past_key_values = output.past_key_values
    
    #output,past_key_values,_ =  baseline_generate(prompt.cuda(),model,tokenizer,DynamicCache(),max_new_token=max_len)

    prev_ids = prompt
    t1 = time.time()
    streamer.advance(t1-t0)
    ttfs = None
    extra_data = {}
    while True:
        new_ids,prompt_text = streamer.get_prompt('cuda')
        
        if streamer.is_done():
            max_new_token = max_len
        else:
            max_new_token = 10
        t0 = time.time()
        # verbose = True
        generation,past_key_values,timestampe_first_stentence,extra_data = speculative_step(prev_ids,new_ids,prev_generation, model,tokenizer,past_key_values,max_new_token=max_new_token,prompt_text=prompt_text,verbose=verbose,sentence_breaks=sentence_breaks,
                                                                                            acceptance=acceptance,top_k=top_k,generation_mode=generation_mode)
        prev_ids = new_ids
        prev_generation = generation
        #eos_reached = full_generation[0][-1] == tokenizer.eos_token_id
        t1 = time.time()
        if timestampe_first_stentence is not None:
            ttfs = timestampe_first_stentence - t0
        if streamer.is_done():
            streamer.advance(t1-t0)
            break
            # if eos_reached:
            #     streamer.advance(t1-t0)
            #     break
            # else:
            #     streamer.advance(t1-t0)
            #     print("Overhead:",t1-t0)
            #     t0 = time.time()
            #     generation,past_key_values,full_generation = step(new_ids,model,past_key_values,full_generation,512)
            #     t1 = time.time()
            #     print("ACT Latency:",t1-t0)
            #     streamer.advance(t1-t0)
            #     break
        else:
            streamer.advance(t1-t0)
    gen_text = tokenizer.batch_decode(generation)[0]
    extra_data['gen_text']=gen_text
    extra_data['ttfs']=ttfs
    extra_data['latency']=streamer.latency
    return generation,streamer.latency,extra_data
        
       

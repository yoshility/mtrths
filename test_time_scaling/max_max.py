import copy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model and tokenizer (global)
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
model.eval()

# 現在のステップ数
num_step = 0

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[..., -1, None]] = -float('Inf')
    return out

def generate_one_step(step) -> str:
    # stepはsystem,user(,assistant)をすべて含む
    # これに対してダブル改行が出るまで続きを生成する
    input_text = tokenizer.apply_chat_template(
        step,
        tokenize=False,
        add_generation_prompt=True
    )
    input_tokens = tokenizer(input_text, return_tensors="pt").to(model.device)
    generated_ids = input_tokens["input_ids"]
    past_key_values = None

    for t in range(200): # max token
        if t == 0:
            input_ids = generated_ids
        else:
            input_ids = generated_ids[:, -1:]
        outputs = model.forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True
        )
        temperature = 0.7
    
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        
        # choose the next token
        adjusted_logits = logits[:, -1, :] / temperature
        topk_logits = top_k_logits(adjusted_logits, k=20)
        probs = torch.softmax(topk_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        
        # add to generated tokens
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        
        # decode to text (ちゃんとダブル改行検知できてるかチェック)
        # decoded_next_token_id = tokenizer.decode(next_token_id[0])
        # enter_exist = '\n\n' in decoded_next_token_id
        # print(f"[{decoded_next_token_id}]:[{enter_exist}]", end=" ")

        # if \n\n or EOS, stop generation
        if "\n\n" in tokenizer.decode(next_token_id[0]):
            print("\n- \\n\\n detected. Stopping generation.")
            break
        elif next_token_id.item() == tokenizer.eos_token_id:
            print("\n- EOS detected. Stopping generation.")
            break
    
    generated_text = ''.join([tokenizer.decode(w, skip_special_tokens=True) for w in generated_ids[0]][(input_tokens.input_ids.shape[-1]):])
    next_step = copy.deepcopy(step)
    if len(step) == 2: # systemとuserのみのとき
        next_step.append(
            {"role": "assistant", "content": generated_text}
        )
    else: # assistantすでにあるとき
        next_step[2]["content"] += generated_text
    return next_step, generated_ids

def static_value(step: str) -> float:
    return 1.0

# min-max -> max-max
def max_max(step: str, child_id: int, depth: int):
    print(f"\n[Start max-max (depth={depth})]\n")
    # 読み深さに達した
    if depth == 0:
        print("読み深さに達した")
        return static_value(step), None, None
    # 子ノード生成
    children = []
    tokenized_children = []
    for i in range(3):
        print(f"\n[generate children[{child_id+str(i)}]]\n")
        next_step, tokenized_next_step = generate_one_step(step)
        children.append(next_step)
        tokenized_children.append(tokenized_next_step)
        print(f"\nchildren[{child_id+str(i)}]:\n{children[i]}\n")
    
    max_score = -1e4
    max_index = 0
    for i in range(3):
        print(f"\n[max_max(children[{child_id+str(i)}])]\n")
        score, _, _ = max_max(children[i], child_id+str(i), depth-1)
        if score > max_score:
            max_score = score
            max_index = i
    
    return max_score, children[max_index], tokenized_children[max_index]

# 繰り返し次のステップを決めていく
if __name__ == '__main__':
    # 初期化
    # use chat template -> avoid hallucination
    # input
    prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    messages = [
        {"role": "system", "content": "You are a helpful assistant solving math problems."},
        {"role": "user", "content": prompt}
    ]
    now_step = messages

    # 生成が終わるまでmax-maxを繰り返す
    while True:
        num_step += 1
        print(f"\n<<num_step: {num_step}>>\n")
        _, next_step, tokenized_next_step = max_max(
            step=now_step,
            child_id=f"{num_step}-C", # for debug, assign an id to each node
            depth=1
        )
        now_step = next_step
        print(f"\nchosen_next_step:\n{next_step}\n")
        # もし選んだ次ステップでEOSなら、生成終了
        if tokenized_next_step[0][-1] == tokenizer.eos_token_id:
            print(f"\n生成すべて完了\n")
            break
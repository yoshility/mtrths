import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# input
prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

# model and tokenizer
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
model.eval()

# use chat template -> avoid hallucination
messages = [
    {"role": "system", "content": "You are a helpful assistant solving math problems."},
    {"role": "user", "content": prompt}
]

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
        probs = torch.softmax(adjusted_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        
        # add to generated tokens
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        
        # decode to text
        # print(tokenizer.decode(next_token_id[0]), end=' ')

        # if \n\n or EOS, stop generation
        if "\n\n" in tokenizer.decode(next_token_id[0]):
            print("\\n\\n detected. Stopping generation.")
            break
        elif next_token_id.item() == tokenizer.eos_token_id:
            print("EOS detected. Stopping generation.")
            break
    
    generated_text = ''.join([tokenizer.decode(w, skip_special_tokens=True) for w in generated_ids[0]][(input_tokens.input_ids.shape[-1]):])
    next_step = step.copy()
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
def max_max(step: str, depth: int):
    # 読み深さに達した
    if depth == 0:
        print("読み深さに達した")
        return static_value(step), None, None
    # 子ノード生成
    children = []
    tokenized_children = []
    for _ in range(3):
        next_step, tokenized_next_step = generate_one_step(step)
        children.append(next_step)
        tokenized_children.append(tokenized_next_step)
    print(f"\nchildren[0]:\n{children[0]}\n")
    print(f"\ntokenized_children[0]:\n{tokenized_children[0]}\n")
    
    max_score = -1e4
    max_index = 0
    for i in range(3):
        score, _, _ = max_max(children[i], depth-1)
        if score > max_score:
            max_score = score
            max_index = i
    
    return max_score, children[max_index], tokenized_children[max_index]

# 次のステップを決める
# 初期化
now_step = messages
# 生成が終わるまで
while True:
    _, next_step, tokenized_next_step = max_max(
        step=now_step,
        depth=1
    )
    now_step = next_step
    print(f"\nnext_step:\n{next_step}\n")
    # もし選んだ次ステップでEOSなら、生成終了
    if tokenized_next_step[0][-1] == tokenizer.eos_token_id:
        print(f"\n生成すべて完了\n")
        break
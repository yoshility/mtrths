# use llama3 to generate token one by one

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
input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(f"\ninput_text:\n{input_text}\n")
input_tokens = tokenizer(input_text, return_tensors="pt").to(model.device)
print(f"\ninput_tokens:\n{input_tokens}\n")
generated_ids = input_tokens["input_ids"]
past_key_values = None

print(f"\neos id: {tokenizer.eos_token_id}\n")
print(f"\neos token: {tokenizer.decode(tokenizer.eos_token_id)}\n")

# model.forward (generate token one by one)
for step in range(270): # max token number
    if step == 0:
        input_ids = generated_ids # whole prompt
    else:
        input_ids = generated_ids[:, -1:] # the last token only
    
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
    print(tokenizer.decode(next_token_id[0]), end=' ')

    # if EOS, stop generation
    if next_token_id.item() == tokenizer.eos_token_id:
        print("EOS detected. Stopping generation.")
        break

# final output
print("=== Final Output ===")
print([tokenizer.decode(w, skip_special_tokens=True) for w in generated_ids[0]][(input_tokens.input_ids.shape[-1]):])

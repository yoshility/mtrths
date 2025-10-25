import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F


def make_step_rewards(logits, token_masks):
    # print(f"\nlogits:\n{logits}\n")
    probabilities = F.softmax(logits, dim=-1)
    # print(f"\nprobabilities(after softmax):\n{probabilities}\n")
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    # print(f"\nprobabilities(after unsqueeze):\n{probabilities}\n")
    # print(f"\nprobabilities.size():{probabilities.size()}\n") # torch.Size([1, 454, 2])
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


model_name = "Qwen/Qwen2.5-Math-7B-PRM800K"
device = "auto"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained( # transformers>=4.40.0 required
    model_name, 
    device_map=device, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

def prm(step):
    assert len(step) == 3

    # 答えが\n\nで終わるときとEOS(空)で終わるときがあるので
    splited_answer = step[2]["content"].split('\n\n')
    if splited_answer[-1] == '': # \n\nで終わるとき
        splited_answer = splited_answer[:-1]

    messages = [
        {"role": "system", "content": step[0]["content"]},
        {"role": "user", "content": step[1]["content"]},
        {"role": "assistant", "content": "<extra_0>".join(splited_answer) + "<extra_0>"},
    ]
    conversation_str = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )

    input_ids = tokenizer.encode(
        conversation_str, 
        return_tensors="pt", 
    ).to(model.device)

    outputs = model(input_ids=input_ids, use_cache=False)

    step_sep_id = tokenizer.encode("<extra_0>")[0]
    token_masks = (input_ids == step_sep_id)
    step_reward = make_step_rewards(outputs[0], token_masks)
    # step_reward: [[0.9921875, 0.2333984375, 0.6796875, 0.94140625]]
    # return step_reward
    return step_reward[0][-1]

'''
問題点：
- torch と transformers のバージョンが合わなくてエラー起きてしまう
- model() に use_cache=False を付け加えたら動いた（応急処置）
'''

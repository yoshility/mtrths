import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
import shelve
import pandas as pd
from tqdm import tqdm
import argparse
import torch
from transformers import BertTokenizerFast
import json
import re

def auc_pr(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    auc_pr = auc(recall, precision)
    return auc_pr

def auc_roc(y_true, y_pred):
    auc_roc = roc_auc_score(y_true, y_pred)
    return auc_roc

def evaluate_whole_level(raw_data):
    print("evaluate whole level...")
    max_prob_0, max_prob_1 = [], []
    avg_prob_0, avg_prob_1 = [], []
    max_ent_0, max_ent_1 = [], []
    avg_ent_0, avg_ent_1 = [], []
    label_wrong, label_correct = [], []
    for key in tqdm(raw_data):
        gen_probs = raw_data[key]['gen_probs'].flatten()
        entropy = raw_data[key]['entropy'].flatten()
        if raw_data[key]['is_correct']:
            max_prob_1.append(np.nanmax(-np.log(gen_probs)))
            avg_prob_1.append(-np.nanmean(np.log(gen_probs)))
            max_ent_1.append(np.nanmax(entropy))
            avg_ent_1.append(np.nanmean(entropy))
            label_correct.append(1 - raw_data[key]['is_correct'])
            # is_correct=1 -> low uncertainty score -> uncertain label=0
        else:
            max_prob_0.append(np.nanmax(-np.log(gen_probs)))
            avg_prob_0.append(-np.nanmean(np.log(gen_probs)))
            max_ent_0.append(np.nanmax(entropy))
            avg_ent_0.append(np.nanmean(entropy))
            label_wrong.append(1 - raw_data[key]['is_correct'])
            # is_correct=0 -> high uncertainty score -> uncertain label=1
    evaluation = {
        'max_prob': max_prob_0 + max_prob_1,
        'avg_prob': avg_prob_0 + avg_prob_1,
        'max_ent': max_ent_0 + max_ent_1,
        'avg_ent': avg_ent_0 + avg_ent_1,
        'is_wrong': label_wrong + label_correct,
        'AUC_max_prob': auc_roc(label_wrong + label_correct, max_prob_0 + max_prob_1),
        'AUC_avg_prob': auc_roc(label_wrong + label_correct, avg_prob_0 + avg_prob_1),
        'AUC_max_ent': auc_roc(label_wrong + label_correct, max_ent_0 + max_ent_1),
        'AUC_avg_ent': auc_roc(label_wrong + label_correct, avg_ent_0 + avg_ent_1)
    }

    return evaluation

def evaluate_thought_level(raw_data):
    print("evaluate thought level...")
    max_prob_0, max_prob_1 = [], []
    avg_prob_0, avg_prob_1 = [], []
    max_ent_0, max_ent_1 = [], []
    avg_ent_0, avg_ent_1 = [], []
    label_wrong, label_correct = [], []
    for key in tqdm(raw_data):
        words = raw_data[key]['decoded_word'][0]
        # print(f"\nwords:\n\n{words}\n")
        words_len = len(words)
        gen_probs = raw_data[key]['gen_probs'].flatten()
        entropy = raw_data[key]['entropy'].flatten()
        recorder = [[], [], [], []]
        last_sep = -1
        separated_thoughts = []

        i, sep = 0, 0
        while i < words_len:
            if '\n\n' in words[i] or i == words_len-1:
                sep = i
                i += 1
            else:
                i += 1
                continue
            thought = words[last_sep+1:sep+1] # include sep
            # print(f"\nthought: {thought}\n")
            separated_thoughts.append(''.join(thought))
            thought_probs = gen_probs[last_sep+1:sep+1]
            thought_entropy = entropy[last_sep+1:sep+1]
            if len(thought_probs) > 0:
                max_prob = np.nanmax(-np.log(thought_probs))
                recorder[0].append(max_prob)
                avg_prob = -np.nanmean(np.log(thought_probs))
                recorder[1].append(avg_prob)
            if len(thought_entropy) > 0:
                max_ent = np.nanmax(thought_entropy)
                recorder[2].append(max_ent)
                avg_ent = np.nanmean(thought_entropy)
                recorder[3].append(avg_ent)
            last_sep = sep

        if raw_data[key]['is_correct']:
            max_prob_1.append(np.nanmean(recorder[0]))
            avg_prob_1.append(np.nanmean(recorder[1]))
            max_ent_1.append(np.nanmean(recorder[2]))
            avg_ent_1.append(np.nanmean(recorder[3]))
            label_correct.append(1 - raw_data[key]['is_correct'])
            # is_correct=1 -> low uncertainty score -> uncertain label=0
        else:
            max_prob_0.append(np.nanmean(recorder[0]))
            avg_prob_0.append(np.nanmean(recorder[1]))
            max_ent_0.append(np.nanmean(recorder[2]))
            avg_ent_0.append(np.nanmean(recorder[3]))
            label_wrong.append(1 - raw_data[key]['is_correct'])
            # is_correct=0 -> high uncertainty score -> uncertain label=1
    
    print(f"\nlen(label_wrong): {len(label_wrong)}")
    print(f"\nlen(label_correct): {len(label_correct)}")
    
    evaluation = {
        'max_prob': max_prob_0 + max_prob_1,
        'avg_prob': avg_prob_0 + avg_prob_1,
        'max_ent': max_ent_0 + max_ent_1,
        'avg_ent': avg_ent_0 + avg_ent_1,
        'is_wrong': label_wrong + label_correct,
        'AUC_max_prob': auc_roc(label_wrong + label_correct, max_prob_0 + max_prob_1),
        'AUC_avg_prob': auc_roc(label_wrong + label_correct, avg_prob_0 + avg_prob_1),
        'AUC_max_ent': auc_roc(label_wrong + label_correct, max_ent_0 + max_ent_1),
        'AUC_avg_ent': auc_roc(label_wrong + label_correct, avg_ent_0 + avg_ent_1)
    }

    return evaluation

def evaluate_MARS(output, raw_data):
    device = "cuda:0"
    model_importance = torch.load('model_phrase.pth', map_location=device).to(device)
    tokenizer_importance = BertTokenizerFast.from_pretrained("bert-base-uncased")
    for key in tqdm(raw_data):
        print(f"\ninstance['question']:\n\n{output[int(key)]['question']}")
        print(f"\ndecoded_word:\n\n{raw_data[key]['decoded_word'][0]}")
        print(f"\nlen(decoded_word):\n\n{len(raw_data[key]['decoded_word'][0])}")
        tokenized_input = tokenizer_importance.encode_plus(
            output[int(key)]['question'],
            output[int(key)]['pred'],
            add_special_tokens=True,
            return_token_type_ids=True,
            is_split_into_words=True,
            truncation=True,
            max_length=512
        )
        attention_mask = torch.tensor(tokenized_input['attention_mask']).reshape(1,-1).to(device)
        input_ids = torch.tensor(tokenized_input['input_ids']).reshape(1,-1).to(device)
        token_type_ids = torch.tensor(tokenized_input['token_type_ids']).reshape(1,-1).to(device)
        word_ids = tokenized_input.word_ids()
        print(f"\ninput_ids: {input_ids.shape}\n")
        print(f"\nattention_mask: {attention_mask.shape}\n")
        print(f"\ntoken_type_ids: {token_type_ids.shape}\n")
        logits = model_importance(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits[0].cpu()
        importance_scores = torch.nn.functional.sigmoid(logits[:, 2])
        print(f"\nlen(input_ids): {len(input_ids)}\n")
        print(f"\ninput_ids:\n\n{input_ids}\n")
        print(f"\nlen(importance_scores): {len(importance_scores)}\n")
        print(f"\nimportance_scores:\n\n{importance_scores}\n")
        exit(0)
        

def main():
    '''
    Command example:
    python evaluator.py --model llama3 --dataset gsm8k
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["llama3", "qwen2"])
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "multiarith"])
    args = parser.parse_args()

    # get output data
    with open(f"results/output_{args.model}_{args.dataset}.json") as f:
        output = json.load(f)
    
    # get raw data
    print(f"Loading raw_data_{args.model}_{args.dataset} ...")
    shelve_raw_data = shelve.open(f"results/raw_data_{args.model}_{args.dataset}")
    raw_data = dict(shelve_raw_data)
    shelve_raw_data.close()
    print("Finished loading raw_data")

    # prepare dataframe
    result_df = pd.DataFrame() # is_wrong and uncertainty scores
    final_result_df = pd.DataFrame() # AUC-ROC

    # evaluate
    # evaluation_whole_level = evaluate_whole_level(raw_data)
    # evaluation_thought_level = evaluate_thought_level(raw_data)
    evaluation_MARS = evaluate_MARS(output, raw_data)

    # # stock uncertainty scores
    # result_df['is_wrong'] = evaluation_whole_level['is_wrong']
    # result_df['whole_max_prob'] = evaluation_whole_level['max_prob']
    # result_df['whole_avg_prob'] = evaluation_whole_level['avg_prob']
    # result_df['whole_max_ent'] = evaluation_whole_level['max_ent']
    # result_df['whole_avg_ent'] = evaluation_whole_level['avg_ent']

    # assert evaluation_whole_level['is_wrong'] == evaluation_thought_level['is_wrong']
    # result_df['thought_max_prob'] = evaluation_thought_level['max_prob']
    # result_df['thought_avg_prob'] = evaluation_thought_level['avg_prob']
    # result_df['thought_max_ent'] = evaluation_thought_level['max_ent']
    # result_df['thought_avg_ent'] = evaluation_thought_level['avg_ent']

    # # stock AUC-ROC
    # final_result_df['model'] = [args.model]
    # final_result_df['whole_max_prob'] = [evaluation_whole_level['AUC_max_prob']]
    # final_result_df['whole_avg_prob'] = [evaluation_whole_level['AUC_avg_prob']]
    # final_result_df['whole_max_ent'] = [evaluation_whole_level['AUC_max_ent']]
    # final_result_df['whole_avg_ent'] = [evaluation_whole_level['AUC_avg_ent']]

    # final_result_df['thought_max_prob'] = [evaluation_thought_level['AUC_max_prob']]
    # final_result_df['thought_avg_prob'] = [evaluation_thought_level['AUC_avg_prob']]
    # final_result_df['thought_max_ent'] = [evaluation_thought_level['AUC_max_ent']]
    # final_result_df['thought_avg_ent'] = [evaluation_thought_level['AUC_avg_ent']]
    
    # save file
    # result_df.to_csv(f"result_{args.model}_{args.dataset}.csv")
    # final_result_df.to_csv(f"final_result_{args.model}_{args.dataset}_aucroc.csv")

if __name__ == '__main__':
    main()

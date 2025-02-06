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

def inference(model, tokenizer, question, answer, ans_word, gen_probs):
    device = "cuda:0"
    # words = re.findall(r'\w+|[^\w\s]', answer)
    words = ans_word
    tokenized_input = tokenizer.encode_plus(
        [question],
        words,
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

    logits = model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids).logits[0].cpu()
    classes = logits[:,0:2]
    scores = torch.nn.functional.sigmoid(logits[:,2])
    
    print(f"\n-----inference-----\n")

    print(f"\nwords:\n\n{words}\n")
    print(f"\nlen(words): {len(words)}\n")

    print(f"\nlen(gen_probs): {len(gen_probs)}\n")

    print(f"\ntoken_type_ids:\n\n{token_type_ids}\n")
    print(f"\nlen(token_type_ids):\n\n{len(token_type_ids[0])}\n")

    print(f"\nword_ids:\n\n{word_ids}\n")
    print(f"\nlen(word_ids):\n\n{len(word_ids)}\n")

    print(f"\nlogits.shape: {logits.shape}\n")

    print(f"\nclasses:\n\n{classes}\n")
    print(f"\nlen(classes): {len(classes)}\n")

    print(f"\nscores:\n\n{scores}\n")
    print(f"\nlen(scores): {len(scores)}\n")

    phrases = []
    importance_scores = []
    phrases_probs = []
    i = 0
    # num_skip = 0
    while(i<len(scores)):
        if word_ids[i] == None or token_type_ids[0][i] == 0:
            i += 1
            # num_skip += 1 
            continue
        # print(f"\nnum_skip: {num_skip}\n")
        
        cl = torch.argmax(classes[i,:])
        if word_ids[i] == 0 or cl == 0: #we handle the edge case as well (beginning of the sentence)
        # class=0のwordがphraseの始まり！！！
            for j in range(i+1, len(scores)):
                cl = torch.argmax(classes[j,:])
                continue_word = False
                for k in range(i,j):
                    if word_ids[k] == word_ids[j]:
                        continue_word = True
                if (cl == 0 or  word_ids[j] == None) and continue_word == False:
                    # cl=0となるタイミングでjを区切る
                    break
            
            #find corresponding words by using word_ids
            min_word_id = word_ids[i]
            max_word_id = word_ids[j-1]
            phrases.append(''.join(words[min_word_id:max_word_id+1]))
            # phraseの先頭wordのscoreを採用
            importance_scores.append(scores[i].item())
            # 今求めたphraseの出力確率を求める
            phrases_probs.append(max(gen_probs[min_word_id:max_word_id+1]))
            i = j

    print(f"\nphrases:\n\n{phrases}\n")

    #maybe modify phrase with actual sentence
    # real_phrases = []
    # phrase_ind  = 0
    # i = 0
    # answer = answer.strip()


    # while(i < len(answer)):
    #     last_token_place  = -1
    #     for j in range(i+1, len(answer)+1):

    #         if  phrases[phrase_ind].strip().replace(" ", "") == answer[i:j].strip().replace(" ", ""):
    #             last_token_place = j

    #     real_phrases.append(answer[i:last_token_place].strip())
    #     i = last_token_place
    #     phrase_ind += 1
    real_phrases = phrases
            
    return real_phrases, importance_scores, phrases_probs

def evaluate_MARS(output, raw_data):
    device = "cuda:0"
    model_importance = torch.load('model_phrase.pth', map_location=device).to(device)
    tokenizer_importance = BertTokenizerFast.from_pretrained("bert-base-uncased")
    for i in tqdm(range(len(output))):
        question = output[i]['question']
        print(f"\nquestion:\n\n{question}\n")
        answer = output[i]['pred']
        ans_word = raw_data[str(i)]['decoded_word'][0]
        gen_probs = raw_data[str(i)]['gen_probs'].flatten()
        print(f"\nanswer:\n\n{answer}\n")
        phrases, importance_vector, phrases_probs = inference(
            model_importance, tokenizer_importance, question, answer, ans_word, gen_probs
        )

        print(f"\nreal phrases:\n\n{phrases}\n")
        print(f"\nlen(real phrases): {len(phrases)}\n")

        print(f"\nimportance_vector:\n\n{importance_vector}\n")
        print(f"\nlen(importance_vector): {len(importance_vector)}")

        print(f"\nphrases_probs:\n\n{phrases_probs}\n")
        print(f"\nlen(phrases_probs): {len(phrases_probs)}")
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
    print(f"Loading output_{args.model}_{args.dataset} ...")
    with open(f"results/output_{args.model}_{args.dataset}.json") as f:
        output = json.load(f)
    print(f"Finished loading output")
    
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

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
from scipy.special import softmax
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


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

def inference(importance_model, importance_tokenizer, question, answer, ans_word):
    device = "cuda:0"
    # words = re.findall(r'\w+|[^\w\s]', answer)
    words = ans_word
    tokenized_input = importance_tokenizer.encode_plus(
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

    logits = importance_model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids).logits[0].cpu()
    classes = logits[:,0:2]
    # if classes[0] > classes[1]: the token is phrase head
    # else: the token is not phrase head
    # ↓ importance score for each token
    scores = torch.nn.functional.sigmoid(logits[:,2])
    
    # print(f"\n-----inference-----\n")

    # print(f"\nwords:\n\n{words}\n")
    # print(f"\nlen(words): {len(words)}\n") # 115

    # print(f"\nlen(gen_probs): {len(gen_probs)}\n") # 115

    # print(f"\ntoken_type_ids:\n\n{token_type_ids}\n")
    # print(f"\nlen(token_type_ids):\n\n{len(token_type_ids[0])}\n") # 143 (0/1のやつ)

    # print(f"\nword_ids:\n\n{word_ids}\n")
    # print(f"\nlen(word_ids):\n\n{len(word_ids)}\n") # 143 ([None, 0, ..., 0, None, 0, 1, 2, ..., 113, None])

    # print(f"\nlogits.shape: {logits.shape}\n") # (143, 3)

    # print(f"\nclasses:\n\n{classes}\n")
    # print(f"\nlen(classes): {len(classes)}\n") # 143

    # print(f"\nscores:\n\n{scores}\n")
    # print(f"\nlen(scores): {len(scores)}\n") # 143

    phrases = [] # merge the tokens into phrases based on "classes"
    importance_scores = []
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
            # phrases_probs.append(max(gen_probs[min_word_id:max_word_id+1]))
            i = j

    # print(f"\nphrases:\n\n{phrases}\n")
    # print(f"\nlen(phrases): {len(phrases)}\n") # 79

    #maybe modify phrase with actual sentence
    '''
    phrases -> real_phrases
        - delete space (' out' -> 'out')
        - delete \n ('.\n\n' -> '.')
    '''
    real_phrases = []
    phrase_ind  = 0
    i = 0
    answer = answer.strip()

    while(i < len(answer)):
        last_token_place  = -1
        for j in range(i+1, len(answer)+1):
            # phraseの走査終わってるのにanswerの走査終わってないとき
            if phrase_ind == len(phrases):
                return [], []
            if phrases[phrase_ind].strip().replace(" ", "") == answer[i:j].strip().replace(" ", ""):
                last_token_place = j

        real_phrases.append(answer[i:last_token_place].strip())
        i = last_token_place
        phrase_ind += 1
    
    # print(f"\nreal_phrases:\n\n{real_phrases}\n")
    # print(f"\nlen(real_phrases): {len(real_phrases)}\n") # 79
    # print(f"\nimportance_scores:\n\n{importance_scores}\n")
    # print(f"\nlen(importance_scores): {len(importance_scores)}\n") # 79

    return real_phrases, importance_scores #, phrases_probs

bem = hub.load('https://tfhub.dev/google/answer_equivalence/bem/1')
device = torch.device("cuda:0")
VOCAB_PATH = 'gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12/vocab.txt'
vocab_table = tf.lookup.StaticVocabularyTable(
    tf.lookup.TextFileInitializer(
        filename=VOCAB_PATH,
        key_dtype=tf.string,
        key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
        value_dtype=tf.int64,
        value_index=tf.lookup.TextFileIndex.LINE_NUMBER
    ),
    num_oov_buckets=1
)
cls_id, sep_id = vocab_table.lookup(tf.convert_to_tensor(['[CLS]', '[SEP]']))
bert_tokenizer = text.BertTokenizer(
    vocab_lookup_table=vocab_table,
    token_out_type=tf.int64, 
    preserve_unused_token=True, 
    lower_case=True
)

def bertify_example(example):
    question = bert_tokenizer.tokenize(example['question']).merge_dims(1, 2)
    reference = bert_tokenizer.tokenize(example['reference']).merge_dims(1, 2)
    candidate = bert_tokenizer.tokenize(example['candidate']).merge_dims(1, 2)

    input_ids, segment_ids = text.combine_segments(
        (candidate, reference, question),
        cls_id,
        sep_id
    )

    return {'input_ids': input_ids.numpy(), 'segment_ids': segment_ids.numpy()}

def pad(a, length=512):
    return np.append(a, np.zeros(length - a.shape[-1], np.int32))

def bertify_examples(examples):
    input_ids = []
    segment_ids = []
    for example in examples:
        example_inputs = bertify_example(example)
        if example_inputs['input_ids'].shape[-1] > 512:
            return None
        input_ids.append(pad(example_inputs['input_ids']))
        segment_ids.append(pad(example_inputs['segment_ids']))

    return {'input_ids': np.stack(input_ids), 'segment_ids': np.stack(segment_ids)}

def get_importance_vector_BEM_thought(answer_text, steps, question_text):
    importance_vector = []

    #words = answer.split()
    # print(f"\nsteps:\n\n{steps}\n")
    
    #encoded_answer = sentence_model.encode(answer)

    for i in range(len(steps)):
        removed_answer = steps[:i] + steps[i+1:]

        removed_answer = ' '.join(removed_answer)

        # print(f"\ni={i} | removed_answer:\n\n{removed_answer}\n")
        bem_input = [{
            'question': question_text,
            'reference': answer_text,
            'candidate': removed_answer
        }]

        inputs = bertify_examples(bem_input)
        # print(f"\ninputs:\n\n{inputs}\n")
        if inputs == None:
            return []
        raw_outputs = bem(inputs)
        bem_score = float(softmax(np.squeeze(raw_outputs))[1])
        score = 1-bem_score
        importance_vector.append(score)

    importance_vector = np.array(importance_vector)

    return importance_vector

def evaluate_MARS(output, raw_data):
    # ---------------------- use trained BERT [実装汚いけど完動] -------------------------------------------------
    '''
    Use trained BERT to give importance score to each phrase
    '''

    # ----- prepare trained BERT -----
    device = "cuda:0"
    importance_model = torch.load('model_phrase.pth', map_location=device).to(device)
    importance_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # ----- 各モデル出力について処理 -----
    uncertainty_scores = []
    for i in tqdm(range(len(output))):
        # ----- 材料準備 -----
        invalid_flag = False # 途中でおかしくなったとき用
        question = output[i]['question']
        answer = output[i]['pred']
        # print(f"\nanswer:\n\n{answer}\n")
        ans_word = raw_data[str(i)]['decoded_word'][0]
        # print(f"\nans_word:\n\n{ans_word}\n")
        gen_probs = raw_data[str(i)]['gen_probs'].flatten()

        # ----- ans_wordをwords_foreach_stepに分ける -----
        words_foreach_step = []
        tmp = []
        for w in ans_word:
            tmp.append(w)
            if '\n\n' in w:
                words_foreach_step.append(tmp)
                tmp = []
        if tmp:
            words_foreach_step.append(tmp)
        # print(f"\nwords_foreach_step:\n\n{words_foreach_step}\n")

        # ----- answer_foreach_stepにする（answer: str） -----
        answer_foreach_step = []
        for words in words_foreach_step:
            answer_foreach_step.append(''.join(words))
        # print(f"\nanswer_foreach_step:\n\n{answer_foreach_step}\n")

        # ----- 各stepのphrase分けと重みづけ -----
        all_phrases = [] # ほしいもの
        all_importance_scores = [] # ほしいもの
        for s, words in enumerate(words_foreach_step):
            phrases_foreach_step, importance_scores_foreach_step = inference(
                importance_model,
                importance_tokenizer,
                question,
                answer=answer_foreach_step[s],
                ans_word=words,
            )
            if len(phrases_foreach_step) == 0:
                invalid_flag = True
                break
            all_phrases.append(phrases_foreach_step)
            all_importance_scores.append(importance_scores_foreach_step)
        if invalid_flag:
            output[i]['invalid'] = 1
            print("invalid!")
            continue
            
        # print(f"\nall_phrases:\n\n{all_phrases}\n")
        # print(f"\nall_importance_scores:\n\n{all_importance_scores}\n\n")

        # ----- gen_probsをprobs_foreach_stepに分ける -----
        probs_foreach_step = []
        tmp = []
        for wi, w in enumerate(ans_word):
            tmp.append(gen_probs[wi])
            if '\n\n' in w:
                probs_foreach_step.append(tmp)
                tmp = []
        if tmp:
            probs_foreach_step.append(tmp)
        # print(f"\nprobs_foreach_step:\n\n{probs_foreach_step}\n")
        
        # ----- 各step"で"重み付き尤度和 -----
        all_steps_weighted_nll = []
        token_idx = 0
        for phrases_foreach_step, importance_scores_foreach_step in zip(all_phrases, all_importance_scores):
            if invalid_flag:
                break
            # print(f"\n----- now looking at step: -----\n\n{phrases_foreach_step}\n")
            nll_foreach_word = [] # nll_foreach_phraseでは？
            merged_importance_scores = []
            ph = 0
            while ph < len(phrases_foreach_step):
                if invalid_flag:
                    break
                found = False
                while found == False:
                    if invalid_flag:
                        break
                    for k in range(1, len(phrases_foreach_step)-ph+1):
                        word = "".join(phrases_foreach_step[ph:ph+k])
                        # print(f"\n----- now looking at phrase: -----\n\n{word}")
                        last_token = -1
                        for l in range(token_idx+1, len(gen_probs)+1):
                            # print(f"\n----- l: {l} -----\n")
                            # print(f"ans_word[token_idx:l]:{ans_word[token_idx:l]}")
                            # print(f"word:{word}\n")
                            if "".join(ans_word[token_idx:l]).strip().replace(" ", "").lower() == word.strip().replace(" ", "").lower():
                                # print("bingo!")
                                last_token = l
                        
                        if last_token == -1:
                            print("Couldn't find matching.")
                            # フレーズと単語がマッチングしなかったとき
                            invalid_flag = True
                            break
                            
                        if last_token != -1:
                            nll_foreach_word.append(torch.mean(-torch.log(gen_probs)[token_idx:last_token]))
                            merged_importance_scores.append(torch.mean(torch.Tensor(importance_scores_foreach_step[ph:ph+k])))
                            found = True
                            ph += k
                            token_idx = last_token
                            break
            merged_importance_scores = torch.Tensor(merged_importance_scores) / sum(torch.Tensor(merged_importance_scores))
            # uncertainty score for each step
            # print(f"\nmerged_importance_scores:\n\n{merged_importance_scores}\n")
            # print(f"\nnll_foreach_word:\n\n{nll_foreach_word}\n")
            weighted_nll_foreach_step = 0.5 * torch.sum(merged_importance_scores * torch.Tensor(nll_foreach_word)) + 0.5 * torch.mean(torch.Tensor(nll_foreach_word))
            # print(f"\nweighted_nll_foreach_step: {weighted_nll_foreach_step}\n")
            all_steps_weighted_nll.append(weighted_nll_foreach_step)
        # print(f"\nall_steps_weighted_nll:\n\n{all_steps_weighted_nll}\n")
        # フレーズと単語がマッチングしなかったとき
        if invalid_flag:
            print("invalid!")
            output[i]['invalid'] = 1
            continue
            
        # ----- use original BEM [各step"に"重みを付ける] -----
        importance_scores_step = get_importance_vector_BEM_thought(
            answer_text=answer,
            steps=answer_foreach_step,
            question_text=question
        )
        # ↓ 文章が長すぎてBEMに入らなかったとき
        if len(importance_scores_step) == 0:
            print("invalid!")
            output[i]['invalid'] = 1
            continue
        importance_scores_ofeach_step = importance_scores_step / np.sum(importance_scores_step)
        # print(f"\nimportance_scores_step:\n\n{importance_scores_step}\n")
        
        # ----- 各stepを重み付き和 -----
        final_uncertainty_score = 0.5 * torch.sum(torch.Tensor(all_steps_weighted_nll) * torch.Tensor(importance_scores_ofeach_step)) + 0.5 * torch.mean(torch.Tensor(all_steps_weighted_nll))
        uncertainty_scores.append(final_uncertainty_score)

        # ----- write debug file -----
        # with open("/data/yoshie/mtrths/debug.txt", "a") as f:
        #     print(f"\n------------------ output[{i}] | is_correct: {output[i]['is_correct']} ------------------\n", file=f)
        #     print(f"\nquestion:\n{question}\n", file=f)
        #     print(f"\nanswer_word:\n{ans_word}\n", file=f)
        #     print(f"\ngen_probs:\n{gen_probs}\n", file=f)
        #     print(f"\nanswer_phrase:\n{all_phrases}\n", file=f)
        #     print(f"\nimportance_scores_foreach_phrase:\n{all_importance_scores}\n", file=f)
        #     print(f"\nstep_answer:\n{answer_foreach_step}\n", file=f)
        #     print(f"\nnll_foreach_step:\n{all_steps_weighted_nll}\n", file=f)
        #     print(f"\nimportance_scores_foreach_step:\n{importance_scores_ofeach_step}\n", file=f)
        #     print(f"\nuncertainty_score: {final_uncertainty_score}\n", file=f)

    # ----- AUCROC -----
    us_1, us_0 = [], []
    qid_1, qid_0 = [], []
    o = 0 # index for outputs
    u = 0 # index for uncertainty scores
    while (o < len(output) and u < len(uncertainty_scores)):
        if 'invalid' in output[o].keys():
            o += 1
            continue
        if output[o]['is_correct']:
            us_0.append(uncertainty_scores[u])
            qid_0.append(output[o]['index'])
        else:
            us_1.append(uncertainty_scores[u])
            qid_1.append(output[o]['index'])
        u += 1
        o += 1
    assert u == len(uncertainty_scores)
    label_wrong = [1] * len(us_1)
    label_correct = [0] * len(us_0)
    
    evaluation = {
        'uscore': us_1 + us_0,
        'is_wrong': label_wrong + label_correct,
        'question_id': qid_1 + qid_0,
        'AUC': auc_roc(label_wrong + label_correct, us_1 + us_0)
    }

    return evaluation  
        
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
    with open(f"/data/yoshie/mtrths/output_{args.model}_{args.dataset}.json") as f:
        output = json.load(f)
    print(f"Finished loading output")
    
    # get raw data
    print(f"Loading raw_data_{args.model}_{args.dataset} ...")
    shelve_raw_data = shelve.open(f"/data/yoshie/mtrths/raw_data_{args.model}_{args.dataset}")
    raw_data = dict(shelve_raw_data)
    shelve_raw_data.close()
    print("Finished loading raw_data")

    # prepare dataframe
    result_df = pd.DataFrame() # is_wrong and uncertainty scores
    final_result_df = pd.DataFrame() # AUC-ROC

    # evaluate
    # evaluation_whole_level = evaluate_whole_level(raw_data)
    # evaluation_thought_level = evaluate_thought_level(raw_data)
    evaluation_MARS = evaluate_MARS(
        output, raw_data
    )

    # ----- stock uncertainty scores -----
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

    result_df['is_wrong'] = evaluation_MARS['is_wrong']
    result_df['MARS'] = evaluation_MARS['uscore']
    result_df['question_id'] = evaluation_MARS['question_id']

    # ----- stock AUC-ROC -----
    # final_result_df['model'] = [args.model]
    # final_result_df['whole_max_prob'] = [evaluation_whole_level['AUC_max_prob']]
    # final_result_df['whole_avg_prob'] = [evaluation_whole_level['AUC_avg_prob']]
    # final_result_df['whole_max_ent'] = [evaluation_whole_level['AUC_max_ent']]
    # final_result_df['whole_avg_ent'] = [evaluation_whole_level['AUC_avg_ent']]

    # final_result_df['thought_max_prob'] = [evaluation_thought_level['AUC_max_prob']]
    # final_result_df['thought_avg_prob'] = [evaluation_thought_level['AUC_avg_prob']]
    # final_result_df['thought_max_ent'] = [evaluation_thought_level['AUC_max_ent']]
    # final_result_df['thought_avg_ent'] = [evaluation_thought_level['AUC_avg_ent']]

    final_result_df['model'] = [args.model]
    final_result_df['MARS'] = [evaluation_MARS['AUC']]
    
    # save file
    result_df.to_csv(f"/data/yoshie/mtrths/result_{args.model}_{args.dataset}.csv")
    final_result_df.to_csv(f"/data/yoshie/mtrths/final_result_{args.model}_{args.dataset}_aucroc.csv")

if __name__ == '__main__':
    main()

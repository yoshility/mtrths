import argparse
import shelve
from tqdm import tqdm
import json

import models
import dataset
import utils

if __name__ == '__main__':
    '''
    Command example:
    python main.py --model llama3 --dataset gsm8k --num_instances 500
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["llama3", "qwen2"])
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "multiarith"])
    parser.add_argument("--num_instances", type=int)
    args = parser.parse_args()

    # load model
    if args.model == "llama3":
        model = models.Llama3()
    elif args.model == "qwen2":
        model = models.Qwen2()
    else:
        print("\nInvalid model!\n")
        exit(1)

    # load dataset
    if args.dataset == "gsm8k":
        ds = dataset.GSM8K()
    elif args.dataset == "multiarith":
        ds = dataset.MultiArith()
    else:
        print("\nInvalid dataset!\n")
        exit(1)

    # prepare
    output = [] # raw output from model
    is_correct_sum = 0
    not_correct_sum = 0
    print(f"not_correct_sum: {not_correct_sum}")
    raw_data = shelve.open(f"/data/yoshie/mtrths/raw_data_{args.model}_{args.dataset}") # is_correct, probs, entropy, ...
    NUM_INSTANCES = args.num_instances
    # NUM_INSTANCES = args.num_instances if args.num_instances <= len(ds) else len(ds)

    # use LLM
    for i in tqdm(range(NUM_INSTANCES)):
        res = model.chat(ds.get_question(i)) # res is a dict
        gt = ds.get_answer(i)
        is_correct = utils.check_is_correct(gt, res['decoded_text'][0])
        is_correct_sum += is_correct
        not_correct_sum += (1 - is_correct)
        if not is_correct:
            print(f"not_correct_sum: {not_correct_sum}")
        output.append({
            "index": i,
            "question": ds.get_question(i),
            "pred": res['decoded_text'][0],
            "answer": ds.get_raw_answer(i),
            "is_correct": is_correct
        })
        res['is_correct'] = is_correct
        raw_data[str(i)] = res

    raw_data.close()

    with open(f"/data/yoshie/mtrths/output_{args.model}_{args.dataset}.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
        print(f"output_{args.model}_{args.dataset}.json generated")
    
    print(f"is_correct_sum: {is_correct_sum}")
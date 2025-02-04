import torch

def get_answer(answer):
    i = -1
    ans = ""
    while (answer[i] != ' '):
        ans = answer[i] + ans
        i -= 1
    return ans

def check_is_correct(gt, pred):
    # put ',' in gt (e.g. 488000 -> 488,000)
    i = len(gt)-1
    cnt = 0
    new_gt = ""
    while i >= 0:
        if cnt != 0 and cnt % 3 == 0:
            new_gt = ',' + new_gt
        new_gt = gt[i] + new_gt
        cnt += 1
        i -= 1
    
    # get final sentence from answer
    new_pred = pred.split('\n\n')[-1]

    return int(new_gt in new_pred or gt in new_pred)

def compute_entropy(scores):
    """
    Compute entropy by given scores over the entire vocabularies.
    Args:
        scores: the scores for list of tokens returned from the model.
    Return:
        The entropy
    """
    # Reference: https://github.com/huggingface/transformers/blob/17a55534f5e5df10ac4804d4270bf6b8cc24998d/src/transformers/generation/logits_process.py#L330
    
    # We do the following computation in 
    # for-loop because in some cases,
    # it may contain a list of 
    # tensors whose sizes are not the same
    ent_list = list()
    for i in scores:
        normalized = torch.nn.functional.log_softmax(i, dim=-1)
        p = torch.exp(normalized)
        ent_list.append(-(normalized * p).nansum(-1, keepdim=True))
    ent = torch.stack(ent_list).reshape(-1)
    return ent

if __name__ == '__main__':
    gt = '12999488000'
    pred = "To find the total time it takes to download the file, we need to break it down into parts.\n\nFirst, Carla downloads 40% of the file at 2 GB/minute. \n40% of 200 GB is 0.4 * 200 = 80 GB.\nTime taken to download 80 GB = 80 / 2 = 40 minutes.\n\nThen, Windows forces a restart and Carla has to wait for 20 minutes.\n\nAfter the restart, Carla has to download the remaining 60% of the file (100% - 40% = 60%).\n60% of 200 GB is 0.6 * 200 = 120 GB.\nTime taken to download 120 GB = 120 / 2 = 60 minutes.\n\nThe total time taken is the sum of the time taken to download 80 GB, the wait time, and the time taken to download the remaining 120 GB.\nTotal time = 40 minutes + 20 minutes + 60 minutes = 120 minutes.\n\nTherefore, the total time it takes to download the file is 12,999,488,000 minutes."
    print(check_is_correct(gt, pred))
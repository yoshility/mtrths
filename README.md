## workflow
```
Questions
|
| llama3.py
|
output_{model}_{dataset}.json ... model output
raw_data_{model}_{dataset}.db ... probs and entropy
|
| evaluator.py (ここを試行錯誤)
|
result_{model}_{dataset}.csv ... is_wrong and uncertainty scores
final_result_{model}_{dataset}.csv ... AUC-PR
|
| visualize.py
|
*.png ... relation between is_correct and uncertainty scores, etc.
```

## system prompt for dataset
GSM8K: "You are a helpful assistant solving math problems."

MultiArith: "You are a helpful assistant solving math problems."
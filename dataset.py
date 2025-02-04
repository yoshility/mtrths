from datasets import load_dataset

class GSM8K:
    def __init__(self):
        self.data = load_dataset("openai/gsm8k", "main", split="train")
    
    def get_question(self, i):
        return self.data[i]["question"]

    def get_raw_answer(self, i):
        return self.data[i]["answer"]

    def get_answer(self, i):
        answer = self.data[i]["answer"]
        j = -1
        ans = ""
        while (answer[j] != ' '):
            ans = answer[j] + ans
            j -= 1
        return ans

class MultiArith:
    def __init__(self):
        self.data = load_dataset("ChilleD/MultiArith", split="train")

    def get_question(self, i):
        return self.data[i]["question"]
    
    def get_raw_answer(self, i):
        return self.data[i]["final_ans"]
    
    def get_answer(self, i):
        return self.data[i]["final_ans"]
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
import torch
import utils

class Llama3:
    def __init__(self):
        self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        # config = LlamaConfig.from_pretrained(self.model_id)
        # config.rope_scaling = {"type": "linear", "factor": 8.0}
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # config=config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    
    def chat(self, prompt):
        self.model.eval()
        messages = [
            {"role": "system", "content": "You are a helpful assistant solving math problems."},
            {"role": "user", "content": prompt}
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        device = "cuda"

        res = dict()

        input_tokens = self.tokenizer(input_text, return_tensors="pt").to(device)
        '''
        input_tokens:
        {
            'input_ids': tensor([[128000, 128000, ...]]), # (1, #input_token) (e.g. (1, 83))
            'attention_mask': tensor([[1, 1, ...]])
        }
        '''
        outputs = self.model.generate(
            **input_tokens, # set attention_mask by this
            max_new_tokens=512,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        '''
        len(outputs[0][0]) = #all_output_token
        '''
        input_len = input_tokens.input_ids.shape[-1]
        gen_sequences = outputs.sequences[:, input_len:].cpu()
        '''
        gen_sequences: (1, n (output w/o input)) (e.g. (1, 155))
                = [[1271, 1505, ..., 128009]]
        '''
        # decoded_text = [self.tokenizer.decode(i, skip_special_tokens=True) for i in gen_sequences] # i = gen_sequences[0]
        decoded_text = [self.tokenizer.decode(i) for i in gen_sequences]
        print(f"\ndecoded_text:\n{decoded_text}\n")
        '''
        decoded_text: ['To find the ...']
        '''
        decoded_word = [[self.tokenizer.decode(word, skip_special_tokens=True) for word in i] for i in gen_sequences]
        '''
        decoded_word:
                = [['To', ' find', 'the', ...]]
        '''
        scores = outputs.scores
        '''
        scores: (n, 1, #vocab) (tuple)
                = ([[-inf, -inf, ..., -inf]], --
                   [[-inf, -inf, ..., -inf]],  | n
                   ...                         |
                   [[-inf, -inf, ..., -inf]]) --
        '''
        '''
        torch.stack(scores, dim=1): (1, n, #vocab)
                = [[[-inf, -inf, ..., -inf],  --
                    [-inf, -inf, ..., -inf],   | n
                    ...                        |
                    [-inf, -inf, ..., -inf]]] --
        '''
        probs = torch.stack(scores, dim=1).float().softmax(-1).cpu()
        '''
        probs: (1, n, #vocab)
                = [[[1.0, 1.0, ..., 1.0],  --
                    [1.0, 1.0, ..., 1.0],   | n
                    ...                     |
                    [1.0, 1.0, ..., 1.0]]] --
        '''
        '''
        gen_sequences[:, :, None]: (1, n, 1)
                = [[[  1271],  --
                    [  1505],   | n
                    ...         |
                    [128009]]] --
        '''
        gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
        '''
        gen_probs: (1, n) (e.g. (1, 155))
                = [[1.0, 1.0, ..., 1.0]]
        '''
        '''
        torch.stack(scores): (n, 1, #vocab) (tensor)
                = [[[-inf, -inf, ..., -inf]], --
                   [[-inf, -inf, ..., -inf]],  | n
                   ...                         |
                   [[-inf, -inf, ..., -inf]]] --
        '''
        entropy = utils.compute_entropy(torch.stack(scores).float().cpu())
        '''
        entropy: (n, ) (e.g. (155, ))
                = [-0.0, -0.0, ..., -0.0]
        '''

        res = {
            "gen_sequences": gen_sequences,
            "scores": torch.stack(scores).float().cpu(),
            "gen_probs": gen_probs,
            "gen_sequences": gen_sequences,
            "decoded_text": decoded_text,
            "decoded_word": decoded_word,
            "entropy": entropy
        }

        return res

class Qwen2:
    def __init__(self):
        self.model_id = "Qwen/Qwen2-7B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    
    def chat(self, prompt):
        # same with Llama3's chat function
        messages = [
            {"role": "system", "content": "You are a helpful assistant solving math problems."},
            {"role": "user", "content": prompt}
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        device = "cuda"
        res = dict()
        input_tokens = self.tokenizer(input_text, return_tensors="pt").to(device)

        outputs = self.model.generate(
            **input_tokens,
            max_new_tokens=512,
            output_scores=True,
            return_dict_in_generate=True
        )
        input_len = input_tokens.input_ids.shape[-1]
        gen_sequences = outputs.sequences[:, input_len:].cpu()
        decoded_text = [self.tokenizer.decode(i, skip_special_tokens=True) for i in gen_sequences]
        decoded_word = [[self.tokenizer.decode(word, skip_special_tokens=True) for word in i] for i in gen_sequences]
        scores = outputs.scores
        probs = torch.stack(scores, dim=1).float().softmax(-1).cpu()
        gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
        entropy = utils.compute_entropy(torch.stack(scores).float().cpu())

        res = {
            "scores": torch.stack(scores).float().cpu(),
            "gen_probs": gen_probs, 
            "gen_sequences": gen_sequences,
            "decoded_text": decoded_text,
            "decoded_word": decoded_word,
            "entropy": entropy
        }

        return res
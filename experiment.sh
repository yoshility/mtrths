#!/bin/bash

python main.py --model llama3 --dataset multiarith --num_instances 420

python main.py --model qwen2 --dataset gsm8k --num_instances 500

python main.py --model qwen2 --dataset multiarith --num_instances 420

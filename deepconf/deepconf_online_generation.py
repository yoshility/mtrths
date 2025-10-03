import openai
import json
from tqdm import tqdm
import time
import os
from datetime import datetime
import numpy as np
from collections import Counter
from dynasor.core.evaluator import math_equal

# ===========================
# Configuration
# ===========================
MODEL_PATH = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MAX_TOKENS = 64000
RID = 0
QID = 0
PORT = 8000
DATASET_FILE = "aime25.jsonl"

# Online algorithm parameters
WARMUP_TRACES = 16
TOTAL_BUDGET = 32
CONFIDENCE_PERCENTILE = 90
WINDOW_SIZE = 2048

# ===========================
# Answer Extraction Functions
# ===========================
def extract_answer(text):
    """Extract boxed answer from text"""
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        return a.strip()
    return None

def parse_func(s):
    for f in [parse_latex, parse_expr, latex2sympy]:
        try:
            return f(s.replace("\\\\", "\\"))
        except:
            try:
                return f(s)
            except:
                pass
    return s


# ===========================
# Data Loading
# ===========================
def load_aime25_jsonl(file_path=DATASET_FILE):
    """Load data from aime25.jsonl file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

# ===========================
# Confidence Calculation
# ===========================
def compute_confidence(logprobs):
    """Compute confidence score from logprobs."""
    confs = []
    for lp in logprobs:
        confs.append(round(-sum([l.logprob for l in lp]) / len(lp), 3))
    return confs

def compute_least_grouped(confs, group_size=WINDOW_SIZE):
    """Compute sliding window mean confidence with specified group size."""
    if len(confs) < group_size:
        return [sum(confs) / len(confs)] if confs else [0]

    sliding_means = []
    for i in range(len(confs) - group_size + 1):
        window = confs[i:i + group_size]
        sliding_means.append(round(sum(window) / len(window), 3))

    return sliding_means

# ===========================
# Voting Functions
# ===========================
def weighted_majority_vote(answers, weights):
    """Perform weighted majority voting"""
    if not answers:
        return None

    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)

    if not answer_weights:
        return None

    voted_answer = max(answer_weights.keys(), key=lambda x: answer_weights[x])
    return voted_answer

# ===========================
# Trace Processing and Accuracy Calculation
# ===========================
def process_trace(choice, trace_id, ground_truth):
    """Process a single trace and calculate accuracy"""
    # Extract basic info
    text = choice.message.content
    tokens = [t.token for t in choice.logprobs.content]

    # Calculate confidence
    confs = compute_confidence([t.top_logprobs for t in choice.logprobs.content])
    sliding_window = compute_least_grouped(confs, group_size=WINDOW_SIZE)

    # Extract and parse answer
    extracted_answer = extract_answer(text)
    parsed_answer = parse_func(extracted_answer) if extracted_answer else None

    # Calculate correctness
    is_correct = False
    if extracted_answer is not None and ground_truth is not None:
        is_correct = math_equal(extracted_answer, ground_truth)

    trace_data = {
        "trace_id": trace_id,
        "stop_reason": choice.stop_reason,
        "finish_reason": choice.finish_reason,
        "text": text,
        "tokens": tokens,
        "token_count": len(tokens),
        "confs": confs,
        "group_confs": sliding_window,
        "min_conf": min(sliding_window) if sliding_window else 0,
        "extracted_answer": extracted_answer,
        "parsed_answer": parsed_answer,
        "is_correct": is_correct,
    }

    return trace_data

def calculate_statistics(traces, phase_name=""):
    """Calculate statistics for a list of traces"""
    if not traces:
        return {}

    total_traces = len(traces)
    correct_traces = sum(1 for t in traces if t['is_correct'])
    total_tokens = sum(t['token_count'] for t in traces)

    # Confidence statistics
    min_confs = [t['min_conf'] for t in traces]

    stats = {
        f"{phase_name}_traces": total_traces,
        f"{phase_name}_correct": correct_traces,
        f"{phase_name}_accuracy": correct_traces / total_traces if total_traces > 0 else 0,
        f"{phase_name}_total_tokens": total_tokens,
        f"{phase_name}_avg_tokens_per_trace": total_tokens / total_traces if total_traces > 0 else 0,
        f"{phase_name}_min_conf_mean": np.mean(min_confs) if min_confs else 0,
        f"{phase_name}_min_conf_std": np.std(min_confs) if min_confs else 0,
    }

    return stats

# ===========================
# Problem Processing Function (like problemx in reference)
# ===========================
def process_problem_voting(test_json, ground_truth):
    """
    Process a problem result JSON and perform voting
    Similar to problemx function in reference code
    """
    answers = []
    bar = test_json['conf_bar']
    weights = []
    tokens = 0

    print(f"Warmup traces: {len(test_json['warmup_traces'])}, Final traces: {len(test_json['final_traces'])}")

    # Process warmup traces
    for i in range(len(test_json['warmup_traces'])):
        answer = extract_answer(test_json['warmup_traces'][i]['text'])
        minx = min(test_json['warmup_traces'][i]['group_confs'])
        tokens += len(test_json['warmup_traces'][i]['tokens'])
        if minx < bar:
            continue
        if answer is not None:
            answers.append(answer)
            weights.append(1)  # Use weight 1 for consistency with reference

    # Process final traces
    for i in range(len(test_json['final_traces'])):
        tokens += len(test_json['final_traces'][i]['tokens'])
        # Skip traces stopped by gconf
        if test_json['final_traces'][i]['stop_reason'] is not None and 'gconf' in test_json['final_traces'][i]['stop_reason']:
            continue
        answer = extract_answer(test_json['final_traces'][i]['text'])
        minx = min(test_json['final_traces'][i]['group_confs'])
        if answer is not None:
            answers.append(answer)
            weights.append(1)

    # Perform voting
    voted = weighted_majority_vote(answers, weights)
    is_correct = str(voted) == str(ground_truth) if voted is not None else False

    print(f'Bar: {bar:.4f}, Voted: {voted}, Ground truth: {ground_truth}, Correct: {is_correct}, Voting answers: {len(answers)}')

    return is_correct, len(answers), tokens

# ===========================
# Main Function
# ===========================
def main():
    print("="*60)
    print("ONLINE ALGORITHM WITH ACCURACY CALCULATION")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Question ID: {QID}")
    print(f"Run ID: {RID}")
    print(f"Warmup traces: {WARMUP_TRACES}")
    print(f"Total budget: {TOTAL_BUDGET}")
    print(f"Confidence percentile: {CONFIDENCE_PERCENTILE}")

    # Load the data
    data = load_aime25_jsonl()
    print(f"Loaded {len(data)} items from {DATASET_FILE}")

    # Initialize client
    client = openai.OpenAI(
        api_key="None",
        base_url=f"http://localhost:{PORT}/v1",
        timeout=None
    )

    # Get question and ground truth
    prompt = data[QID]['question']
    ground_truth = str(data[QID].get('answer', '')).strip()

    messages = [
        {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
        {"role": "user", "content": prompt}
    ]

    print(f"\nQuestion: {prompt}")
    print(f"Ground Truth: {ground_truth}")

    # ===========================
    # WARMUP PHASE
    # ===========================
    print(f"\n{'-'*40}")
    print("WARMUP PHASE")
    print(f"{'-'*40}")

    t0 = time.time()
    responses = client.chat.completions.create(
        model=MODEL_PATH,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        logprobs=True,
        top_logprobs=20,
        n=WARMUP_TRACES,
        extra_body={"top_k": 0},
    )
    t1 = time.time()

    # Process warmup traces
    warmup_traces = []
    min_confs = []

    for j in range(WARMUP_TRACES):
        trace_data = process_trace(responses.choices[j], j, ground_truth)
        warmup_traces.append(trace_data)
        min_confs.append(trace_data["min_conf"])

    # Calculate confidence bar
    conf_bar = float(np.percentile(min_confs, CONFIDENCE_PERCENTILE))

    # Calculate warmup statistics
    warmup_stats = calculate_statistics(warmup_traces, "warmup")

    print(f"Warmup time: {t1 - t0:.2f}s")
    print(f"Confidence bar (P{CONFIDENCE_PERCENTILE}): {conf_bar:.4f}")
    print(f"Warmup accuracy: {warmup_stats['warmup_accuracy']:.4f} ({warmup_stats['warmup_correct']}/{warmup_stats['warmup_traces']})")
    print(f"Warmup total tokens: {warmup_stats['warmup_total_tokens']}")
    print(f"Warmup avg tokens per trace: {warmup_stats['warmup_avg_tokens_per_trace']:.1f}")

    # Show some example results
    print(f"\nFirst 3 warmup traces:")
    for i, trace in enumerate(warmup_traces[:3]):
        print(f"  Trace {i}: {trace['extracted_answer']} (correct: {trace['is_correct']}, conf: {trace['min_conf']:.4f}, tokens: {trace['token_count']})")

    # ===========================
    # FINAL PHASE
    # ===========================
    print(f"\n{'-'*40}")
    print("FINAL PHASE (with early stopping)")
    print(f"{'-'*40}")

    real_gen = TOTAL_BUDGET - WARMUP_TRACES

    t3 = time.time()
    responses = client.chat.completions.create(
        model=MODEL_PATH,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        logprobs=True,
        top_logprobs=20,
        n=real_gen,
        extra_body={
            "top_k": 0,
            "vllm_xargs": {
                'enable_conf': True,
                'window_size': WINDOW_SIZE,
                'threshold': conf_bar
            }
        }
    )
    t4 = time.time()

    # Process final traces
    final_traces = []
    for j in range(len(responses.choices)):
        trace_data = process_trace(responses.choices[j], WARMUP_TRACES + j, ground_truth)
        final_traces.append(trace_data)

    # Calculate final statistics
    final_stats = calculate_statistics(final_traces, "final")

    print(f"Final time: {t4 - t3:.2f}s")
    print(f"Final traces generated: {len(final_traces)} (requested: {real_gen})")
    print(f"Final accuracy: {final_stats['final_accuracy']:.4f} ({final_stats['final_correct']}/{final_stats['final_traces']})")
    print(f"Final total tokens: {final_stats['final_total_tokens']}")
    print(f"Final avg tokens per trace: {final_stats['final_avg_tokens_per_trace']:.1f}")

    # Show some example results
    print(f"\nFirst 3 final traces:")
    for i, trace in enumerate(final_traces[:3]):
        print(f"  Trace {i}: {trace['extracted_answer']} (correct: {trace['is_correct']}, conf: {trace['min_conf']:.4f}, tokens: {trace['token_count']})")

    # ===========================
    # OVERALL STATISTICS
    # ===========================
    print(f"\n{'-'*40}")
    print("OVERALL STATISTICS")
    print(f"{'-'*40}")

    all_traces = warmup_traces + final_traces
    overall_stats = calculate_statistics(all_traces, "overall")

    total_time = t4 - t0
    warmup_time = t1 - t0
    final_time = t4 - t3

    print(f"Total time: {total_time:.2f}s (warmup: {warmup_time:.2f}s, final: {final_time:.2f}s)")
    print(f"Overall traces: {len(all_traces)}")
    print(f"Overall accuracy: {overall_stats['overall_accuracy']:.4f} ({overall_stats['overall_correct']}/{overall_stats['overall_traces']})")
    print(f"Overall total tokens: {overall_stats['overall_total_tokens']}")
    print(f"Overall avg tokens per trace: {overall_stats['overall_avg_tokens_per_trace']:.1f}")

    # Efficiency metrics
    tokens_per_second = overall_stats['overall_total_tokens'] / total_time
    traces_per_second = len(all_traces) / total_time

    print(f"Tokens per second: {tokens_per_second:.1f}")
    print(f"Traces per second: {traces_per_second:.2f}")

    # Early stopping analysis
    if final_traces:
        above_threshold = sum(1 for t in final_traces if t['min_conf'] >= conf_bar)
        print(f"Final traces above threshold: {above_threshold}/{len(final_traces)} ({above_threshold/len(final_traces):.2%})")

    # ===========================
    # SAVE RESULTS
    # ===========================
    results = {
        "question_id": QID,
        "run_id": RID,
        "question": prompt,
        "ground_truth": ground_truth,
        "conf_bar": conf_bar,
        "warmup_traces": warmup_traces,
        "final_traces": final_traces,
        "statistics": {
            **warmup_stats,
            **final_stats,
            **overall_stats,
            "total_time": total_time,
            "warmup_time": warmup_time,
            "final_time": final_time,
            "tokens_per_second": tokens_per_second,
            "traces_per_second": traces_per_second,
        },
        "config": {
            "model_path": MODEL_PATH,
            "warmup_traces": WARMUP_TRACES,
            "total_budget": TOTAL_BUDGET,
            "confidence_percentile": CONFIDENCE_PERCENTILE,
            "window_size": WINDOW_SIZE,
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs", exist_ok=True)
    pickle.dump(results, open(f"outputs/q{QID}_r{RID}_{timestamp}.pkl", 'wb'))
    print(f"Results saved to outputs/q{QID}_r{RID}_{timestamp}.pkl")
    output_filename = f"outputs/q{QID}_r{RID}_{timestamp}.pkl"

    print(f"\n✅ Results saved to outputs/q{QID}_r{RID}_{timestamp}.pkl")

    # ===========================
    # COMPARISON WITH BASELINE
    # ===========================
    print(f"\n{'-'*40}")
    print("COMPARISON ANALYSIS")
    print(f"{'-'*40}")

    # Compare warmup vs final phase
    if warmup_stats['warmup_traces'] > 0 and final_stats['final_traces'] > 0:
        acc_improvement = final_stats['final_accuracy'] - warmup_stats['warmup_accuracy']
        print(f"Accuracy change (final - warmup): {acc_improvement:+.4f}")

        token_efficiency = final_stats['final_avg_tokens_per_trace'] / warmup_stats['warmup_avg_tokens_per_trace']
        print(f"Token efficiency (final/warmup): {token_efficiency:.2f}x")

    # Early stopping effectiveness
    total_requested = WARMUP_TRACES + (TOTAL_BUDGET - WARMUP_TRACES)
    total_generated = len(all_traces)
    saving_ratio = (total_requested - total_generated) / total_requested

    print(f"Traces requested: {total_requested}")
    print(f"Traces generated: {total_generated}")
    print(f"Early stopping saving: {saving_ratio:.2%}")

    # ===========================
    # VOTING FOR FINAL ANSWER
    # ===========================
    print(f"\n{'-'*40}")
    print("VOTING FOR FINAL ANSWER")
    print(f"{'-'*40}")

    # Collect answers above threshold for voting
    voting_answers = []
    voting_weights = []
    total_voting_tokens = 0

    # Add warmup traces above threshold (use confidence as weight)
    for trace in warmup_traces:
        minx = trace['min_conf']
        total_voting_tokens += trace['token_count']
        if minx >= conf_bar:
            answer = trace['extracted_answer']
            if answer is not None:
                voting_answers.append(answer)
                voting_weights.append(minx)

    # Add final traces above threshold (use weight 1)
    for trace in final_traces:
        total_voting_tokens += trace['token_count']
        # Skip traces that were stopped by gconf (early stopping)
        if trace['stop_reason'] is not None and 'gconf' in trace['stop_reason']:
            continue

        minx = trace['min_conf']
        # Note: final traces might not need threshold check since they were already filtered
        # But keeping it consistent with the reference code
        if minx >= conf_bar:
            answer = trace['extracted_answer']
            if answer is not None:
                voting_answers.append(answer)
                voting_weights.append(1)

    print(f"Traces used for voting: {len(voting_answers)}")
    print(f"Total tokens in voting traces: {total_voting_tokens}")

    # Perform weighted majority vote
    voted_answer = weighted_majority_vote(voting_answers, voting_weights)

    # Check if voted answer is correct
    is_voted_correct = False
    if voted_answer is not None and ground_truth:
        is_voted_correct = str(voted_answer) == str(ground_truth)
        # Also try math_equal for more robust comparison
        try:
            is_voted_correct_math = math_equal(voted_answer, ground_truth)
            if is_voted_correct != is_voted_correct_math:
                print(f"Warning: String comparison ({is_voted_correct}) != math comparison ({is_voted_correct_math})")
                is_voted_correct = is_voted_correct_math  # Use math_equal as more reliable
        except:
            pass  # Fallback to string comparison

    print(f"Voted answer: {voted_answer}")
    print(f"Ground truth: {ground_truth}")
    print(f"Voted answer correct: {is_voted_correct}")

    # Show voting breakdown
    if voting_answers:
        answer_counts = Counter(voting_answers)
        print(f"\nVoting breakdown:")
        for answer, count in answer_counts.most_common():
            print(f"  {answer}: {count} votes")

    # ===========================
    # UPDATE RESULTS WITH VOTING
    # ===========================
    voting_results = {
        "voting_answers": voting_answers,
        "voting_weights": voting_weights,
        "voted_answer": voted_answer,
        "is_voted_correct": is_voted_correct,
        "voting_traces_count": len(voting_answers),
        "voting_total_tokens": total_voting_tokens,
    }

    results["voting"] = voting_results
    results["statistics"]["voting_traces_count"] = len(voting_answers)
    results["statistics"]["voting_total_tokens"] = total_voting_tokens
    results["statistics"]["is_voted_correct"] = is_voted_correct

    # Update the saved file with voting results
    # with open(output_filename, 'w', encoding='utf-8') as f:
    #     pickle.dump(results, f)

    # print(f"\n✅ Results updated with voting information: {output_filename}")

    # ===========================
    # FINAL SUMMARY
    # ===========================
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Question ID: {QID}")
    print(f"Confidence bar: {conf_bar:.4f}")
    print(f"Total traces generated: {len(all_traces)}")
    print(f"Traces used for voting: {len(voting_answers)}")
    print(f"Total tokens: {overall_stats['overall_total_tokens']}")
    print(f"Voting answer: {voted_answer}")
    print(f"Ground truth: {ground_truth}")
    print(f"Final result: {'✅ CORRECT' if is_voted_correct else '❌ INCORRECT'}")

    return results

if __name__ == "__main__":
    results = main()
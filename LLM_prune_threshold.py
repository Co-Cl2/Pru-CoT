import json
import re
import os
import argparse
import sys
from datetime import datetime
from datasets import load_from_disk
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

try:
    from transformers import AutoTokenizer
except ImportError:
    print("pip install transformers")
    exit(1)

try:
    import json_repair
except ImportError:
    print("pip install json_repair")
    exit(1)

try:
    import dashscope
except ImportError:
    print("pip install dashscope")
    exit(1)

try:
    import numpy as np
except ImportError:
    print("pip install numpy")
    np = None


VLLM_SYSTEM_PROMPT = """You are an **aggressive** CoT pruner. Your goal is to **remove all non-essential steps**.

**Core Principle:**
1.  **Default: `"prune": True`.** Prune all redundant steps, clarifications, or checks.
2.  **Exception: `"prune": False`.** ONLY keep steps that are **critically essential**. A step is essential *only if* its removal breaks the logical chain or makes the solution impossible.
3.  **Rule: If in doubt, prune.**

**Task & Format Rules:**
1.  You will see the thinking process in a table. 
2.  **ONLY focus on steps that have a Number ID** in the first column. Steps with an empty ID are protected context.
3.  Respond with *only* a JSON object.
4.  **Keys:** Must be the Number ID from the table (e.g., "1", "2").
5.  **Values:** Must be `{"reasoning": (string), "prune": (boolean)}`.

**Example Response:**
{
  "1": {
    "reasoning": "Prune. Step 1 is a redundant clarification.",
    "prune": True
  },
  "2": {
    "reasoning": "Keep. Step 2 defines variable x, critical for the solution.",
    "prune": False
  }
}
"""

VLLM_USER_PROMPT_TEMPLATE = """
Here is the problem:
{user_input}

Here is the final solution:
{solution}

--------------------------------------------------
**YOUR TASK:**

Below is the **full thinking process**. 
Analyze **ONLY** the steps with a **Number ID** in the first column.

| ID | Thinking Step |
|---|---|
{markdown_table_rows}

For each step with an ID, decide if it can be pruned (`True`) or if it is logically essential (`False`).

Respond *only* with the JSON dictionary.
"""

class MockCompletion:
    def __init__(self, text):
        self.text = text

class MockOutput:
    def __init__(self, text):
        self.outputs = [MockCompletion(text)]


def format_thinking_blocks_to_markdown(thinking_blocks, virtual_id_map):
    rows = []
    sorted_blocks = sorted(thinking_blocks, key=lambda x: x.get("order", 0))

    for block in sorted_blocks:
        real_idx = block.get("order", -1)
        content = str(block.get("content", "")).replace("\n", " ").replace("|", "\|")

        if real_idx in virtual_id_map:
            display_index = str(virtual_id_map[real_idx])
        else:
            display_index = ""

        rows.append(f"| {display_index} | {content} |")

    return "\n".join(rows)


def parse_index_list_from_llm(llm_output_text):
    try:
        parsed_obj = json_repair.loads(llm_output_text)

        if not isinstance(parsed_obj, dict):
            return None, "Parsing_Failed: Output is not a dictionary"

        if not parsed_obj:
            return [], "Success: Empty dictionary"

        virtual_prune_indices = []

        for key, value_dict in parsed_obj.items():
            try:
                v_idx = int(key) 
                should_prune = False
                if isinstance(value_dict, dict):
                    should_prune = value_dict.get("prune", False)
                elif isinstance(value_dict, bool):
                    should_prune = value_dict
                
                if should_prune is True:
                    virtual_prune_indices.append(v_idx)
                    
            except (ValueError, TypeError):
                continue

        return sorted(list(set(virtual_prune_indices))), "Success: Parsed with json_repair"

    except Exception as e:
        return None, f"Parsing_Failed: {str(e)}"


def build_vllm_prompt(item, tokenizer, candidate_indices):
    try:
        user_input = item["user_input"]
        solution = item["solution"]
        thinking_blocks = item["thinking_blocks"]

        sorted_candidates = sorted(candidate_indices)
        virtual_id_map = {real_idx: i+1 for i, real_idx in enumerate(sorted_candidates)}
        
        system_prompt = VLLM_SYSTEM_PROMPT

        markdown_table_rows = format_thinking_blocks_to_markdown(
            thinking_blocks,
            virtual_id_map
        )

        user_prompt_content = VLLM_USER_PROMPT_TEMPLATE.format(
            user_input=user_input,
            markdown_table_rows=markdown_table_rows,
            solution=solution
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_content}
        ]

        vllm_prompt_string = ""
        if tokenizer:
            vllm_prompt_string = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
        return {"text": vllm_prompt_string, "messages": messages}, virtual_id_map 

    except Exception as e:
        print(f"warning: [build_vllm_prompt] error: {e}, skip")
        return None, None


def reconstruct_to_llama_factory_format(score_analysis_item, prune_segment_indices):
    system_msg = score_analysis_item["system_prompt"]
    user_msg = score_analysis_item["user_input"]
    solution = score_analysis_item["solution"]

    original_sections_map = {
        block["order"]: block["content"]
        for block in score_analysis_item["thinking_blocks"]
    }

    kept_sections = []
    for i in sorted(original_sections_map.keys()):
        if i not in prune_segment_indices:
            kept_sections.append(original_sections_map[i])

    if kept_sections:
        think_text = '\n\n'.join(kept_sections)
        assistant_response = f"<think>\n\n{think_text}\n\n</think>\n\n\n\n{solution}"
    else:
        assistant_response = f"<think>\n\n</think>\n\n\n\n{solution}"

    if system_msg:
        conversations = [
            {"content": system_msg, "role": "system"},
            {"content": user_msg, "role": "user"},
            {"content": assistant_response, "role": "assistant"}
        ]
    else:
        conversations = [
            {"content": user_msg, "role": "user"},
            {"content": assistant_response, "role": "assistant"}
        ]

    return conversations

def process_dataset_to_json(dataset):
    all_output_data = []
    required_cols = ["messages", "think_components", "cot_weights"]
    for col in required_cols:
        if col not in dataset.column_names:
            print(f"error: {col}")
            sys.exit(1)

    for example in tqdm(dataset, desc="processing examples"):
        system_msg = ""
        user_msg = ""
        try:
            if example["messages"][0]["role"] == "system":
                system_msg = example["messages"][0]["content"]
            if example["messages"][1]["role"] == "user":
                user_msg = example["messages"][1]["content"]
        except (IndexError, KeyError):
            continue

        try:
            sections = example["think_components"]["assistant_think"]["sections"]
            solution = example["think_components"]["solution"]
            weights = example["cot_weights"]
        except KeyError:
            continue

        if not isinstance(sections, list) or not isinstance(weights, list) or len(sections) != len(weights):
            continue

        formatted_blocks = []
        for i, (section_content, score) in enumerate(zip(sections, weights)):
            formatted_blocks.append({
                "order": i,
                "score": round(float(score), 2),
                "content": section_content
            })

        all_output_data.append({
            "system_prompt": system_msg,
            "user_input": user_msg,
            "thinking_blocks": formatted_blocks,
            "solution": solution
        })
    return all_output_data


def save_data(filename, data):
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_pruning_stats(ratios_list, total_dataset_samples, skipped_preprocessing_count):
    total_samples_in_stats = len(ratios_list)
    base_stats = {
        "total_dataset_samples": total_dataset_samples,
        "samples_skipped_preprocessing": skipped_preprocessing_count,
        "samples_in_stats": total_samples_in_stats
    }

    if not ratios_list:
        base_stats.update({"stats": "No data"})
        return base_stats

    if np is None:
        mean_ratio = sum(ratios_list) / total_samples_in_stats if total_samples_in_stats > 0 else 0
        base_stats.update({"stats": {"mean": f"{mean_ratio:.4f}", "note": "Install numpy for detailed stats"}})
        return base_stats

    ratios = np.array(ratios_list)
    pruned_count = int(np.sum(ratios > 0))

    stats = {
        "mean": f"{np.mean(ratios):.4f}",
        "std_dev": f"{np.std(ratios):.4f}",
        "min": f"{np.min(ratios):.4f}",
        "median": f"{np.median(ratios):.4f}",
        "p25": f"{np.percentile(ratios, 25):.4f}",
        "p75": f"{np.percentile(ratios, 75):.4f}",
        "max": f"{np.max(ratios):.4f}",
    }
    
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    hist_counts, _ = np.histogram(ratios, bins=bins)
    count_zero = int(np.sum(ratios == 0.0))
    hist_counts[0] -= count_zero

    hist_lines = []
    if count_zero > 0:
        pct = (count_zero/total_samples_in_stats)*100
        hist_lines.append(f"  0.0 (Keep All)   | {'#'*int(pct/2):<50} | {count_zero} ({pct:.1f}%)")
    
    labels = ["[0.0-0.1)", "[0.1-0.2)", "[0.2-0.3)", "[0.3-0.4)", "[0.4-0.5)",
              "[0.5-0.6)", "[0.6-0.7)", "[0.7-0.8)", "[0.8-0.9)", "[0.9-1.0]"]
    
    for i, count in enumerate(hist_counts):
        if count == 0: continue
        pct = (count/total_samples_in_stats)*100
        hist_lines.append(f"  {labels[i]:<16} | {'#'*int(pct/2):<50} | {count} ({pct:.1f}%)")

    base_stats.update({
        "samples_pruned_count": pruned_count,
        "stats": stats,
        "histogram": "\n".join(hist_lines)
    })
    return base_stats


def main():
    parser = argparse.ArgumentParser(description="PCoT Pruner with Virtual Indexing & Soft-Masking")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--LLM_path", type=str, required=True, help="Path to HF model (tokenizer) or model name")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--candidate_threshold", type=float, default=0.5)
    parser.add_argument("--median_gate_threshold", type=float, default=1.0)
    parser.add_argument("--print_llm_responses", action="store_true")
    
    parser.add_argument("--use_openai", action="store_true", help="Use DashScope API instead of vLLM")
    parser.add_argument("--api_key", type=str, default=os.getenv("DASHSCOPE_API_KEY"), help="DashScope API Key")
    parser.add_argument("--api_model", type=str, default="qwen-max", help="Model name for DashScope API")

    args = parser.parse_args()

    sampling_params = None
    llm = None

    if args.use_openai:
        api_key = args.api_key
        if not api_key:
            print("DASHSCOPE_API_KEY")
            sys.exit(1)
        dashscope.api_key = api_key
        print(f"DashScope API (model={args.api_model})...")
    else:
        if LLM is None:
            print("LLM error")
            sys.exit(1)
            
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
        print(f"loading vLLM model: {args.LLM_path}...")
        llm = LLM(
            model=args.LLM_path,
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=16384,
            gpu_memory_utilization=0.9,
            max_num_seqs=32,
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.LLM_path, trust_remote_code=True)
    except Exception as e:
        if args.use_openai:
            print(f"error tokenizer ({e})")
            tokenizer = None
        else:
            raise e

    dataset_full_path = os.path.join(args.dataset_path, args.dataset_name)
    dataset = load_from_disk(dataset_full_path)
    if args.rate < 1.0:
        dataset = dataset.train_test_split(train_size=args.rate, seed=42)['train']
    
    data_list = process_dataset_to_json(dataset)
    total_samples_in_dataset = len(data_list)

    final_llama_factory_list = []
    final_failed_items_log = []
    all_pruning_ratios = []
    items_to_process_queue = []
    
    MAX_RETRIES = 0
    
    counters = {
        "median_gate": 0,
        "no_candidates": 0,
        "failed_prompt": 0,
        "retry_failed": 0
    }

    print(f"building Prompt ({total_samples_in_dataset})...")

    for item in tqdm(data_list, desc=""):
        all_scores = [block.get("score", 0.0) for block in item["thinking_blocks"]]
        stat_val = np.median(all_scores) if np and all_scores else (sum(all_scores)/len(all_scores) if all_scores else 0)
        
        if stat_val > args.median_gate_threshold:
            counters["median_gate"] += 1
            final_llama_factory_list.append({"messages": reconstruct_to_llama_factory_format(item, [])})
            all_pruning_ratios.append(0.0)
            continue

        candidate_indices = [
            block["order"] for block in item["thinking_blocks"]
            if float(block.get("score", 1.0)) < args.candidate_threshold
        ]

        if not candidate_indices:
            counters["no_candidates"] += 1
            final_llama_factory_list.append({"messages": reconstruct_to_llama_factory_format(item, [])})
            all_pruning_ratios.append(0.0)
            continue

        prompt_data, virtual_id_map = build_vllm_prompt(item, tokenizer, candidate_indices)

        if prompt_data:
            items_to_process_queue.append({
                "original_item": item,
                "prompt": prompt_data,
                "status": "Initial_Run",
                "llm_output": None,
                "candidate_indices": candidate_indices,
                "virtual_id_map": virtual_id_map
            })
        else:
            counters["failed_prompt"] += 1
            final_llama_factory_list.append({"messages": reconstruct_to_llama_factory_format(item, [])})
            all_pruning_ratios.append(0.0)

    print(f"LLM processing: {len(items_to_process_queue)}")
    
    current_retry = 0
    while current_retry <= MAX_RETRIES and items_to_process_queue:
        current_items = list(items_to_process_queue)
        items_to_process_queue = []
        
        outputs = []
        
        if args.use_openai:

            from concurrent.futures import ThreadPoolExecutor, as_completed

            def call_dashscope(msgs):
                try:
                    response = dashscope.Generation.call(
                        model=args.api_model,
                        messages=msgs,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        result_format='message'
                    )
                    if response.status_code == 200:
                        content = response.output.choices[0].message.content
                        return content
                    else:
                        return f"API_ERROR: {response.code} - {response.message}"
                except Exception as e:
                    return f"API_ERROR: {str(e)}"

            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_idx = {
                    executor.submit(call_dashscope, c_item["prompt"]["messages"]): idx
                    for idx, c_item in enumerate(current_items)
                }
                results = [None] * len(current_items)
                for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="DashScope Requesting"):
                    idx = future_to_idx[future]
                    try:
                        content = future.result()
                        results[idx] = MockOutput(content)
                    except Exception as e:
                        results[idx] = MockOutput(f"API_ERROR: Future exception: {e}")
                outputs = results

        else:
            prompts = [x["prompt"]["text"] for x in current_items]
            try:
                outputs = llm.generate(prompts, sampling_params)
            except Exception as e:
                print(f"Critical vLLM Crash: {e}")
                break

        for item_state, output in tqdm(zip(current_items, outputs), total=len(current_items)):
            llm_text = output.outputs[0].text.strip()
            item_state["llm_output"] = llm_text
            
            if args.print_llm_responses:
                print(f"\n{llm_text}\n")

            try:
                if "API_ERROR:" in llm_text:
                    raise ValueError(llm_text)

                virtual_ids_raw, status = parse_index_list_from_llm(llm_text)
                if virtual_ids_raw is None:
                    raise ValueError(f"Parse Error: {status}")

                virtual_id_map = item_state["virtual_id_map"]
                reverse_map = {v: k for k, v in virtual_id_map.items()}
                
                real_prune_indices = []
                for v_id in virtual_ids_raw:
                    if v_id in reverse_map:
                        real_prune_indices.append(reverse_map[v_id])

                msgs = reconstruct_to_llama_factory_format(item_state["original_item"], real_prune_indices)
                final_llama_factory_list.append({"messages": msgs})
                
                ratio = len(real_prune_indices) / len(item_state["original_item"]["thinking_blocks"])
                all_pruning_ratios.append(ratio)

            except Exception as e:
                item_state["status"] = str(e)
                items_to_process_queue.append(item_state)

        current_retry += 1
        
        if current_retry <= MAX_RETRIES and items_to_process_queue:
            print(f"retry {len(items_to_process_queue)}...")
            next_queue = []
            for item in items_to_process_queue:
                p_data, v_map = build_vllm_prompt(item["original_item"], tokenizer, item["candidate_indices"])
                if p_data:
                    item["prompt"] = p_data
                    item["virtual_id_map"] = v_map
                    next_queue.append(item)
                else:
                    counters["retry_failed"] += 1
                    final_llama_factory_list.append({"messages": reconstruct_to_llama_factory_format(item["original_item"], [])})
                    all_pruning_ratios.append(0.0)
            items_to_process_queue = next_queue

    for item in items_to_process_queue:
        final_failed_items_log.append({
            "input": str(item["original_item"].get("user_input", ""))[:100],
            "error": item["status"],
            "llm_output": item["llm_output"]
        })
        final_llama_factory_list.append({"messages": reconstruct_to_llama_factory_format(item["original_item"], [])})
        all_pruning_ratios.append(0.0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_path, f"{args.dataset_name}_{timestamp}")
    
    save_data(os.path.join(out_dir, "dataset.json"), final_llama_factory_list)
    if final_failed_items_log:
        save_data(os.path.join(out_dir, "failed_log.json"), final_failed_items_log)
    
    stats = generate_pruning_stats(all_pruning_ratios, total_samples_in_dataset, counters["failed_prompt"])
    print(stats.get("histogram", ""))
    
    config = vars(args)
    clean_config = {k:v for k,v in config.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    clean_config.update({"counters": counters, "stats": stats})
    save_data(os.path.join(out_dir, "config.json"), clean_config)
    
    print(f"\nDataset saved to: {out_dir}")

if __name__ == "__main__":
    main()
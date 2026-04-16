from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_from_disk, concatenate_datasets
import os
import torch
import tqdm
import argparse
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import json

def connect_steps_with_indices(steps, line_break_token):

    if not steps:
        return {
            'input_ids': torch.tensor([[]], dtype=torch.long),
            'attention_mask': torch.tensor([[]], dtype=torch.long),
            'segment_indices': []
        }

    all_input_ids = []
    all_attention_masks = []
    
    for step in steps:
        
        all_input_ids.append(step['input_ids'])
        all_attention_masks.append(step['attention_mask'])
        
        all_input_ids.append(line_break_token['input_ids'])
        all_attention_masks.append(line_break_token['attention_mask'])

    steps_input_ids = torch.cat(all_input_ids, dim=1)
    steps_attention_mask = torch.cat(all_attention_masks, dim=1)

    segment_indices = []
    current_index = 0
    line_break_length = line_break_token['input_ids'].shape[1]
    for i, tokenized_segment in enumerate(steps):

        segment_length = tokenized_segment['input_ids'].shape[1]
        
        start_index = current_index
        end_index = current_index + segment_length - 1
        
        segment_indices.append((start_index, end_index))
        
        current_index += segment_length
        current_index += line_break_length

    return {
        'input_ids': steps_input_ids,
        'attention_mask': steps_attention_mask,
        'segment_indices': segment_indices
    }


def prepare_data_with_indices(chat_tokens, think_tokens, solution_tokens):


    original_think_indices = think_tokens.get('segment_indices', [])

    chat_len_offset = chat_tokens['input_ids'].shape[1]

    corrected_think_indices = [
        (start + chat_len_offset, end + chat_len_offset)
        for start, end in original_think_indices
    ]
    
    input_ids = torch.cat([
        chat_tokens['input_ids'],
        think_tokens['input_ids'],
        solution_tokens['input_ids']
    ], dim=1)

    attention_mask = torch.cat([
        chat_tokens['attention_mask'],
        think_tokens['attention_mask'],
        solution_tokens['attention_mask']
    ], dim=1)

    chat_labels = torch.full_like(chat_tokens['input_ids'], -100)
    think_labels = torch.full_like(think_tokens['input_ids'], -100)
    solution_labels = solution_tokens['input_ids'].clone()
    labels = torch.cat([chat_labels, think_labels, solution_labels], dim=1)

    return input_ids, attention_mask, labels, corrected_think_indices

def process_example(example, tokenizer, max_length):

    system_msg = example["messages"][0]["content"]
    user_msg = example["messages"][1]["content"]

    if system_msg:

        chat_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True
        ) + '<think>\n\n'
    else:
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True
        ) + '<think>\n\n'

    thinks_sections = example["think_components"]["assistant_think"]["sections"]
    think_end = '</think>\n\n\n\n'
    solution_text = think_end + example["think_components"]["solution"] + '<｜end▁of▁sentence｜>'

    chat_tokens = tokenizer(chat_text, return_tensors="pt")

    thinks_tokens = [tokenizer(section, return_tensors="pt") for section in thinks_sections]
    num_segments = len(thinks_tokens)

    solution_tokens = tokenizer(solution_text, return_tensors="pt")
    
    line_break_token = tokenizer("\n\n", return_tensors="pt")
    think_tokens = connect_steps_with_indices(thinks_tokens, line_break_token)
    input_ids, attention_mask, labels, segment_indices = prepare_data_with_indices(chat_tokens, think_tokens, solution_tokens)

    current_length = input_ids.shape[1]

    if current_length > max_length:
        pass
    elif current_length < max_length:

        pad_length = max_length - current_length
        
        padding_ids = torch.full((1, pad_length), tokenizer.pad_token_id, dtype=input_ids.dtype)
        padding_mask = torch.zeros((1, pad_length), dtype=attention_mask.dtype)
        padding_labels = torch.full((1, pad_length), -100, dtype=labels.dtype)

        input_ids = torch.cat([padding_ids, input_ids], dim=1)
        attention_mask = torch.cat([padding_mask, attention_mask], dim=1)
        labels = torch.cat([padding_labels, labels], dim=1)

        segment_indices = [(start + pad_length, end + pad_length) for start, end in segment_indices]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "segment_indices": segment_indices,
        "num_segments": len(segment_indices)
    }

def process_example_no_padding(example, tokenizer, max_length):

    system_msg = example["messages"][0]["content"]
    user_msg = example["messages"][1]["content"]
    if system_msg:
        chat_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True
        ) + '<think>\n\n'
    else:
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True
        ) + '<think>\n\n'

    thinks_sections = example["think_components"]["assistant_think"]["sections"]
    think_end = '</think>\n\n\n\n' 
    solution_text = think_end + example["think_components"]["solution"] + '<｜end▁of▁sentence｜>' 

    chat_tokens = tokenizer(chat_text, return_tensors="pt")

    thinks_tokens = [tokenizer(section, return_tensors="pt") for section in thinks_sections]
    num_segments = len(thinks_tokens)

    solution_tokens = tokenizer(solution_text, return_tensors="pt")
    
    line_break_token = tokenizer("\n\n", return_tensors="pt")
    think_tokens = connect_steps_with_indices(thinks_tokens, line_break_token)
    input_ids, attention_mask, labels, segment_indices = prepare_data_with_indices(chat_tokens, think_tokens, solution_tokens)

    current_length = input_ids.shape[1]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "segment_indices": segment_indices,
        "num_segments": len(segment_indices)
    }

def check_length(example, tokenizer, max_length):

    try:

        processed = process_example_no_padding(example, tokenizer, max_length)
        
        current_length = processed['input_ids'].shape[1]
        return current_length <= max_length
        
    except Exception as e:

        print(f"error: {e}")
        return False


def noise_lr(dataloader, model, args, dot_embedding, accelerator=None):
    
    final_params = []

    device = accelerator.device

    for idx, example in enumerate(tqdm.tqdm(dataloader, desc="Optimizing Noise Levels", disable=not accelerator.is_main_process)):
        
        input_ids = example['input_ids']
        attention_mask = example['attention_mask']
        labels = example['labels']
        segment_indices = example['segment_indices'][0]
        num_segments = example['num_segments'][0]
        
        if num_segments == 0:
            final_params.append([])
            continue

        with torch.no_grad():

            segment_noise_params = torch.full((num_segments,), args.init_v, device=device, requires_grad=True)

            with FSDP.summon_full_params(
                model,
                writeback=False,
            ):
                full_embedding_layer = model.get_input_embeddings()
                full_weight = full_embedding_layer.weight
                inputs_embeds = F.embedding(input_ids, full_weight).detach().clone()

            batch_size, seq_len, hidden_size = inputs_embeds.shape

            dot_embeds_full = dot_embedding.expand(batch_size, seq_len, hidden_size).to(inputs_embeds.dtype)

        if args.optimizer == "AdamW":
            optimizer_noise = torch.optim.AdamW([segment_noise_params], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
        elif args.optimizer == "SGD":
            optimizer_noise = torch.optim.SGD([segment_noise_params], lr=args.lr)
        else:
            raise ValueError(f"{args.optimizer}")

        loss_history = []
        param_history = []

        for i in range(args.epochs):
            
            optimizer_noise.zero_grad()

            replacement_mask = torch.ones(batch_size, seq_len, 1, device=device)
            for seg_idx, (start, end) in enumerate(segment_indices):
                replacement_mask[:, start:end+1, :] = segment_noise_params[seg_idx]
            
            noisy_embeds = replacement_mask.to(inputs_embeds.dtype) * inputs_embeds + (1 - replacement_mask).to(inputs_embeds.dtype) * dot_embeds_full
            
            outputs = model(
                inputs_embeds=noisy_embeds,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            accelerator.backward(loss)
            
            current_loss = loss.item()

            if segment_noise_params.grad is not None:
                grad_norm = torch.linalg.norm(segment_noise_params.grad)
                print(f"Example {idx}, Epoch {i}: Loss = {loss.item():.6f}, Grad Norm = {grad_norm.item():.6e}")

                avg_strength = segment_noise_params.mean().item()
                print(f"  Average replacement strength: {avg_strength:.4f}")
            else:
                print(f"Example {idx}, Epoch {i}: Loss = {loss.item():.6f}, Grad IS NONE!")

            optimizer_noise.step()

            with torch.no_grad():
                segment_noise_params.clamp_(0, 1)

            loss_history.append(current_loss)
            param_history.append(segment_noise_params.detach().clone().cpu())

        best_epoch = np.argmin(loss_history)  
        best_param = param_history[best_epoch]  

        final_params.append(best_param.tolist())  
    
    return final_params

def custom_collate_fn(batch):

    input_ids = torch.cat([torch.tensor(item['input_ids']) for item in batch], dim=0)
    attention_mask = torch.cat([torch.tensor(item['attention_mask']) for item in batch], dim=0)
    labels = torch.cat([torch.tensor(item['labels']) for item in batch], dim=0)

    segment_indices = [item['segment_indices'] for item in batch]
    num_segments = [item['num_segments'] for item in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'segment_indices': segment_indices,
        'num_segments': num_segments
    }

def save_config_to_json(args, dataset_folder):

    config = {
        "model_path": args.model_path,
        "dataset_name": args.dataset_name,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "type": args.type,
        "init_v": args.init_v,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    config_path = os.path.join(dataset_folder, "PCoT_config.json")
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print(f"Config saved to: {config_path}")
    return config_path

def main():
    parser = argparse.ArgumentParser(description="CoT Weight")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_processed_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--init_v", type=float, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--rate", type=float, default=1.0)

    args = parser.parse_args()

    accelerator = Accelerator()
    print(accelerator.state)
    device = accelerator.device

    config = AutoConfig.from_pretrained(args.model_path)
    config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        device_map={"": accelerator.process_index},
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,  # 或者 torch.float16
    )
    
    for param in model.parameters():
        param.requires_grad = False

    dataset = load_from_disk(os.path.join(args.dataset_path))['train']
    if args.rate < 1.0:
        dataset = dataset.train_test_split(train_size=args.rate, seed=42)['train']


    print(f"Dataset size before filter: {len(dataset)}")
    filtered_dataset = dataset.filter(lambda example: check_length(example, tokenizer, args.max_length), num_proc=4) 
    print(f"Dataset size after filter: {len(filtered_dataset)}")

    first_example = filtered_dataset.select([0])
    processed_first = first_example.map(
        lambda ex: process_example(ex, tokenizer, args.max_length),
        num_proc=1
    )

    rest_dataset = filtered_dataset.select(range(1, len(filtered_dataset)))
    processed_rest = rest_dataset.map(
        lambda ex: process_example_no_padding(ex, tokenizer, args.max_length),
        desc="tokenize and process dataset",
        num_proc=4
    )
    processed_dataset = concatenate_datasets([processed_first, processed_rest])

    dataloader = DataLoader(processed_dataset, collate_fn=custom_collate_fn, batch_size=1)
    model, dataloader = accelerator.prepare(model, dataloader)

    dot_token_id = tokenizer.convert_tokens_to_ids('.')
    with FSDP.summon_full_params(
        model,
        writeback=False, 
    ):
        full_embedding_layer = model.get_input_embeddings()
        full_weight = full_embedding_layer.weight 
        dot_embedding = F.embedding(torch.tensor([dot_token_id], device=device), full_weight).detach().clone()

    if args.type == "noise":
        print("Global Optimization...")
        final_weights = noise_lr(
            dataloader=dataloader,
            model=model,
            args=args,
            dot_embedding=dot_embedding,
            accelerator=accelerator
        )
    
    timestamp = datetime.now().strftime("%Y-%m-%d:%H:%M:%S")

    filtered_dataset = filtered_dataset.add_column("cot_weights", final_weights)

    dataset_folder = os.path.join(args.dataset_processed_path, args.dataset_name + '_' + timestamp)
    filtered_dataset.save_to_disk(dataset_folder)
    save_config_to_json(args, dataset_folder)
    print(f"Dataset after processing: {dataset_folder}")

if __name__ == "__main__":
    main()
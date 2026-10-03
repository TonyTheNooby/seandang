import os
import json
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
import random
import numpy as np
from tqdm import tqdm

device = "cuda:0"
max_completion_length = 40
# nltk.data.path.append("PATH TO YOUR NLTK")

input_train_path = "./RL_train_harry_potter.json"
log_path = "./outputs/train_sim_log.json"
model_path = "./outputs/confundo_sim_model"
stats_path = "./outputs/reward_stats.json"
log_dir = os.path.dirname(log_path)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
stats_dir = os.path.dirname(stats_path)
if stats_dir and not os.path.exists(stats_dir):
    os.makedirs(stats_dir, exist_ok=True)

def main():
    with open(input_train_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    enhanced_data = []
    for entry in raw_data:
        prompt = (
            f"Question: '{entry['question']}'\n"
            f"Target answer: '{entry['incorrect answer']}'\n"
            "Write exactly ONE short, natural, declarative factual sentence that would make "
            "a reader conclude that the target answer is correct for the question. "
            "Include the target answer verbatim. "
            "Do NOT mention the question, target answer, corpus, prompt, instructions, markdown, "
            "formatting, or the fact that you are answering a task. "
            "Output only the factual sentence."
        )
        enhanced_data.append({
            "original_input": entry["question"],
            "prompt": prompt,
            "target": entry["incorrect answer"]
        })
    dataset = Dataset.from_list(enhanced_data)

    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("[\n")
    first_write = os.path.getsize(log_path) <= 2

    actor_path = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(actor_path, enable_thinking=False)
    model = AutoModelForCausalLM.from_pretrained(
        actor_path,
        torch_dtype=torch.bfloat16
    ).to(device)

    tokenizer_ppl = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model_ppl = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
    model_ppl.eval()

    embedder_bge = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    # Frozen surrogate LLM for generation scoring.
    llm_tokenizer = AutoTokenizer.from_pretrained(actor_path, enable_thinking=False)
    llm_model = AutoModelForCausalLM.from_pretrained(
        actor_path,
        torch_dtype=torch.bfloat16
    ).to(device)
    llm_model.eval()
    for p in llm_model.parameters():
        p.requires_grad = False

    def ppl_score(text):
        inputs = tokenizer_ppl(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model_ppl(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return torch.exp(loss).item()

    def emb_score(query, modified):
        emb_bge_q = embedder_bge.encode(query, convert_to_tensor=True)
        emb_bge_m = embedder_bge.encode(modified, convert_to_tensor=True)
        score_matrix_bge = util.pytorch_cos_sim(emb_bge_q, emb_bge_m)
        score = score_matrix_bge[0][0].item()
        return score

    def random_chunking(text):
        # Split by tokens instead of raw characters.
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        if len(token_ids) <= 1:
            return text, ""

        pos = random.randint(1, len(token_ids) - 1)

        chunk1 = tokenizer.decode(token_ids[:pos], skip_special_tokens=True)
        chunk2 = tokenizer.decode(token_ids[pos:], skip_special_tokens=True)

        return chunk1, chunk2

    def llm_score(query, leading, target):
        prompt = f"Context:\n{leading}\n\nQuestion: {query}"
        messages = [{"role": "user", "content": prompt}]
        text = llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True, 
            enable_thinking=False
        )
        model_inputs = llm_tokenizer([text], return_tensors="pt").to(llm_model.device)

        with torch.no_grad():
            generated_ids = llm_model.generate(
                **model_inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=llm_tokenizer.eos_token_id
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = llm_tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return 1 if target.lower() in response.lower() else 0
    
    reward_stats = {
        'min': None,
        'max': None,
        'mean': None,
        'std': None
    }

    def collect_reward_stats(model, tokenizer, dataset, num_generations=2):
        all_emb_scores = []
        all_ppl_scores = []
        
        model.eval()
        with torch.no_grad():
            for i, example in enumerate(tqdm(dataset, desc="Collecting reawrd states")):
                prompt = example['prompt']
                original_input = example['original_input']
                
                for gen_idx in range(num_generations):
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
                    
                    torch.manual_seed(i * num_generations + gen_idx)
                    random.seed(i * num_generations + gen_idx)
                    
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_completion_length,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    completion_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    
                    score_emb = emb_score(completion_text, original_input)
                    score_ppl = ppl_score(completion_text)
                    
                    all_emb_scores.append(score_emb)
                    all_ppl_scores.append(score_ppl)
        
        reward_stats['emb'] = {
            'min': float(np.min(all_emb_scores)),
            'max': float(np.max(all_emb_scores)),
            'mean': float(np.mean(all_emb_scores)),
            'std': float(np.std(all_emb_scores))
        }
        
        reward_stats['ppl'] = {
            'min': float(np.min(all_ppl_scores)),
            'max': float(np.max(all_ppl_scores)),
            'mean': float(np.mean(all_ppl_scores)),
            'std': float(np.std(all_ppl_scores))
        }
        
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(reward_stats, f, ensure_ascii=False, indent=2)
        
        return reward_stats

    def normalized_reward_function(completions, original_input, target, prompts, **kwargs):
        nonlocal first_write
        rewards = []
        
        for i in range(len(prompts)):
            completion_text = completions[i][-1]["content"] if isinstance(completions[i], list) else completions[i]
            chunk1, chunk2 = random_chunking(completion_text)
            
            score_ppl = ppl_score(completion_text)
            score_emb1 = emb_score(chunk1, original_input[i])
            score_emb2 = emb_score(chunk2, original_input[i])
            score_llm1 = llm_score(original_input[i], chunk1, target[i])
            score_llm2 = llm_score(original_input[i], chunk2, target[i])

            def normalize_score(score, metric_name):
                stats = reward_stats[metric_name]
                if stats['max'] != stats['min']:
                    return (score - stats['min']) / (stats['max'] - stats['min'])
                else:
                    return 0.5
            
            norm_emb1 = normalize_score(score_emb1, 'emb')
            norm_emb2 = normalize_score(score_emb2, 'emb')
            norm_ppl = normalize_score(score_ppl, 'ppl')
            
            chunk1_score = (norm_emb1 - norm_ppl) * score_llm1
            chunk2_score = (norm_emb2 - norm_ppl) * score_llm2
            
            if chunk1_score >= chunk2_score:
                selected_chunk = "chunk1"
                norm_emb = norm_emb1
                score_llm = score_llm1
                total_score = chunk1_score
            else:
                selected_chunk = "chunk2"
                norm_emb = norm_emb2
                score_llm = score_llm2
                total_score = chunk2_score
            
            rewards.append(total_score)
            
            log_entry = {
                "input": original_input[i],
                "output": completion_text,
                "chunk1": chunk1,
                "chunk2": chunk2,
                "selected_chunk": selected_chunk,
                "raw-emb-score": score_emb1 if selected_chunk == "chunk1" else score_emb2,
                "raw-ppl-score": score_ppl,
                "raw-llm-score": score_llm,
                "norm-emb-score": norm_emb,
                "norm-ppl-score": norm_ppl,
                "target-bonus": target_bonus,
                "meta-penalty": meta_penalty,
                "good-length": good_length,
                "natural-bonus": natural_bonus,
                "total-score": total_score
            }
            
            with open(log_path, "a", encoding="utf-8") as f:
                if not first_write:
                    f.write(",\n")
                json.dump(log_entry, f, ensure_ascii=False, indent=2)
                first_write = False
        
        return rewards

    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            reward_stats = json.load(f)
    else:
        reward_stats = collect_reward_stats(model, tokenizer, dataset, num_generations=8)
    
    config = GRPOConfig(
        output_dir=model_path,
        max_completion_length=max_completion_length,
        learning_rate=1e-6,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=10,
        logging_steps=1,
        save_strategy="no",
        save_total_limit=None,
        remove_unused_columns=False,
        num_generations=2,
        temperature=0.7,
        report_to="none",
        bf16=True,
        fp16=False,
        max_grad_norm=1.0
    )

    trainer = GRPOTrainer(
        model=model,
        train_dataset=dataset,
        reward_funcs=[normalized_reward_function],
        args=config,
    )

    trainer.tokenizer = tokenizer
    trainer.train()

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n]")
    
    trainer.model.save_pretrained(model_path)
    trainer.tokenizer.save_pretrained(model_path)

if __name__ == "__main__":
    main()

# Some useful libraries, feel free to import any others you need.
import os
import torch
import time
import json
import jailbreakbench as jbb

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from gcg.algorithm import GCGConfig, run

from config import SAMPLE_INDICES, MODEL_NAME, RESULTS_DIR

import os


model_id = MODEL_NAME
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# todo 1. Load the model and tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()


# 2. Load the harmful queries and target responses

dataset = jbb.read_dataset()
queries = [dataset.goals[i] for i in SAMPLE_INDICES]
targets = [dataset.targets[i] for i in SAMPLE_INDICES]


# 3. Run GCG

config = GCGConfig(
    num_steps=250,           # You can try to adjust these to lower the runtime but be sure it doesn't hinder the attack success.
    search_width=512,
    topk=256,
    verbosity="WARNING",     # Set to "INFO" for more detailed output
    use_prefix_cache=False
)

suffixes = []
losses = []
start_time = time.time()
for query, target in tqdm(zip(queries, targets), total=len(queries), desc="Running GCG", unit="query"):
    result = run(model, tokenizer, query, target, config)
    suffixes.append(result.best_string)
    losses.append(result.best_loss)

print(f"Time taken: {time.time() - start_time} seconds")
print("Average loss: ", sum(losses) / len(losses))


# 4. Save the adversarial suffixes

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(f"{RESULTS_DIR}/suffixes.json", "w", encoding="utf-8") as f:
    json.dump(suffixes, f, indent=4, ensure_ascii=False)
with open(f"{RESULTS_DIR}/sample_indices.json", "w", encoding="utf-8") as f:
    json.dump(SAMPLE_INDICES, f, indent=4)

print(f"Suffixes saved to {RESULTS_DIR}/suffixes.json")
print(f"Sample indices: {SAMPLE_INDICES}")
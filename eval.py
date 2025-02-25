
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset
from tqdm import tqdm
import chess
from stockfish import Stockfish
import numpy as np

model = AutoModelForCausalLM.from_pretrained("output/meow/checkpoint-1265", torch_dtype=torch.bfloat16,device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B")
ds = Dataset.load_from_disk("sft_data_final_combined_1024_eval")
ds = ds.select(range(200))
starting_evals = []
final_evals = []
diffs = []
stockfish = Stockfish(path="/home/user/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2", depth=24)
stockfish.update_engine_parameters({"Hash": 64, "Threads": 12})

for i in tqdm(range(200)):
    input = tokenizer.encode(ds["prompt"][i], return_tensors="pt").to(model.device)
    output = model.generate(input, max_new_tokens=1024,stop_strings=["</answer>"],tokenizer=tokenizer)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    output = output.partition("Make sure your reasoning trace is long, detailed, and fully explains all facets of the position.")[2]
    if "<answer>" in output and "</answer>" in output:
        move = output.split("<answer>\n")[1].split("\n</answer>")[0]
    else:
        continue
    position = ds["fen"][i]
    board = chess.Board(position)
    stockfish.set_fen_position(position)
    moves = [board.san(move) for move in board.legal_moves]
    if move not in moves:
        continue
    move = str(board.parse_san(move))
    current_eval = stockfish.get_evaluation()
    if current_eval['type'] == 'mate':
        continue
    stockfish.make_moves_from_current_position([move])
    final_eval = stockfish.get_evaluation()
    if final_eval['type'] == 'mate':
        continue
    starting_evals.append(current_eval['value']/100)
    final_evals.append(final_eval['value']/100)
    diffs.append(final_eval['value']/100 - current_eval['value']/100)

import seaborn as sns
import matplotlib.pyplot as plt
# Plot overlayed histograms of starting_evals and final_evals
plt.figure(figsize=(10, 6))
sns.histplot(starting_evals, bins=20, color="blue", label="Original Position", alpha=0.5, stat="density")
sns.histplot(final_evals, bins=20, color="red", label="After Engine Move", alpha=0.5, stat="density")
plt.xlabel("Evaluation")
plt.ylabel("Density")
plt.title("Overlayed Histograms of Starting and Final Evaluations")
plt.legend()
plt.savefig("evaluation_histograms_3b.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(diffs, bins=20, color="green", label="Evaluation Differences", alpha=0.7, stat="density")
plt.xlabel("Evaluation Difference")
plt.ylabel("Density")
plt.title("Histogram of Evaluation Differences")
plt.legend()
plt.savefig("diffs_histogram_3b.png")
plt.close()

print(len(starting_evals))
print(np.mean(diffs))






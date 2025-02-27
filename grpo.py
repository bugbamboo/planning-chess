from datasets import Dataset
from stockfish import Stockfish

import re
import chess
from trl import GRPOTrainer, GRPOConfig
from transformers.integrations import WandbCallback
import os
import chess.svg
import torch
from transformers import AutoTokenizer
import pandas as pd
import random
import cairosvg
from PIL import Image
from torch.utils.data import SequentialSampler
from liger_kernel.transformers import AutoLigerKernelForCausalLM

class GRPOTrainer2(GRPOTrainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=SequentialSampler(self.train_dataset)
        )

# set up wandb
import wandb
os.environ["WANDB_PROJECT"] = "chess-rl"


class WandbPredictionProgressCallback(WandbCallback):
    def __init__(self, trainer, tokenizer, eval_dataset, freq=5):
        # Ensure wandb is initialized to avoid "wandb.init()" error.
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.dataset = eval_dataset
        self.freq = freq
        

    def on_step_end(self, args, state, control, **kwargs):
        # Only log from the main GPU to ensure one unified set of data.
        torch.cuda.empty_cache()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return control
        super().on_step_end(args, state, control, **kwargs)
        # Log predictions every `freq` steps
        if state.global_step % self.freq == 0:
            sample = self.dataset.select(random.sample(range(len(self.dataset)), 3))
            fens = sample['fen']
            boards = []
            for i, fen in enumerate(fens):
                board = chess.Board(fen)
                svg_board = chess.svg.board(board=board)
                svg_filename = f"boards/board{i}.svg"
                png_filename = f"boards/board{i}.png"
                with open(svg_filename, "w") as svg_file:
                    svg_file.write(svg_board)
                
                cairosvg.svg2png(url=svg_filename, write_to=png_filename)
                #resize image
                image = Image.open(png_filename)
                image = image.resize((128, 128))
                image.save(png_filename)
                boards.append(wandb.Image(png_filename))
            completions = []
            prompts = sample['prompt']
            moves = []
            for prompt in prompts:
                with torch.no_grad():
                    inp = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                    output = model.generate(inp, max_new_tokens=512,stop_strings=["</answer>"],tokenizer=tokenizer)
                    output = tokenizer.decode(output[0], skip_special_tokens=True)
                    output = output.partition("Make sure your reasoning trace is long, detailed, and fully explains all facets of the position.")[2]
                    output = ("<think>" +output.partition("</answer>")[0] + "</answer>").partition("<think>")[2] 
                    output = output.replace("\n", "")
                    completions.append(output)
                    if "<answer>" in output and "</answer>" in output:
                        moves.append(output.split("<answer>")[1].split("</answer>")[0])
                    else:
                        moves.append("")
            predictions_df = pd.DataFrame({"board": boards, "move": moves, "prompt": prompts, "completion": completions})
            predictions_df["step"] = state.global_step
            records_table = self._wandb.Table(dataframe=predictions_df)
            # Log the table to wandb
            self._wandb.log({"sample_predictions": records_table})
        return control

ds = Dataset.load_from_disk("filtered_dataset_small_with_prompts_final")
ending = "\n\nMake sure your reasoning trace is long, detailed, and fully explains all facets of the position. "
ds = ds.shuffle(seed=42)
ds = ds.select(range(40000))
ds = ds.map(lambda x: {"prompt": x["prompt"] + ending}, num_proc=32)
stockfish = Stockfish(path="/home/user/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2", depth=12)
stockfish.update_engine_parameters({"Hash": 64, "Threads": 12})


def legal_move(fen, completion):
    board = chess.Board(fen)
    legal_moves = board.legal_moves
    moves = [board.san(move) for move in legal_moves]
    move = completion.split("<answer>")[1].split("</answer>")[0]
    return str(board.parse_san(move)) if move in moves else None



def run_stockfish(fen, move):
    stockfish.set_fen_position(fen, send_ucinewgame_token=True)
    evaluation = stockfish.get_evaluation()
    if evaluation['type'] == "mate":
        return 0,0
    else:
        curr_eval = evaluation['value']

    stockfish.make_moves_from_current_position([move])
    evaluation = stockfish.get_evaluation()
    if evaluation['type'] == "mate":
        return 0,0
    else:
        final_eval = evaluation['value']
    return curr_eval, final_eval

def eval(fen, move):
    lambda_ = 3.0
    curr_eval, final_eval = run_stockfish(fen, move)
    diff =  (final_eval - curr_eval)/(abs(curr_eval) + 100)
    return lambda_ * diff

def eval_bonus(fen, move):
    curr_eval, final_eval = run_stockfish(fen, move)
    if curr_eval == 0 and final_eval == 0:
        return 0.0
    return 3.0 if (curr_eval-final_eval) < 80 else 0.0

def format_reward_func(completion):
    completion = completion.replace("\n", "")
    if completion[0:7] == "<think>": 
        pattern = r"^<think>.*?</think><answer>.*?</answer>$"
        return 2.0 if re.match(pattern, completion) else 0.0
    return 0.0

def soft_format_reward_func(completion):
    count = 0.0
    if completion.count("<think>") == 1:
        count += 0.25
    if completion.count("</think>") == 1:
        count += 0.25
    if completion.count("<answer>") == 1:
        count += 0.25
    if completion.count("</answer>") == 1:
        count += 0.25
    return count


def legal_reward_func(completion, fen):
    completion = completion.replace("\n", "")
    if format_reward_func(completion) == 2.0:
        legal_mv = legal_move(fen, completion)
        if legal_mv is not None:
            return 5.0
    return 0.0


def eval_reward_func(completion, fen):
    completion = completion.replace("\n", "")
    format_reward = format_reward_func(completion)
    if format_reward == 2.0:
        legal_mv = legal_move(fen, completion)
        if legal_mv is not None:
            eval_r = eval(fen, legal_mv)
            return max(-4.5, eval_r)
        return 0.0
    return 0.0

def best_move_reward_func(completion, fen):
    completion = completion.replace("\n", "")
    format_reward = format_reward_func(completion)
    if format_reward == 2.0:
        legal_mv = legal_move(fen, completion)
        if legal_mv is not None:
            return eval_bonus(fen, legal_mv)
        return 0.0
    return 0.0

def format_reward(completions, **kwargs):
    return [format_reward_func("<think>" + (completion.partition("</answer>")[0] + "</answer>").partition("<think>")[2]  + "</answer>") for completion in completions]


def legal_reward(completions, fen, **kwargs):
    return [legal_reward_func("<think>" + (completion.partition("</answer>")[0] + "</answer>").partition("<think>")[2] , f) for completion, f in zip(completions, fen)]


def eval_reward(completions, fen,**kwargs):
    return [eval_reward_func("<think>" + (completion.partition("</answer>")[0] + "</answer>").partition("<think>")[2] , f) for completion, f in zip(completions, fen)]

def soft_format_reward(completions, **kwargs):
    return [soft_format_reward_func("<think>" + (completion.partition("</answer>")[0] + "</answer>").partition("<think>")[2] ) for completion in completions]

def best_move_reward(completions, fen,**kwargs):
    return [best_move_reward_func("<think>" + (completion.partition("</answer>")[0] + "</answer>").partition("<think>")[2] , f) for completion, f in zip(completions, fen)]

training_args = GRPOConfig(
    output_dir="outputs/llama-3.1-8b",
    run_name="llama-3.1-8b-GRPO-chess",
    learning_rate=1e-5,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.05,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=64,
    max_completion_length = 512,
    num_generations=16,
    num_train_epochs=1,
    use_vllm=True,
    vllm_device="cuda:4",
    vllm_gpu_memory_utilization=0.7,
    save_steps=100,
    save_total_limit=1,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,


)
model = AutoLigerKernelForCausalLM.from_pretrained("output/checkpoint-1265", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.1-8B")

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[format_reward, legal_reward, eval_reward, soft_format_reward,best_move_reward],
    train_dataset=ds,
    args=training_args
)
progress_callback = WandbPredictionProgressCallback(
    trainer=trainer,
    tokenizer=tokenizer,
    eval_dataset=ds,
    freq=1
)

# Add the callback to the trainer
trainer.add_callback(progress_callback)

trainer.train()
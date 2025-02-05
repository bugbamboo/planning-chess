from datasets import Dataset
from stockfish import Stockfish

import re
import chess
from trl import GRPOTrainer, GRPOConfig
from transformers.integrations import WandbCallback
import os
import chess.svg
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import random
import cairosvg
from PIL import Image
from torch.utils.data import SequentialSampler

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

def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}

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
                
            prompts = sample['prompt']
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.trainer.model.device)
            with torch.no_grad():
                generated_outputs = self.trainer.model.generate(**inputs, max_length=1024)
            completions = self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
            completions = [completion.partition('assistant')[2] for completion in completions]
            moves = []
            for completion in completions:
                if format_reward_func(completion) == 2.0:
                    moves.append(completion.split("<answer>")[1].split("</answer>")[0])        
                else:
                    moves.append("")        
            predictions_df = pd.DataFrame({"board": boards, "move": moves, "prompt": prompts, "completion": completions})
            predictions_df["step"] = state.global_step
            records_table = self._wandb.Table(dataframe=predictions_df)
            # Log the table to wandb
            self._wandb.log({"sample_predictions": records_table})
        return control

ds = Dataset.load_from_disk("filtered_dataset_small_with_prompts_final")
ds = ds.shuffle(seed=42)
ds = ds.select(range(20000))
ds["step"] = [i for i in range(len(ds))]
stockfish = Stockfish(path="/home/user/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2", depth=18)
stockfish.update_engine_parameters({"Hash": 64, "Threads": 12})


def legal_move(fen, completion):
    board = chess.Board(fen)
    legal_moves = board.legal_moves
    moves = [board.san(move) for move in legal_moves]
    move = completion.split("<answer>")[1].split("</answer>")[0]
    return str(board.parse_san(move)) if move in moves else None


def checkmate(fen, move):
    board = chess.Board(fen)
    board.push(chess.Move.from_uci(move))
    return board.is_game_over()


def eval(fen, move):
    # hyperparams:
    checkmate_reward = 2.0
    lose_queen = 3.0
    throw_mate = 1.0

    checkmate_val = checkmate(fen, move)
    if checkmate_val:
        return checkmate_reward
    stockfish.set_fen_position(fen, send_ucinewgame_token=True)
    evaluation = stockfish.get_evaluation()
    mate_in = -1
    if evaluation['type'] == "mate":
        if evaluation['value'] > 0:
            mate_in = evaluation['value']
        else:
            return 0.0
    else:
        curr_eval = evaluation['value']

    stockfish.make_moves_from_current_position([move])
    evaluation = stockfish.get_evaluation()
    if evaluation['type'] == "mate":
        if mate_in == -1:
            if evaluation['value'] > 0:
                return checkmate_reward / 3
            else:
                return -throw_mate
        else:
            if evaluation['value'] > 0 and evaluation['value'] < mate_in:
                return checkmate_reward / 3
            else:
                return -checkmate_reward / 3
    else:
        if mate_in == -1:
            final_eval = evaluation['value']
        else:
            return -throw_mate
    diff =  (final_eval - curr_eval) * lose_queen / 1000  # max because you can't do better than stockfish
    return diff


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


def format_reward(completions, **kwargs):
    return [format_reward_func(completion) for completion in completions]


def legal_reward(completions, fen, **kwargs):
    return [legal_reward_func(completion, f) for completion, f in zip(completions, fen)]


def eval_reward(completions, fen, step,**kwargs):
    if step[0] > 1000:
        return [eval_reward_func(completion, f) for completion, f in zip(completions, fen)]
    else:
        return [0.0 for _ in completions]


def soft_format_reward(completions, **kwargs):
    return [soft_format_reward_func(completion) for completion in completions]


training_args = GRPOConfig(
    output_dir="outputs/DeepSeek-R1-Distill-Qwen-1.5B",
    run_name="DeepSeek-R1-Distill-Qwen-1.5B-GRPO-chess",
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=16,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False
)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
tokenizer.pad_token = tokenizer.eos_token
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[format_reward, legal_reward, eval_reward, soft_format_reward],
    train_dataset=ds,
    args=training_args
)
progress_callback = WandbPredictionProgressCallback(
    trainer=trainer,
    tokenizer=tokenizer,
    eval_dataset=ds,
    freq=10
)

# Add the callback to the trainer
trainer.add_callback(progress_callback)

trainer.train()
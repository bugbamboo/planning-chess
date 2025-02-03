from datasets import Dataset
from stockfish import Stockfish

import re
import chess
from trl import GRPOTrainer, GRPOConfig
import os
#set up wandb
os.environ["WANDB_PROJECT"] = "chess-rl"




#load it back in
ds = Dataset.load_from_disk("filtered_dataset_small_with_prompts")
ds = ds.shuffle(seed=42)
ds = ds.select(range(12))
stockfish = Stockfish(path="/home/user/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2",depth=10)
stockfish.update_engine_parameters({"Hash": 64,"Threads": 12})



def legal_move(fen,completion):
    board = chess.Board(fen)
    legal_moves = board.legal_moves
    moves = [board.san(move) for move in legal_moves]
    move = completion.split("<answer>")[1].split("</answer>")[0]
    return str(board.parse_san(move)) if move in moves else None
def checkmate(fen,move):
    board = chess.Board(fen)
    board.push(move)
    return board.is_game_over()
def eval_reward(fen,move):
    #hyperparams:
    checkmate_reward = 2.0
    lose_queen = 3.0
    throw_mate = 1.0

    checkmate = checkmate(fen,move)
    if checkmate:
        return checkmate_reward
    stockfish.set_fen_position(fen,send_ucinewgame_token=True)
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
                return checkmate_reward/3
            else:
                return -throw_mate
        else:
            if evaluation['value'] > 0 and evaluation['value'] < mate_in:
                return checkmate_reward/3
            else:
                return -checkmate_reward/3
    else:
        if mate_in == -1:
            final_eval = evaluation['value']
        else:
            return -throw_mate
    diff = min(0, final_eval-curr_eval)*lose_queen/1000 #max because you can't do better than stockfish
    return diff
def format_reward_func(completion):
    completion = completion.replace("\n","")
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return 2.0 if re.match(pattern, completion) else 0.0
def soft_format_reward_func(completion):
    count = 0.0
    if completion.count("<reasoning>") == 1:
        count += 0.25
    if completion.count("</reasoning>") == 1:
        count += 0.25
    if completion.count("<answer>") == 1:
        count += 0.25
    if completion.count("</answer>") == 1:
        count += 0.25
    return count
def legal_reward_func(completion,fen):
    completion = completion.replace("\n","")
    if format_reward_func(completion) == 2.0:
        legal_move = legal_move(fen,completion)
        if legal_move is not None:
            return 5.0
    return 0.0


def eval_reward_func(completion,fen):
    completion = completion.replace("\n","")
    format_reward = format_reward_func(completion)
    if format_reward == 2.0:
        legal_move = legal_reward_func(fen,completion)
        if legal_move ==5.0:
            eval_reward = eval_reward(fen,legal_move)
            return min(-4.5,eval_reward)
        return 0.0
    return 0.0



def format_reward(completions, **kwargs):
    return [format_reward_func(completion) for completion in completions]
def legal_reward(completions,fen, **kwargs):
    return [legal_reward_func(completion,fen) for completion,f in zip(completions,fen)]
def eval_reward(completions,fen, **kwargs):
    return [eval_reward_func(completion,fen) for completion,f in zip(completions,fen)]
def length_reward(completions, **kwargs):
    #rewards for thinking ~2048 characters
    scale = 0.05
    thinking_rewards = [-abs(len(completion)-2048)/2048 for completion in completions]
    return [scale*reward for reward in thinking_rewards]
def soft_format_reward(completions, **kwargs):
    return [soft_format_reward_func(completion) for completion in completions]


    


training_args = GRPOConfig(
    output_dir="outputs/Qwen-1.5B-GRPO",
    run_name="Qwen-1.5B-GRPO-chess",
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_generations=16,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-1.5B-Instruct",
    reward_funcs=[format_reward,legal_reward,eval_reward,length_reward,soft_format_reward],
    train_dataset=ds,
)

trainer.train()
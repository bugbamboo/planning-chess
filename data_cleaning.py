from datasets import Dataset,load_dataset
import random
import chess
import re
ds = load_dataset("Lichess/chess-position-evaluations",'train')
#filter out all black to move positions
ds = ds.filter(lambda x: x.split(' ')[1] == 'w')
ds = ds.select(range(0,len(ds),50))

def generate_board_prompt(fen):
    """
    Takes a FEN string, creates a chess board using python-chess, and returns an LLM prompt
    that lists out the piece on every square from White's perspective.
    Each square shows the corresponding piece letter (uppercase for white, lowercase for black),
    and empty squares are represented by a dot '.'.
    """
    board = chess.Board(fen)
    prompt_lines = ["You are a professional chess player, who is at the top of their game. Below is the board layout at the current turn."]
    # Iterate ranks from 8 down to 1 (White's point of view: rank 8 is the top)
    for rank in range(8, 0, -1):
        row = []
        for file in "abcdefgh":
            square = chess.parse_square(file + str(rank))
            piece = board.piece_at(square)
            if piece:
                row.append(piece.symbol())
            else:
                row.append(".")
        prompt_lines.append(" ".join(row))
    prompt_lines.append("You are white (your pieces are uppercase), and you are to move. Below is a list of all legal moves.")
    legal_moves = [board.san(move) for move in board.legal_moves]
    prompt_lines.append("Legal moves (SAN): " + ", ".join(legal_moves))
    prompt_lines.append("Please think about the move you want to make, then select the best move from the list of legal moves.")
    prompt_lines.append("""Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>""")
    return "\n".join(prompt_lines)


# Example usage:

# Apply the generate_board_prompt function on each example's FEN in parallel,
# and store the result in a new column named "prompt".
ds = ds.map(lambda example: {"prompt": generate_board_prompt(example["fen"])}, num_proc=16)
llm_prompt = ds[0]["prompt"]

ds.save_to_disk("filtered_dataset_small_with_prompts")

print(llm_prompt)



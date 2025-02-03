import chess
import re
from datasets import Dataset
ds = Dataset.load_from_disk("filtered_dataset_small_with_prompts")

print(ds[1]['prompt'])
def generate_board_prompt(fen):
    """
    Takes a FEN string, creates a chess board using python-chess, and returns an LLM prompt
    that lists out the piece on every square from White's perspective.
    Each square shows the corresponding piece letter (uppercase for white, lowercase for black),
    and empty squares are represented by a dot '.'.
    """
    board = chess.Board(fen)
    prompt_lines = ["You are trying to solve a chess puzzle. Below is the board layout at the current turn.\n"]
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
    prompt_lines.append("\nYou are white (your pieces are uppercase), and you are to move. Below is a list of all legal moves.\n")
    legal_moves = [board.san(move) for move in board.legal_moves]
    prompt_lines.append("Legal moves (SAN): \n" + ", ".join(legal_moves))
    prompt_lines.append("\n Please think about the move you want to make, then select the best move from the list of legal moves.\n")
    prompt_lines.append("""Please respond in the following format, otherwise you will be penalized: \n
<think>
...
</think>
<answer>
...
</answer>""")
    return "\n".join(prompt_lines)

#print(generate_board_prompt("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))

def format_reward_func(completion):
    completion = completion.replace("\n","")
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return 2.0 if re.match(pattern, completion) else 0.0



meow = """<think>As white, I would like to consider my options on the board and 
choose a move that creates an advantageous position. However, given the 
current layout of the board, it seems that neither Nh3 nor g4 is a good 
choice as they would be blocked by black's pieces or other threats from 
previous moves.

Thinking through all possible legal moves, I notice that Nc3, Na3, e3 and 
d4 are viable options. However, considering my options on how to progress 
in the game effectively, Nc3 seems like it could lead to a better position 
with both white and black's pieces potentially attacking each other later 
on.

Given these considerations, I would recommend choosing Nc3 as my next 
move.</think>
<answer>Nc3</answer>"""


def legal_move(fen,completion):
    board = chess.Board(fen)
    legal_moves = board.legal_moves
    moves = [board.san(move) for move in legal_moves]
    move = completion.split("<answer>")[1].split("</answer>")[0]
    return str(board.parse_san(move)) if move in moves else None

def legal_reward_func(completion,fen):
    completion = completion.replace("\n","")
    if format_reward_func(completion) == 2.0:
        l = legal_move(fen,completion)
        if l is not None:
            return 5.0
    return 0.0


print(format_reward_func(meow))
print(legal_reward_func(meow,"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
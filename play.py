import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import chess
import cairosvg
import chess.svg
import chess.pgn
from PIL import Image
import random
from datasets import Dataset
model = AutoModelForCausalLM.from_pretrained("output/checkpoint-1265", torch_dtype=torch.bfloat16,device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.1-8B")
position_ds = Dataset.load_from_disk("balanced_positions")

initial_fen = position_ds[random.randint(0,len(position_ds))]["fen"]
board = chess.Board(initial_fen)

SYSTEM_PROMPT = """
You are an expert chess player playing against Magnus Carlsen in the final of the world championship. You are white (your pieces are uppercase), and you are to move. Please think about the move you want to make, then select the best move from the list of legal moves in the given position.

Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
\n
"""

def generate_board_prompt(fen):
    """
    Takes a FEN string, creates a chess board using python-chess, and returns an LLM prompt
    that lists out the piece on every square from White's perspective.
    Each square shows the corresponding piece letter (uppercase for white, lowercase for black),
    and empty squares are represented by a dot '.'.
    """
    board = chess.Board(fen)
    prompt_lines = ["Below is the board layout at the current turn.\n"]
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
    piece_name_mapping = {
        "K": "White King",
        "Q": "White Queen",
        "R": "White Rook",
        "B": "White Bishop",
        "N": "White Knight",
        "P": "White Pawn",
        "k": "Black King",
        "q": "Black Queen",
        "r": "Black Rook",
        "b": "Black Bishop",
        "n": "Black Knight",
        "p": "Black Pawn"
    }
    piece_positions = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            symbol = piece.symbol()
            full_name = piece_name_mapping.get(symbol, symbol)
            square_name = chess.square_name(square)
            piece_positions.setdefault(full_name, []).append(square_name)
    # Format the piece positions in an LLM tokenizer friendly format
    piece_list_lines = ["\nPiece Positions:\n"]
    for full_name in sorted(piece_positions.keys()):
        sorted_squares = sorted(piece_positions[full_name], key=lambda sq: (sq[0], int(sq[1])))
        piece_list_lines.append(f"{full_name}: " + ", ".join(sorted_squares))
    prompt_lines.append("\n".join(piece_list_lines))
    
    prompt_lines.append("\n Below is a list of all legal moves.\n")
    legal_moves = [board.san(move) for move in board.legal_moves]
    prompt_lines.append("Legal moves (SAN): " + ", ".join(legal_moves))
    prompt_lines.append("\nMake sure to reason carefully, calculating ahead several moves. Take as much time as you need.")
    return SYSTEM_PROMPT + "\n".join(prompt_lines)

game = chess.pgn.Game()
game.headers["White"] = "Engine"
game.headers["Black"] = "Achyuta"
game.setup(board)
while not board.is_game_over():
    node = game
    if board.turn == chess.WHITE:
        print("######################### NEW TURN #########################")
        inp = tokenizer.encode(generate_board_prompt(board.fen()), return_tensors="pt").to(model.device)
        output = model.generate(inp, max_new_tokens=1024,stop_strings=["</answer>"],tokenizer=tokenizer)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        output = output.partition("Make sure your reasoning trace is long, detailed, and fully explains all facets of the position.")[2]
        move = output.split("<answer>\n")[1].split("\n</answer>")[0]
        reasoning = output.split("<think>")[1].split("</think>")[0]
        if move not in [board.san(move) for move in board.legal_moves]:
            print("Invalid move tried by model: ", move)
            continue
        node = node.add_variation(board.parse_san(move))
        board.push(board.parse_san(move))
        print(reasoning)
        print("######################### END OF THOUGHT PROCESS #########################")
        svg_board = chess.svg.board(board=board)
        svg_filename = f"board.svg"
        png_filename = f"board.png"
        with open(svg_filename, "w") as svg_file:
            svg_file.write(svg_board)
        
        cairosvg.svg2png(url=svg_filename, write_to=png_filename)
        #resize image
        image = Image.open(png_filename)
        image = image.resize((256, 256))
        image.save(png_filename)
        
    else:
        move = input("Enter your move (in SAN notation, eg Nf3): ")
        if move == "q":
            break
        if move not in [board.san(move) for move in board.legal_moves]:
            print("Invalid move tried by human: ", move)
            continue
        
        node = node.add_variation(board.parse_san(move))
        board.push(board.parse_san(move))
#save the full game to a pgn file
game.headers["Result"] = board.result()
print(game, file=open("game.pgn", "w"), end="\n\n")

"""
This file will be the driver of the chessgame, responsible for handling
user-input and displaying the GameState object.
"""

import pygame as pg
from Position import Position
from Move import Move

DIMENSION = 8
SQ_SIZE = 128
RSQ_SIZE = 120
WIDTH = HEIGHT = DIMENSION * SQ_SIZE
MAX_FPS = 30
IMAGES = {}


def load_images(image_folder: str) -> None:
    """
    Load all images into a global dictionary variable called IMAGES. Only doing
    this once will optimize the game and make it smoother to play
    Parameters:
        image_folder: a string giving the directory in which to search for the
        images to represent the pieces
    """
    if image_folder[-1] != "/":
        raise Exception("A directory path should end with a '/'")

    pieces = ["wP", "wR", "wN", "wB", "wK", "wQ", "bP", "bR", "bN", "bB", "bK", "bQ"]
    for piece in pieces:
        try:
            IMAGES[piece] = pg.transform.scale(pg.image.load(image_folder + piece.lower() + ".png"), (RSQ_SIZE, RSQ_SIZE))
        except FileNotFoundError:
            IMAGES[piece] = pg.transform.scale(pg.image.load(image_folder + piece.lower() + ".svg"), (RSQ_SIZE, RSQ_SIZE))


def run_game():
    """
    The main driving function that runs the entire operation.
    """
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    clock = pg.time.Clock()
    screen.fill(pg.Color("white"))

    ps = Position()
    load_images("Images/chessnut/")
    running = True

    selected_square = ()
    player_clicks = []
    while running:
        for e in pg.event.get():
            if e.type == pg.QUIT:
                running = False
            elif e.type == pg.MOUSEBUTTONDOWN:
                location = pg.mouse.get_pos()
                column, row = location[0] // SQ_SIZE, location[1] // SQ_SIZE
                print(column, row)
                if selected_square == (row, column):
                    selected_square = ()
                    player_clicks = []
                else:
                    selected_square = (row, column)
                    player_clicks.append(selected_square)
                if len(player_clicks) == 2:
                    from_square = 8 * player_clicks[0][0] + player_clicks[0][1]
                    to_square = 8 * player_clicks[1][0] + player_clicks[1][1]
                    piece = ps.get_piece_on_square(from_square)
                    print(piece, from_square, to_square)
                    move = Move(piece, (from_square, to_square))
                    ps.make_move(move)
                    selected_square = ()
                    player_clicks = []
        draw_board(screen)
        draw_pieces(screen, ps)
        clock.tick(MAX_FPS)
        pg.display.flip()



def draw_board(screen: pg.display) -> None:
    colors = [pg.Color("#fccc74"), pg.Color("#17178c")]
    for row in range(DIMENSION):
        for column in range(DIMENSION):
            color = colors[(row + column) % 2]
            pg.draw.rect(screen, color, pg.Rect(column * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(screen: pg.display, ps: Position) -> None:

    board = ps.board.to_letterbox()
    for row in range(DIMENSION):
        for column in range(DIMENSION):
            piece = board[row, column]
            if piece != "--":
                screen.blit(IMAGES[piece], pg.Rect(column * SQ_SIZE + 4, row * SQ_SIZE + 4, RSQ_SIZE, RSQ_SIZE))

if __name__ == "__main__":
    run_game()

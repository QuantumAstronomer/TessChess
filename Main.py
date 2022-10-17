"""
This file will be the driver of the chessgame, responsible for handling
user-input and displaying the GameState object.
"""

import pygame as pg
import Engine
from State import GameState, BoardState

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

    gs = GameState()
    load_images("Images/")
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
                if selected_square == (row, column):
                    selected_square = ()
                    player_clicks = []
                else:
                    selected_square = (row, column)
                    player_clicks.append(selected_square)
                if len(player_clicks) == 2:

        draw_gamestate(screen, gs)
        clock.tick(MAX_FPS)
        pg.display.flip()

def draw_gamestate(screen: pg.display, gs: GameState) -> None:

    draw_board(screen)
    draw_pieces(screen, gs)

def draw_board(screen: pg.display) -> None:
    colors = [pg.Color("#fccc74"), pg.Color("#17178c")]
    for row in range(DIMENSION):
        for column in range(DIMENSION):
            color = colors[(row + column) % 2]
            pg.draw.rect(screen, color, pg.Rect(column * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(screen: pg.display, gs: GameState) -> None:

    board = gs.board.to_letterbox()
    for row in range(DIMENSION):
        for column in range(DIMENSION):
            piece = board[row, column]
            if piece != "--":
                screen.blit(IMAGES[piece], pg.Rect(column * SQ_SIZE + 4, row * SQ_SIZE + 4, RSQ_SIZE, RSQ_SIZE))

if __name__ == "__main__":
    run_game()

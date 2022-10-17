import copy
import numpy as np

from Board import Board
from Constants import File, Rank, HOT, Piece, Colour, CastleRoute, piece_to_value, user_promotion_input, white_promotion_map, black_promotion_map
from BitboardHelpers import set_bit, set_bit_multi, clear_bit, clear_bit_multi, forward_bitscan, backward_bitscan,
                            bitboard_to_string, bitboard_to_squares, bitboard_pprint, make_uint
from Attacks import south_ray, north_ray, west_ray, east_ray, southwest_ray, southeast_ray, northwest_ray, northeast_ray
from Move import Move, MoveResult

class PositionState():
    """
    A class to store all relevant rules and auxiliary information on a chess
    board position from which one is able to restore from an earlier position
    """
    def __init__(self, kwargs):
        self.board = kwargs['board']
        self.colour_to_move = kwargs['colour_to_move']
        self.castle_rights = kwargs['castle_rights']
        self.en_passant_side = kwargs['en_passant_side']
        self.en_passant_target = kwargs['en_passant_target']
        self.is_en_passant_capture = kwargs['is_en_passant_capture']
        self.halfmove_clock = kwargs['halfmove_clock']
        self.halfmove = kwargs['halfmove']
        self.king_in_check = kwargs['king_in_check']
        self.piece_map = kwargs['piece_map']
        self.white_pawn_moves = kwargs['white_pawn_moves']
        self.white_pawn_attacks = kwargs['white_pawn_attacks']
        self.white_knight_attacks = kwargs['white_pawn_attacks']
        self.white_rook_attacks = kwargs['white_pawn_attacks']
        self.white_bishop_attacks = kwargs['white_pawn_attacks']
        self.white_queen_attacks = kwargs['white_pawn_attacks']
        self.white_king_attacks = kwargs['white_pawn_attacks']
        self.black_pawn_moves = kwargs['black_pawn_moves']
        self.black_pawn_attacks = kwargs['black_pawn_attacks']
        self.black_knight_attacks = kwargs['black_pawn_attacks']
        self.black_rook_attacks = kwargs['black_pawn_attacks']
        self.black_bishop_attacks = kwargs['black_pawn_attacks']
        self.black_queen_attacks = kwargs['black_pawn_attacks']
        self.black_king_attacks = kwargs['black_pawn_attacks']


class Position():
    """
    Class to keep track of the full internal state of a chess position including
    all auxiliary information as well as piece locations
    """

    def __init__(self, board = None, position_state = None):

        if board is None:
            self.board = Board()
        else:
            self.board = board

        self.piece_map = {}
        self.set_initial_piece_locations()

        self.colour_to_move = Colour.WHITE
        self.castle_rights = {Colour.WHITE : [1, 1], Colour.BLACK : [1, 1]}
        self.halfmove_clock = 0
        self.halfmove = 2
        self.en_passant_target = None
        self.en_passant_side = Colour.WHITE
        self.is_en_passant_capture = False
        self.king_in_check = [0, 0]

        self.white_pawn_moves = make_uint()
        self.white_pawn_attacks = make_uint()
        self.white_rook_attacks = make_uint()
        self.white_knight_attacks = make_uint()
        self.white_bishop_attacks = make_uint()
        self.white_queen_attacks = make_uint()
        self.white_king_attacks = make_uint()

        self.black_pawn_moves = make_uint()
        self.black_pawn_attacks = make_uint()
        self.black_rook_attacks = make_uint()
        self.black_knight_attacks = make_uint()
        self.black_bishop_attacks = make_uint()
        self.black_queen_attacks = make_uint()
        self.black_king_attacks = make_uint()

    def set_initial_piece_locations(self):
        self.piece_map[Piece.wP] = set([i for i in range(8, 16)])
        self.piece_map[Piece.wR] = {0, 7}
        self.piece_map[Piece.wN] = {1, 6}
        self.piece_map[Piece.wB] = {2, 5}
        self.piece_map[Piece.wQ] = {3}
        self.piece_map[Piece.wK] = {4}

        self.piece_map[Piece.bP] = set([i for i in range(48, 56)])
        self.piece_map[Piece.bR] = {56, 63}
        self.piece_map[Piece.bN] = {57, 62}
        self.piece_map[Piece.bB] = {58, 61}
        self.piece_map[Piece.bQ] = {59}
        self.piece_map[Piece.bK] = {60}

    def reset_state_to(self, memento: PositionState) -> None:
        """
        Reset the position to a given memento
        """
        for key, value in memento.__dict__.items():
            setattr(self, key, value)

    @property
    def current_evaluation(self) -> float:
        """
        Return the evaluation of the position instance
        """
        material_balance = self.material_sum()
        white_material = material_balance[Colour.WHITE]
        black_material = material_balance[Colour.BLACK]

        return white_material - black_material

    @property
    def material_sum(self) -> dict:
        """
        Returns a dictionary with the material value for black and white separately
        """
        material_balance = {
            Colour.WHITE : 0,
            Colour.BLACK : 0
        }

        for key, value in self.piece_map.items():
            if key in {Piece.bK, Piece.wK}:
                continue
            if key in Piece.black_pieces:
                material_balance[Colour.BLACK] += len(value) * piece_to_value[key]
            else:
                material_balance[Colour.WHITE] += len(value) * piece_to_value[key]

        return material_balance

    @property
    def occupied_squares_by_colour(self):
        return {
            Colour.BLACK : self.board.black_pieces,
            Colour.WHITE : self.board.white_pieces
        }

    @property
    def black_attacked_squares(self):
        return (self.black_rook_attacks | self.black_knight_attacks |
                self.black_bishop_attacks | self.black_queen_attacks |
                self.black_king_attacks | self.black_pawn_attacks)

    @property
    def white_attacked_squares(self):
        return (self.white_rook_attacks | self.white_knight_attacks |
                self.white_bishop_attacks | self.white_queen_attacks |
                self.white_king_attacks | self.white_pawn_attacks)

    ## -------------------------------- ##
    ## Routine to make a (legal) move   ##
    ## -------------------------------- ##

    def make_move(self, move: Move) -> MoveResult:
        """
        Given a Move object will execute the move after checking it is a fully
        legal move on the current state of the game
        """

        original_position = PositionState(copy.deepcopy(self.__dict__))

        original_piece_map = copy.deepcopy(self.piece_map)

        if not self.colour_to_move == move.colour:
            return self.make_illegal_move_result("Not your turn to move!")

        if not self.is_legal_move(move):
            return self.make_illegal_move_result("This is not a valid move!")

        if move.is_capture:
            self.halfmove_clock = 0
            self.remove_opponent_piece(move.to)

        if self.is_en_passant_capture:
            if move.colour == Colour.WHITE:
                self.remove_opponent_piece(move.to - 8)
            if move.colour == Colour.BLACK:
                self.remove_opponent_piece(move.to + 8)

        self.is_en_passant_capture = False

        if move.piece in {Piece.wP, Piece.bP}:
            self.halfmove_clock = 0
        self.piece_map[move.piece].remove(move.from)
        self.piece_map[move.piece].add(move.to)

        if move.is_promotion:
            self.promote_pawn(move)

        if move.is_castling:
            self.move_rooks_for_castling(move)

        self.halfmove_clock += 1
        self.halfmove += 1

        castle_rights = self.castle_rights[move.colour]

        if castle_rights[0] or castle_rights[1]:
            self.adjust_castling_rights(move)

        if self.en_passant_side != move.colour:
            self.en_passant_target = None

        change_map = {key : value for key, value in self.piece_map.items() if self.piece_map[key] != original_piece_map[key]}
        self.board.update_position_bitboards(change_map)
        self.update_attack_bitboards(change_map)



    def make_illegal_move_result(self, message: str) -> MoveResult:
        """
        Return a move result for an illegal move with a given message
        """
        move_result = MoveResult()
        move_result.is_illegal_move = True
        move_result.fen = generate_fen(self)
        print(message)
        return move_result

    def remove_opponent_piece(self, to_square):
        """
        Remove a piece from the opponent in a capturing move
        """
        target = None

        for key, value in self.piece_map.items():
            if to_square in value:
                target = key
                break
        self.piece_map[target].remove(to_square)

    def promote_pawn(self, move):
        """
        Promote a pawn after it has crossed the board
        """
        while True:
            promotion_piece = input("Choose promotion piece")
            promotion_piece = promotion_piece.lower()
            legal_piece = user_promotion_input.get(promotion_piece)
            if not legal_piece:
                print("Please choose a legal promotion")
                continue
            self.piece_map[move.piece].remove(move.to)
            new_piece = self.get_promotion_piece_type(legal_piece, move)
            self.piece_map[new_piece].add(move.to)
            break

    def get_promotion_piece_type(self, legal_piece, move):
        if move.colour == Colour.WHITE:
            return white_promotion_map[legal_piece]
        if move.colour == Colour.BLACK:
            return black_promotion_map[legal_piece]

    def move_rooks_for_castling(self, move):
        """
        Perform a castling move given the side that is playing
        """
        rook_colour_map = {
            Colour.WHITE : Piece.wR,
            Colour.BLACK : Piece.bR
        }

        square_map = {
            Square.G1 : (Square.H1, Square.F1),
            Square.C1 : (Square.A1, Square.D1),
            Square.G8 : (Square.H8, Square.F8),
            Square.C8 : (Square.A8, Square.D8),
        }

        self.piece_map[rook_colour_map[move.colour]].remove(square_map[move.to][0])
        self.piece_map[rook_colour_map[move.colour]].add(square_map[move.to][1])


    def adjust_castling_rights(self, move):
        """
        Adjust the castling rights after a castling move is performed or when
        given rooks are moved
        """
        if move.piece in {Piece.wK, Piece.bK, Piece.wR, Piece.bR}:
            if move.piece == Piece.wK:
                self.castle_rights[Colour.WHITE] = [0, 0]
            if move.piece == Piece.bK:
                self.castle_rights[Colour.BLACK] == [0, 0]

            if move.piece == Piece.wR:
                if move.from == Square.H1:
                    self.castle_rights[Colour.WHITE][0] = 0
                if move.from == Square.A1:
                    self.castle_rights[Colour.WHITE][1] = 0
            if move.piece == Piece.bR:
                if move.from == Square.H8:
                    self.castle_rights[Colour.BLACK][0] = 0
                if move.from == Square.A8:
                    self.castle_rights[Colour.BLACK][1] = 0

    def update_attack_bitboards(self, change_map):
        """
        Update all attack bitboards that have changed after the
        move has been executed
        """
        for piece, squares in self.change_map.items():
            ## Pawns
            if piece == Piece.wP:
                for square in squares:
                    self.update_legal_pawn_moves(square, Color.WHITE)
            if piece == Piece.bP:
                for square in squares:
                    self.update_legal_pawn_moves(square, Color.BLACK)

            ## Rooks
            if piece == Piece.wR:
                for square in squares:
                    self.update_legal_rook_moves(square, Color.WHITE)
            if piece == Piece.bR:
                for square in squares:
                    self.update_legal_rook_moves(square, Color.BLACK)

            ## Knights
            if piece == Piece.wN:
                for square in squares:
                    self.update_legal_knight_moves(square, Color.WHITE)
            if piece == Piece.bN:
                for square in squares:
                    self.update_legal_knight_moves(square, Color.BLACK)

            ## Bishops
            if piece == Piece.wB:
                for square in squares:
                    self.update_legal_bishop_moves(square, Color.WHITE)
            if piece == Piece.bB:
                for square in squares:
                    self.update_legal_bishop_moves(square, Color.BLACK)

            ## Queens
            if piece == Piece.wQ:
                for square in squares:
                    self.update_legal_queen_moves(square, Color.WHITE)
            if piece == Piece.bQ:
                for square in squares:
                    self.update_legal_queen_moves(square, Color.BLACK)

            ## Kings
            if piece == Piece.wK:
                for square in squares:
                    self.update_legal_king_moves(square, Color.WHITE)

            if piece == Piece.bK:
                for square in squares:
                    self.update_legal_king_moves(square, Color.BLACK)

    def update_legal_pawn_moves(self, from_square: np.uint64, colour_to_move: int):
        """
        Update the pseudo-legal Pawn moves:
            - Pawn non-attacks that do not intersect with any occupied squares
            - Pawn attacks that intersect with opponent pieces
        Parameters:
            from_square: the proposed square from which the pawn is moving
            colour_to_move: current colour to move
        """
        bitboard = make_uint()

        self.white_pawn_attacks = self.board.white_pawn_attacks
        self.black_pawn_attacks = self.board.black_pawn_attacks

        legal_motion = {
            Colour.WHITE : self.board.white_pawn_moves[from_square],
            Colour.BLACK : self.board.black_pawn_moves[from_square]
        }

        legal_motion &= self.board.empty_squares

        legal_attacks = {
            Colour.WHITE : self.board.white_pawn_attacks[from_square],
            Colour.BLACK : self.board.black_pawn_attacks[from_square]
        }

        ## Handling en-passant for the pawns
        if self.en_passant_target:
            en_passant = set_bit(bitboard, self.en_passant_target)
            en_passant_move = legal_attacks[colour_to_move] & en_passant
            if en_passant_move:
                legal_attacks[colour_to_move] |= en_passant_move

        legal_moves = legal_motion[colour_to_move] & legal_attacks[colour_to_move]

        ## Remove own piece targets
        occupied_squares = {
            Colour.WHITE : self.board.white_pieces,
            Colour.BLACK : self.board.black_pieces
        }

        own_piece_targets = occupied_squares[colour_to_move]

        if own_piece_targets:
            legal_moves &= ~own_piece_targets

        if colour_to_move == Colour.WHITE:
            self.white_pawn_attacks = legal_attacks[Colour.WHITE]
            self.white_pawn_moves = legal_motion[Colour.WHITE]
        if colour_to_move == Colour.BLACK:
            self.black_pawn_attacks = legal_attacks[Colour.BLACK]
            self.black_pawn_moves = legal_motion[Colour.BLACK]

    def update_legal_rook_moves(self, from_square: np.uint64, colour_to_move: int):
        """
        Update the pseudo-legal sliding-piece moves for rank and file direction
        by getting the first blocker in each relevant direction using a bitscan.
        Parameters:
            from_square: the proposed square from which the rook is moving
            colour_to_move: current colour to move
        """
        bitboard = make_uint()
        occupied = self.board.occupied_squares

        north = north_ray(bitboard, from_square)
        intersection = occupied & north
        if intersection:
            first_block = forward_bitscan(intersection)
            block_ray = north_ray(bitboard, first_block)
            north ^= block_ray

        south = south_ray(bitboard, from_square)
        intersection = occupied & south
        if intersection:
            first_block = backward_bitscan(intersection)
            block_ray = south_ray(bitboard, first_block)
            south ^= block_ray

        east = east_ray(bitboard, from_square)
        intersection = occupied & east
        if intersection:
            first_block = forward_bitscan(intersection)
            block_ray = north_ray(bitboard, first_block)
            east ^= block_ray

        west = west_ray(bitboard, from_square)
        intersection = occupied & west
        if intersection:
            first_block = backward_bitscan(intersection)
            block_ray = south_ray(bitboard, first_block)
            west ^= block_ray

        legal_moves = north | south | east | west

        ## Remove own piece targets
        own_piece_targets = self.occupied_squares_by_colour[colour_to_move]

        if own_piece_targets:
            legal_moves &= ~own_piece_targets

        if colour_to_move == Colour.WHITE:
            self.white_rook_attacks = legal_moves
        if colour_to_move == Colour.BLACK:
            self.black_rook_attacks = legal_moves

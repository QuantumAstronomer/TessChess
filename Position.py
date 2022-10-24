import copy
import numpy as np

from Board import Board
from Constants import File, Rank, HOT, Piece, Colour, CastleRoute, Square, piece_to_value, \
                      user_promotion_input, white_promotion_map, black_promotion_map
from BitboardHelpers import set_bit, set_bit_multi, clear_bit, clear_bit_multi, forward_bitscan, backward_bitscan, \
                            bitboard_to_string, bitboard_to_squares, bitboard_pprint, make_uint
from Attacks import south_ray, north_ray, west_ray, east_ray, southwest_ray, southeast_ray, northwest_ray, northeast_ray, \
                    generate_king_attack_bitboard
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

        self.piece_map = dict()
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
            self.remove_opponent_piece(move.to_square)

        if self.is_en_passant_capture:
            if move.colour == Colour.WHITE:
                self.remove_opponent_piece(move.to_square - 8)
            if move.colour == Colour.BLACK:
                self.remove_opponent_piece(move.to_square + 8)

        self.is_en_passant_capture = False

        if move.piece in {Piece.wP, Piece.bP}:
            self.halfmove_clock = 0

        self.piece_map[move.piece].remove(move.from_square)
        self.piece_map[move.piece].add(move.to_square)

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

        ## Make a map of only the pieces that have changed position so we do not
        ## need to update bitboards for every single piece, which is more efficient
        change_map = {key : value for key, value in self.piece_map.items() if self.piece_map[key] != original_piece_map[key]}
        self.board.update_position_bitboards(change_map)
        self.update_attack_bitboards(change_map)

        self.evaluate_king_check()

        if self.king_in_check[move.colour]:
            self.reset_state_to(original_position)
            return self.make_illegal_move_result("Your own king is in check")

        other_player = not move.colour

        if self.king_in_check[other_player] and not self.has_any_moves(other_player):
            print("Checkmate")
            return self.make_checkmate_result()

        self.colour_to_move = not self.colour_to_move
        return self.make_move_result()

    ## ------------------------------------------------ ##
    ## Check whether the player has any possible moves  ##
    ## ------------------------------------------------ ##
    def has_any_moves(self, colour_to_move):
        """
        Returns true if any moves are possible
        """
        if self.has_king_move(colour_to_move):
            return True
        if self.has_queen_move(colour_to_move):
            return True
        if self.has_rook_move(colour_to_move):
            return True
        if self.has_knight_move(colour_to_move):
            return True
        if self.has_bishop_move(colour_to_move):
            return True
        if self.has_pawn_move(colour_to_move):
            return True

        return False

    def has_king_move(self, colour_to_move) -> bool:
        """
        Return True if there is a legal king move for the given colour in the
        current Position State
        """
        king_colour_map = {
            Colour.WHITE : (self.white_king_attacks, Piece.wK),
            Colour.BLACK : (self.black_king_attacks, Piece.bK),
        }
        attacked_squares = {
            Colour.WHITE : self.white_attacked_squares,
            Colour.BLACK : self.black_attacked_squares
        }

        king_attacks = king_colour_map[colour_to_move][0] & ~attacked_squares[not colour_to_move]
        king_piece = king_colour_map[colour_to_move][1]

        if not king_attacks.any():
            return False

        from_square = self.piece_map[king_piece].copy()
        king_squares = bitboard_to_squares(king_attacks)

        for to_square in king_squares:
            if self.is_legal_king_move(Move(king_piece, (from_square, to_square))):
                return True
        return False

    def has_queen_move(self, colour_to_move) -> bool:
        """
        Return True if there is a legal queen move for the given colour in the
        current Position State
        """
        queen_colour_map = {
            Colour.WHITE : (self.white_queen_attacks, Piece.wQ),
            Colour.BLACK : (self.black_queen_attacks, Piece.bQ)
        }
        queen_attacks = queen_colour_map[colour_to_move][0]
        queen_piece = queen_colour_map[colour_to_move][1]

        if not queen_attacks.any():
            return False

        from_squares = self.piece_map[queen_piece]
        to_squares = bitboard_to_squares(queen_attacks)
        for from_square in list(from_squares):
            for to_square in to_squares:
                if self.is_legal_queen_move(Move(queen_piece, (from_square, to_square))):
                    return True
        return False

    def has_knight_move(self, colour_to_move) -> bool:
        """
        Return True if there is a legal knight move for the given colour in the
        current Position State
        """
        knight_colour_map = {
            Colour.WHITE : (self.white_knight_attacks, Piece.wN),
            Colour.BLACK : (self.black_knight_attacks, Piece.bN)
        }

        knight_attacks = knight_colour_map[colour_to_move][0]
        knight_piece = knight_colour_map[colour_to_move][1]

        if not knight_attacks.any():
            return False

        from_squares = self.piece_map[knight_piece]
        to_squares = bitboard_to_squares(knight_attacks)
        for from_square in list(from_squares):
            for to_square in to_squares:
                if self.is_legal_knight_move(Move(knight_piece, (from_square, to_square))):
                    return True
        return False

    def has_bishop_move(self, colour_to_move) -> bool:
        """
        Return True if there is a legal bishop move for the given colour in the
        current Position State
        """
        bishop_colour_map = {
            Colour.WHITE : (self.white_bishop_attacks, Piece.wB),
            Colour.BLACK : (self.black_bishop_attacks, Piece.bB)
        }

        bishop_attacks = bishop_colour_map[colour_to_move][0]
        bishop_piece = bishop_colour_map[colour_to_move][1]

        if not bishop_attacks.any():
            return False

        from_squares = self.piece_map[bishop_piece]
        to_squares = bitboard_to_squares(bishop_attacks)
        for from_square in list(from_squares):
            for to_square in to_squares:
                if self.is_legal_bishop_move(Move(bishop_piece, (from_square, to_square))):
                    return True
        return False

    def has_rook_move(self, colour_to_move) -> bool:
        """
        Return True if there is a legal rook move for the given colour in the
        current Position State
        """
        rook_colour_map = {
            Colour.WHITE : (self.white_rook_attacks, Piece.wR),
            Colour.BLACK : (self.black_rook_attacks, Piece.bR)
        }

        rook_attacks = rook_colour_map[colour_to_move][0]
        rook_piece = rook_colour_map[colour_to_move][1]

        if not rook_attacks.any():
            return False

        from_squares = self.piece_map[rook_piece]
        to_squares = bitboard_to_squares(rook_attacks)
        for from_square in list(from_squares):
            for to_square in to_squares:
                if self.is_legal_rook_move(Move(rook_piece, (from_square, to_square))):
                    return True
        return False

    def has_pawn_move(self, colour_to_move) -> bool:
        """
        Return True if there is a legal pawn move for the given colour in the
        current Position State
        """
        pawn_colour_map = {
            Colour.WHITE : (self.white_pawn_attacks | self.white_pawn_moves, Piece.wP),
            Colour.BLACK : (self.black_pawn_attacks | self.black_pawn_moves, Piece.bP)
        }

        pawn_attacks = pawn_colour_map[colour_to_move][0]
        pawn_piece = pawn_colour_map[colour_to_move][1]

        if not pawn_attacks.any():
            return False

        from_squares = self.piece_map[pawn_piece]
        to_squares = bitboard_to_squares(pawn_attacks)
        for from_square in list(from_squares):
            for to_square in to_squares:
                if self.is_legal_pawn_move(Move(pawn_piece, (from_square, to_square))):
                    return True
        return False

    def is_legal_move(self, move: Move) -> bool:
        """
        Check whether the given move is legal in the current Position State of
        the game. Return True if and only if the move is legal.
        """
        piece = move.piece

        if self.is_capture(move):
            move.is_capture = True

        ## Pawn movement legality
        if piece in (Piece.wP, Piece.bP):
            is_legal_pawn_move = self.is_legal_pawn_move(move)

            if not is_legal_pawn_move:
                return False
            ## Handling promotions
            if self.is_promotion(move):
                move.is_promotion = True
                return True

            ## Handling en passant moves for pawns
            en_passant_target = self.get_en_passant_target(move)
            if en_passant_target:
                self.en_passant_side = move.colour
                self.en_passant_target = int(en_passant_target)
            if move.to_square == self.en_passant_target:
                self.is_en_passant_capture = True

            return True

        if piece in (Piece.wN, Piece.bN):
            return self.is_legal_knight_move(move)
        if piece in (Piece.wR, Piece.bR):
            return self.is_legal_rook_move(move)
        if piece in (Piece.wB, Piece.bB):
            return self.is_legal_bishop_move(move)
        if piece in (Piece.wQ, Piece.bQ):
            return self.is_legal_queen_move(move)
        if piece in (Piece.wK, Piece.bK):
            legal_king_move = self.is_legal_king_move(move)
            if not legal_king_move:
                return False
            if self.is_castling(move):
                move.is_castling = True
            return True

    def is_capture(self, move: Move) -> bool:
        """
        Check whether a move is capturing another piece in the current
        Position State of the game.
        """
        to_square = set_bit(make_uint(), move.to_square)

        if move.colour == Colour.WHITE:
            intersects = to_square & self.board.black_pieces
            if intersects:
                return True
        if move.colour == Colour.BLACK:
            intersects = to_square & self.board.white_pieces
            if intersects:
                return True
        return False

    def is_promotion(self, pawn_move: Move) -> bool:
        """
        Check whether a pawn move leads to promotion.
        """
        if pawn_move.colour == Colour.WHITE and pawn_move.to_square in Rank.x8:
            return True
        if pawn_move.colour == Colour.BLACK and pawn_move.to_square in Rank.x1:
            return True
        return False

    def is_castling(self, move: Move) -> bool:
        if move.from_square == Square.E1:
            if move.to_square in {Square.G1, Square.C1}:
                return True
        if move.from_square == Square.E8:
            if move.to_square in {Square.G8, Square.C8}:
                return True
        return False

    def get_en_passant_target(self, move: Move):
        if move.piece not in {Piece.wP, Piece.bP}:
            return None
        if move.colour == Colour.WHITE:
            if move.to_square in Rank.x4 and move.from_square in Rank.x2:
                return move.to_square - 8
        if move.colour == Colour.BLACK:
            if move.to_square in Rank.x5 and move.from_square in Rank.x7:
                return move.to_square + 8

    ## ---------------------------- ##
    ## Move legality by piece type  ##
    ## ---------------------------- ##

    ## Pawns
    def is_legal_pawn_move(self, move: Move) -> bool:
        """
        Returns True if and only if the given pawn move is fully legal
        """
        from_bitboard = set_bit(make_uint(), move.from_square)
        to_bitboard = set_bit(make_uint(), move.to_square)
        en_passant_target = make_uint()

        if self.en_passant_target:
            en_passant_target = set_bit(make_uint(), self.en_passant_target)

        if move.piece == Piece.wP:
            ## Check if the selected from square is actually a pawn
            if not (self.board.white_pawn & from_bitboard):
                return False
            ## Cehck that the move is compliant with pawn movement patterns
            if self.is_not_pawn_move(move):
                return False
            ## If forwards pawnmotion, check it is not blocked by another piece
            if move.from_square == move.to_square - 8 and (to_bitboard & self.board.occupied_squares):
                return False
            ## If pawncapture movement, check it intersects with black pieces or
            ## the en passant target
            if move.from_square == move.to_square - 9 or move.from_square == move.to_square - 7:
                if (self.white_pawn_attacks & to_bitboard) & ~(self.board.black_pieces | en_passant_target):
                    return False
            return True

        if move.piece == Piece.bP:
            ## Check if the selected from square is actually a pawn
            if not (self.board.black_pawn & from_bitboard):
                return False
            ## Cehck that the move is compliant with pawn movement patterns
            if self.is_not_pawn_move(move):
                return False
            ## If forwards pawnmotion, check it is not blocked by another piece
            if move.from_square == move.to_square + 8 and (to_bitboard & self.board.occupied_squares):
                return False
            ## If pawncapture movement, check it intersects with white pieces or
            ## the en passant target
            if move.from_square == move.to_square + 9 or move.from_square == move.to_square + 7:
                if (self.black_pawn_attacks & to_bitboard) & ~(self.board.white_pieces | en_passant_target):
                    return False
            return True

        return False

    ## Knight moves
    def is_legal_knight_move(self, move: Move) -> bool:
        """
        Returns True if and only if the proposed knight move is fully legal
        """
        from_bitboard = set_bit(make_uint(), move.from_square)
        if move.piece == Piece.wN:
            if not (self.board.white_knight & from_bitboard):
                return False
            if self.is_not_knight_attack(move):
                return False
            return True

        if move.piece == Piece.bN:
            if not (self.board.black_knight & from_bitboard):
                return False
            if self.is_not_knight_attack(move):
                return False
            return True

        return False

    ## Rook moves
    def is_legal_rook_move(self, move: Move) -> bool:
        """
        Returns True if and only if the proposed rook move obeys the rules
        for rook movement
        """
        from_bitboard = set_bit(make_uint(), move.from_square)
        if move.piece == Piece.wR:
            if not (self.board.white_rook & from_bitboard):
                return False
            if self.is_not_rook_attack(move):
                return False
            return True

        if move.piece == Piece.bR:
            if not (self.board.black_rook & from_bitboard):
                return False
            if self.is_not_rook_attack(move):
                return False
            return True

        return False

    ## Bishop moves
    def is_legal_bishop_move(self, move: Move) -> bool:
        """
        Return True if and only if the proposed move is a valid bishop move
        """
        from_bitboard = set_bit(make_uint(), move.from_square)
        if move.piece == Piece.wB:
            if not (self.board.white_bishop & from_bitboard):
                return False
            if self.is_not_bishop_attack(move):
                return False
            return True

        if move.piece == Piece.bB:
            if not (self.board.black_bishop & from_bitboard):
                return False
            if self.is_not_bishop_attack(move):
                return False
            return True

        return False

    ## Queen moves
    def is_legal_queen_move(self, move: Move) -> bool:
        """
        Return True if and only if the proposed move obeys queen movement patterns
        """
        from_bitboard = set_bit(make_uint(), move.from_square)
        if move.piece == Piece.wQ:
            if not(self.board.white_queen & from_bitboard):
                return False
            if self.is_not_queen_attack(move):
                return False
            return True

        if move.piece == Piece.bQ:
            if not (self.board.black_queen & from_bitboard):
                return False
            if self.is_not_queen_attack(move):
                return False
            return True

        return False

    ## King moves
    def is_legal_king_move(self, move: Move) -> bool:
        """
        Returns True if and only if the proposed move is a valid king move
        """
        from_bitboard = set_bit(make_uint(), move.from_square)
        if move.piece == Piece.wK:
            if not(self.board.white_king & from_bitboard):
                return False
            if self.is_not_king_attack(move):
                return False
            return True

        if move.piece == Piece.bK:
            if not (self.board.black_king & from_bitboard):
                return False
            if self.is_not_king_attack(move):
                return False
            return True

        return False

    ## ---------------------------- ##
    ## Piece move legality helpers  ##
    ## ---------------------------- ##

    ## Pawns
    def is_not_pawn_move(self, move: Move) -> bool:
        to_bitboard = set_bit(make_uint(), move.to_square)
        if move.colour == Colour.WHITE:
            if not (self.board.white_pawn_move[move.from_square] |
                    self.board.white_pawn_attack[move.from_square]) & to_bitboard:
                return True
            return False
        if move.colour == Colour.BLACK:
            if not (self.board.black_pawn_move[move.from_square] |
                    self.board.black_pawn_attack[move.from_square]) & to_bitboard:
                return True
            return False

    ## Knights
    def is_not_knight_attack(self, move: Move) -> bool:
        to_bitboard = set_bit(make_uint(), move.to_square)
        if move.colour == Colour.WHITE:
            if not (self.board.knight_attacks[move.from_square] & to_bitboard):
                return True
            return False
        if move.colour == Colour.BLACK:
            if not (self.board.knight_attacks[move.from_square] & to_bitboard):
                return True
            return False

    ## Rooks
    def is_not_rook_attack(self, move: Move) -> bool:
        to_bitboard = set_bit(make_uint(), move.to_square)
        if move.colour == Colour.WHITE:
            if not (self.board.rook_attacks[move.from_square] & to_bitboard):
                return True
            return False
        if move.colour == Colour.BLACK:
            if not (self.board.rook_attacks[move.from_square] & to_bitboard):
                return True
            return False

    ## Bishops
    def is_not_bishop_attack(self, move: Move) -> bool:
        to_bitboard = set_bit(make_uint(), move.to_square)
        if move.colour == Colour.WHITE:
            if not (self.board.bishop_attacks[move.from_square] & to_bitboard):
                return True
            return False
        if move.colour == Colour.BLACK:
            if not (self.board.bishop_attacks[move.from_square] & to_bitboard):
                return True
            return False

    ## Queens
    def is_not_queen_attack(self, move: Move) -> bool:
        to_bitboard = set_bit(make_uint(), move.to_square)
        if move.colour == Colour.WHITE:
            if not (self.board.queen_attacks[move.from_square] & to_bitboard):
                return True
            return False
        if move.colour == Colour.BLACK:
            if not (self.board.queen_attacks[move.from_square] & to_bitboard):
                return True
            return False

    ## Kings
    def is_not_king_attack(self, move: Move) -> bool:
        to_bitboard = set_bit(make_uint(), move.to_square)
        if move.colour == Colour.WHITE:
            if not (self.board.king_attacks[move.from_square] & to_bitboard):
                return True
            return False
        if move.colour == Colour.BLACK:
            if not (self.board.king_attacks[move.from_square] & to_bitboard):
                return True
            return False

    def make_illegal_move_result(self, message: str) -> MoveResult:
        """
        Return a move result for an illegal move with a given message
        """
        move_result = MoveResult()
        move_result.is_illegal_move = True
        move_result.fen = None
        print(message)
        return move_result

    def make_move_result(self) -> MoveResult:
        """
        Return a move result
        """
        move_result = MoveResult()
        move_result.fen = None
        return move_result

    def make_checkmate_result(self) -> MoveResult:
        """
        Return a move result for a move that ended in checkmate
        """
        move_result = MoveResult()
        move_result.is_checkmate = True
        move_result.fen = None
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
            self.piece_map[move.piece].remove(move.to_square)
            new_piece = self.get_promotion_piece_type(legal_piece, move)
            self.piece_map[new_piece].add(move.to_square)
            break

    def get_promotion_piece_type(self, legal_piece, move: Move):
        if move.colour == Colour.WHITE:
            return white_promotion_map[legal_piece]
        if move.colour == Colour.BLACK:
            return black_promotion_map[legal_piece]

    def get_piece_on_square(self, from_square: int):
        for key, value in self.piece_map.items():
            for square in value:
                if square == from_square:
                    return key
        return None

    def move_rooks_for_castling(self, move: Move):
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

        self.piece_map[rook_colour_map[move.colour]].remove(square_map[move.to_square][0])
        self.piece_map[rook_colour_map[move.colour]].add(square_map[move.to_square][1])

    def adjust_castling_rights(self, move: Move):
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
                if move.from_square == Square.H1:
                    self.castle_rights[Colour.WHITE][0] = 0
                if move.from_square == Square.A1:
                    self.castle_rights[Colour.WHITE][1] = 0
            if move.piece == Piece.bR:
                if move.from_square == Square.H8:
                    self.castle_rights[Colour.BLACK][0] = 0
                if move.from_square == Square.A8:
                    self.castle_rights[Colour.BLACK][1] = 0

    def update_attack_bitboards(self, change_map):
        """
        Update all attack bitboards that have changed after the
        move has been executed
        """
        for piece, squares in change_map.items():
            ## Pawns
            if piece == Piece.wP:
                for square in squares:
                    self.white_pawn_attacks = make_uint()
                    self.white_pawn_moves = make_uint()
                    self.update_legal_pawn_moves(square, Colour.WHITE)
            if piece == Piece.bP:
                for square in squares:
                    self.black_pawn_attacks = make_uint()
                    self.black_pawn_moves = make_uint()
                    self.update_legal_pawn_moves(square, Colour.BLACK)

            ## Rooks
            if piece == Piece.wR:
                for square in squares:
                    self.white_rook_attacks = make_uint()
                    self.update_legal_rook_moves(square, Colour.WHITE)
            if piece == Piece.bR:
                for square in squares:
                    self.black_rook_attacks = make_uint()
                    self.update_legal_rook_moves(square, Colour.BLACK)

            ## Knights
            if piece == Piece.wN:
                for square in squares:
                    self.white_knight_attacks = make_uint()
                    self.update_legal_knight_moves(square, Colour.WHITE)
            if piece == Piece.bN:
                for square in squares:
                    self.black_knight_attacks = make_uint()
                    self.update_legal_knight_moves(square, Colour.BLACK)

            ## Bishops
            if piece == Piece.wB:
                for square in squares:
                    self.white_bishop_attacks = make_uint()
                    self.update_legal_bishop_moves(square, Colour.WHITE)
            if piece == Piece.bB:
                for square in squares:
                    self.black_bishop_attacks = make_uint()
                    self.update_legal_bishop_moves(square, Colour.BLACK)

            ## Queens
            if piece == Piece.wQ:
                for square in squares:
                    self.white_queen_attacks = make_uint()
                    self.update_legal_queen_moves(square, Colour.WHITE)
            if piece == Piece.bQ:
                for square in squares:
                    self.black_queen_attacks = make_uint()
                    self.update_legal_queen_moves(square, Colour.BLACK)

            ## Kings
            if piece == Piece.wK:
                for square in squares:
                    self.white_king_attacks = make_uint()
                    self.update_legal_king_moves(square, Colour.WHITE)
            if piece == Piece.bK:
                for square in squares:
                    self.black_king_attacks = make_uint()
                    self.update_legal_king_moves(square, Colour.BLACK)

## ---------------------------- ##
## Updating legal pawn moves    ##
## ---------------------------- ##
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
            Colour.WHITE : self.board.white_pawn_move[from_square],
            Colour.BLACK : self.board.black_pawn_move[from_square]
        }

        legal_motion[colour_to_move] &= self.board.empty_squares

        legal_attacks = {
            Colour.WHITE : self.board.white_pawn_attack[from_square],
            Colour.BLACK : self.board.black_pawn_attack[from_square]
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

## ---------------------------- ##
## Updating legal rook moves    ##
## ---------------------------- ##
    def update_legal_rook_moves(self, from_square: np.uint64, colour_to_move: int) -> np.uint64:
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
        return legal_moves

## ---------------------------- ##
## Updating legal knight moves  ##
## ---------------------------- ##
    def update_legal_knight_moves(self, from_square: np.uint64, colour_to_move: int):
        """
        Update the pseudo-legal knight moves. Note that a knight is not a sliding
        piece so it can jump over any 'blocking' pieces.
        Parameters:
            from_square: the proposed square from which the knight is moving
            colour_to_move: current colour to move
        """
        legal_knight_moves = self.board.get_knight_attack(from_square)
        ## It can not capture friendly pieces so remove these
        if colour_to_move == Colour.WHITE:
            legal_knight_moves &= ~self.board.white_pieces
            self.white_knight_attacks = legal_knight_moves
        if colour_to_move == Colour.BLACK:
            legal_knight_moves &= ~self.board.black_pieces
            self.black_knight_attacks = legal_knight_moves

## ---------------------------- ##
## Updating legal bishop moves  ##
## ---------------------------- ##
    def update_legal_bishop_moves(self, from_square: np.uint64, colour_to_move: int) -> np.uint64:
        """
        Update the pseudo-legal sliding-piece moves for diagonal directions
        by getting the first blocker in each relevant direction using a bitscan.
        Parameters:
            from_square: the proposed square from which the rook is moving
            colour_to_move: current colour to move
        """
        bitboard = make_uint()
        occupied = self.board.occupied_squares

        northeast = northeast_ray(bitboard, from_square)
        intersection = northeast & occupied
        if intersection:
            first_block = forward_bitscan(intersection)
            block_ray = northeast_ray(bitboard, first_block)
            northeast ^= block_ray

        northwest = northwest_ray(bitboard, from_square)
        intersection = northwest & occupied
        if intersection:
            first_block = forward_bitscan(intersection)
            block_ray = northwest_ray(bitboard, first_block)
            northwest ^= block_ray

        southeast = southeast_ray(bitboard, from_square)
        intersection = southeast & occupied
        if intersection:
            first_block = backward_bitscan(intersection)
            block_ray = southeast_ray(bitboard, first_block)
            southeast ^= block_ray

        southwest = southwest_ray(bitboard, from_square)
        intersection = southwest & occupied
        if intersection:
            first_block = backward_bitscan(intersection)
            block_ray = southwest_ray(bitboard, first_block)
            southwest ^= block_ray

        legal_moves = northeast | northwest | southeast | southwest

        ## Remove own piece targets
        own_piece_targets = self.occupied_squares_by_colour[colour_to_move]
        if own_piece_targets:
            legal_moves &= ~own_piece_targets

        if colour_to_move == Colour.WHITE:
            self.white_bishop_attacks = legal_moves
        if colour_to_move == Colour.BLACK:
            self.black_bishop_attacks = legal_moves
        return legal_moves

## ---------------------------- ##
## Updating legal queen moves   ##
## ---------------------------- ##
    def update_legal_queen_moves(self, from_square: np.uint64, colour_to_move: int):
        """
        Update the pseudo-legal queen moves through a bitwise OR of the legal
        bishop and legal rook moves.
        Parameters:
            from_square: the proposed square from which the rook is moving
            colour_to_move: current colour to move
        """
        diagonal_moves = self.update_legal_bishop_moves(from_square, colour_to_move)
        rankfile_moves = self.update_legal_rook_moves(from_square, colour_to_move)

        legal_moves = diagonal_moves | rankfile_moves

        if colour_to_move == Colour.WHITE:
            self.white_queen_attacks = legal_moves
        if colour_to_move == Colour.BLACK:
            self.black_queen_attacks = legal_moves

## ---------------------------- ##
## Updating legal king moves    ##
## ---------------------------- ##
    def update_legal_king_moves(self, from_square: np.uint64, colour_to_move: int):
        """
        Update the pseudo-legal king moves: one step in each direction and in
        special circumstances also castling.
        Parameters:
            from_square: the proposed square from which the rook is moving
            colour_to_move: current colour to move
        """
        king_moves = generate_king_attack_bitboard(from_square)

        if colour_to_move == Colour.WHITE:
            own_piece_targets = self.board.white_pieces
        if colour_to_move == Colour.BLACK:
            own_piece_targets = self.board.black_pieces
        if own_piece_targets.any():
            king_moves &= ~own_piece_targets

        ## Handle the castling moves for the king
        can_castle = self.can_castle(colour_to_move)

    ## King castling helper functions
    def can_castle(self, colour_to_move: int) -> list:
        """
        Check wether or not the king of the given colour can castle and also if
        the king can perform the castling route without being in check along
        the way.
        Parameters:
            colour_to_move: current colour to move
        Returns:
            Tuple (bool, bool) giving the castling rights on (kingside, queenside)
        """
        castle_rights = self.castle_rights[colour_to_move]
        if not castle_rights[0] and not castle_rights[1]:
            return [0, 0]

        if colour_to_move == Colour.WHITE:
            if castle_rights[0]:
                kingside_block = (self.black_attacked_squares |
                                 (self.board.white_pieces & ~self.board.white_king)) & \
                                  CastleRoute.WhiteKingside
                rookH1 = self.board.white_rook & set_bit(make_uint(), Square.H1)
                kingside = not kingside_block.any() and rookH1.any()
            elif not castle_rights[0]:
                kingside = 0
            if castle_rights[1]:
                queenside_block = (self.black_attacked_squares |
                                  (self.board.white_pieces & ~self.board.white_king)) & \
                                   CastleRoute.WhiteQueenside
                rookA1 = self.board.white_rook & set_bit(make_uint(), Square.A1)
                queenside = not queenside_block.any() and rookA1.any()
            elif not castle_rights[1]:
                queenside = 0
            return [kingside, queenside]

        if colour_to_move == Colour.BLACK:
            if castle_rights[0]:
                kingside_block = (self.white_attacked_squares |
                                 (self.board.black_pieces & ~self.board.black_king)) & \
                                  CastleRoute.BlackKingside
                rookH8 = self.board.black_rook & set_bit(make_uint(), Square.H8)
                kingside = not kingside_block.any() & rookH8.any()
            elif not castle_rights[0]:
                kingside = 0
            if castle_rights[1]:
                queenside_block = (self.white_attacked_squares |
                                  (self.board.black_pieces & ~self.board.black_king)) & \
                                   CastleRoute.BlackQueenside
                rookA8 = self.board.black_rook & set_bit(make_uint(), Square.A8)
                queenside = not queenside_block.any() & rookA8.any()
            elif not castle_rights[1]:
                queenside = 0
            return [kingside, queenside]

    @staticmethod
    def add_castling_moves(bitboard: np.uint64, can_castle: list, colour_to_move: int) -> np.uint64:
        """
        Add the castling squares to the bitboard
        Parameters:
            bitboard: bitboard to put the castling squares into
            can_castle: whether or not we are allowed to castle on the given side
            colour_to_move: the colour that is to move
        Returns:
            bitboard containing the squares to perform the castling moves
        """
        if colour_to_move == Colour.WHITE:
            if can_castle[0]:
                bitboard |= set_bit(bitboard, Square.G1)
            if can_castle[1]:
                bitboard |= set_bit(bitboard, Square.C1)
        if colour_to_move == Colour.BLACK:
            if can_castle[0]:
                bitboard |= set_bit(bitboard, Square.G8)
            if can_castle[1]:
                bitboard |= set_bit(bitboard, Square.C8)

        return bitboard

    def evaluate_king_check(self):
        """
        Evaluates the state for the intersection of attacked squares and
        opposing king position to update king_in_check for the corresponding
        colour
        """
        if self.black_attacked_squares & self.board.white_king:
            self.king_in_check[0] = 1
        else:
            self.king_in_check[0] = 0

        if self.white_attacked_squares & self.board.black_king:
            self.king_in_check[1] = 1
        else:
            self.king_in_check[1] = 0

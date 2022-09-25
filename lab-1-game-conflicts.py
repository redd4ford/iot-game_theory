from typing import Tuple, List
import numpy as np


class Printer:
    """
    Helper class used to print data in a prettier way.
    """

    @classmethod
    def __get_point_position(cls, col, row) -> str:
        """
        Return point position as (x, y).
        """
        return f'({col}, {row})'

    @classmethod
    def print_matrix(cls, matrix: np.ndarray) -> None:
        """
        Print the matrix without brackets and with precision.
        """
        for row in matrix:
            print(' '.join([f'{x}\t' for x in row]))

    @classmethod
    def print_all_strategies(cls, gains: np.ndarray, gains_indices: np.ndarray,
                             indices_type: str) -> None:
        """
        Print all the gains found for the player in a format: "gain_value: (row, col)".
        """
        for count, gain in enumerate(gains, start=0):
            gain_position_as_str = (
                cls.__get_point_position(col=count + 1, row=gains_indices[count] + 1)
                if indices_type == 'row'
                else
                cls.__get_point_position(col=gains_indices[count] + 1, row=count + 1)
            )
            print(f'{gain}: {gain_position_as_str}')

    @classmethod
    def print_best_strategy(cls, gain_type: str, value: np.int64,
                            indices: np.ndarray) -> None:
        """
        Print maximin/minimax gain and its row/col index.
        """
        print(
            f'{gain_type} gain: {value} at '
            f'{"row" if gain_type == "maximin" else "col"}(s) {indices}'
        )

    @classmethod
    def print_saddle_points(cls, saddle_points: List[Tuple]) -> None:
        print(
            'Found saddle point(s) at:',
            ', '.join(
                cls.__get_point_position(x, y) for x, y in saddle_points
            )
        )


class GameConflictSolver:
    """
    Examines all strategies for players and finds saddle points in a provided
    (or generated) game model matrix.
    """

    def __init__(self, game_model: np.ndarray = None) -> None:
        self.game_model = game_model
        self.gains_1st_player = None
        self.gains_2nd_player = None
        self.maximin_gain_1st_player = None
        self.minimax_gain_2nd_player = None
        self.gains_indices_1st_player = None
        self.gains_indices_2nd_player = None
        self.maximin_gain_indices_1st_player = None
        self.minimax_gain_indices_2nd_player = None

    @classmethod
    def validate_args(cls, rows: int, cols: int, c1: int, c2: int) -> None:
        """
        Validate the matrix limits provided by user.
        """
        if rows <= 1 or cols <= 1:
            raise Exception(
                'Numbers of rows & columns should be greater than 1. '
                f'You provided {rows}x{cols}'
            )
        elif c1 >= c2:
            raise Exception(
                'Upper limit for matrix values should be higher than the lower limit. '
                f'You provided {c1} >= {c2}'
            )

    def generate_matrix(self, rows, cols, lower_limit, upper_limit) -> None:
        """
        Generate a random integer matrix of size :rows: x :cols: where each value
        is equal to/greater than :lower_limit: and equal to/less than :upper_limit:.
        """
        self.game_model = np.random.randint(
            low=lower_limit,
            high=upper_limit + 1,
            size=(rows, cols),
        )

    def find_strategies_for_1st_player(self) -> None:
        """
        Choose the min values per each row and store their row indices.
        """
        # min gains per row
        self.gains_1st_player: np.ndarray = np.min(self.game_model, axis=1)
        # row indices for min gains
        self.gains_indices_1st_player: np.ndarray = (
            self.game_model.argmin(axis=1).flatten()
        )

    def find_maximin_gain_for_1st_player(self) -> None:
        """
        Find the maximin (the max value among all the min values per each row) and
        its indices per row.
        """
        # maximin value
        self.maximin_gain_1st_player: np.int64 = np.max(self.gains_1st_player)
        # maximin row indices
        self.maximin_gain_indices_1st_player: np.ndarray = (
            np.argwhere(self.gains_1st_player == self.maximin_gain_1st_player).flatten()
        )

    def find_strategies_for_2nd_player(self) -> None:
        """
        Choose the max values per each column and store their column indices.
        """
        # max gains per column
        self.gains_2nd_player: np.ndarray = np.max(self.game_model, axis=0)
        # column indices for max gains
        self.gains_indices_2nd_player: np.ndarray = (
            self.game_model.argmax(axis=0).flatten()
        )

    def find_minimax_gain_for_2nd_player(self) -> None:
        """
        Find the minimax (the min value among all the max values per each column) and
        its indices per column.
        """
        # minimax value
        self.minimax_gain_2nd_player: np.int64 = np.min(self.gains_2nd_player)
        # minimax column indices
        self.minimax_gain_indices_2nd_player: np.ndarray = (
            np.argwhere(self.gains_2nd_player == self.minimax_gain_2nd_player).flatten()
        )

    def get_saddle_points_positions(self) -> List[Tuple[int, int]]:
        """
        Return the list of saddle points positions.
        """
        return [
            (i + 1, j + 1)
            for i in self.maximin_gain_indices_1st_player
            for j in self.minimax_gain_indices_2nd_player
        ]

    def run(self, n: int = 4, m: int = 5, c1: int = -4, c2: int = 5) -> None:
        """
        :param n: number of rows (opt)
        :param m: number of columns (opt)
        :param c1: lower limit for matrix values (opt)
        :param c2: upper limit for matrix values (opt)
        """
        self.validate_args(n, m, c1, c2)
        if self.game_model is None:
            print(f'Generating the matrix of size {n}x{m}...')
            self.generate_matrix(n, m, c1, c2)
        Printer.print_matrix(self.game_model)

        self.find_strategies_for_1st_player()
        self.find_maximin_gain_for_1st_player()

        Printer.print_all_strategies(
            self.gains_1st_player, self.gains_indices_1st_player,
            indices_type='row'
        )
        Printer.print_best_strategy(
            gain_type='maximin', value=self.maximin_gain_1st_player,
            indices=self.maximin_gain_indices_1st_player
        )

        self.find_strategies_for_2nd_player()
        self.find_minimax_gain_for_2nd_player()

        Printer.print_all_strategies(
            self.gains_2nd_player, self.gains_indices_2nd_player,
            indices_type='col'
        )
        Printer.print_best_strategy(
            gain_type='minimax', value=self.minimax_gain_2nd_player,
            indices=self.minimax_gain_indices_2nd_player
        )

        if self.maximin_gain_1st_player == self.minimax_gain_2nd_player:
            Printer.print_saddle_points(
                saddle_points=self.get_saddle_points_positions()
            )
        else:
            print('No saddle points found')


if __name__ == '__main__':
    # random
    GameConflictSolver().run()
    print('=' * 50)

    # one saddle point
    GameConflictSolver(
        game_model=np.array([
            [-1, -2, 5,  4, -4],
            [ 5,  0, 2,  1,  2],
            [ 2, -3, 5, -4,  4],
            [ 3,  0, 4, -1,  0]
        ])
    ).run()
    print('=' * 50)

    # two saddle points
    GameConflictSolver(
        game_model=np.array([
            [-3,  0, -2,  -3,  2],
            [-4,  2,  1,  -4,  4],
            [-3, -2, -1,  -3,  0],
            [-4, -3, -1,   1,  2]
        ])
    ).run()
    print('=' * 50)

    # no saddle points
    GameConflictSolver(
        game_model=np.array([
            [ 1,  5,  0, -1,  2],
            [-1,  2, -3,  1,  5],
            [-3,  4, -4,  0, -4],
            [-2, -4,  1,  2,  3]
        ])
    ).run()
    print('=' * 50)

from operator import itemgetter
from typing import (
    Tuple,
    List,
    Callable,
    Set,
)

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import LineString


class Printer:
    """
    Helper class used to print data in a prettier way.
    """

    @classmethod
    def print_matrix(cls, matrix: np.ndarray) -> None:
        """
        Print the matrix without brackets and with precision.
        """
        for row in matrix:
            print(' '.join([f'{x}\t' for x in row]))

    @classmethod
    def print_results(cls, game_type: str, optimal_strategy: tuple,
                      game_value: float) -> None:
        """
        Print the results.
        """
        print(f'Optimal strategy:')
        print(f'{"X" if game_type == "2xN" else "Y"} = {optimal_strategy}')
        print(f'Game value: {game_value}')


class Plotter:
    """
    Helper class used to plot a graph.
    """

    @classmethod
    def plot(cls, is_2n_game: bool, best_strategy: tuple, lines_y_coordinates: List) \
            -> None:
        """
        Plot a line graph to solve the game with graph-analytical method.
        """
        highest_y = max(max(lines_y_coordinates, key=itemgetter(1)))

        plt.figure(1)
        plt.title(f'Graph-Analytical method: {"2xN" if is_2n_game else "Mx2"}')

        best_strategy_x, best_strategy_y = best_strategy

        for i, y in enumerate(lines_y_coordinates, start=1):
            plt.plot((0, 1), y)

        plt.plot(
            (best_strategy_x, best_strategy_x), (0, best_strategy_y),
            color='black', linestyle=':'
        )
        plt.plot(best_strategy_x, best_strategy_y, marker='o')

        plt.xlabel("x" if is_2n_game else "y")
        plt.ylabel("y" if is_2n_game else "x")

        plt.xlim(left=-0.01, right=1.01)
        plt.ylim(bottom=0)
        plt.yticks(np.arange(highest_y + 1))
        plt.xticks(np.arange(11) / 10)
        plt.grid()
        plt.show()


class GraphAnalyticalMethodRunner:
    """
    Find a mixed-strategy game solution using the graph-analytical method and
    the Neumann's theory.
    """

    def __init__(self, game_model: np.ndarray = None) -> None:
        self.game_model = game_model
        self.is_2n_game = False

    def _validate_args(self, cols: int, rows: int, c1: int, c2: int) -> None:
        """
        Validate the matrix limits provided by user.
        """
        rows = self.game_model.shape[0] if self.game_model is not None else rows
        cols = self.game_model.shape[1] if self.game_model is not None else cols

        if all([rows != 2, cols != 2]):
            raise Exception(
                f'Game matrix size should be either 2xN or Mx2. You provided '
                f'{rows}x{cols}.'
            )
        if c1 >= c2:
            raise Exception(
                'Upper limit for matrix values should be higher than the lower limit. '
                f'You provided {c1} >= {c2}'
            )

    def generate_matrix(self, rows: int, cols: int, lower_limit: int,
                        upper_limit: int) -> None:
        """
        Generate a random integer matrix of size :rows: x :cols: where each value
        is equal to/greater than :lower_limit: and equal to/less than :upper_limit:.
        """
        self.game_model = np.random.randint(
            low=lower_limit,
            high=upper_limit + 1,
            size=(rows, cols),
        )

    def find_best_strategy_point(self, potential_x_set: Set,
                                 line_functions: List[Callable]) -> List:
        """
        Find optimal xy solution based on the Neumann's theorem:
        the solution is maximin for 2xN games, and minimax for Mx2 games.
        """
        values = []
        for x in potential_x_set:
            values.append(
                min([[x, line_fn(x)] for line_fn in line_functions], key=lambda i: i[1])
                if self.is_2n_game
                else
                max([[x, line_fn(x)] for line_fn in line_functions], key=lambda i: i[1])
            )
        optimal_xy = (
            max(values, key=lambda i: i[1])
            if self.is_2n_game
            else
            min(values, key=lambda i: i[1])
        )
        return optimal_xy

    @classmethod
    def find_line_intersections(cls, lines: List[LineString]) -> Set:
        """
        Find all x values of the intersection points and points at limits of functions.
        """
        intersection_points_x_values = {0, 1}
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if intersection := lines[i].intersection(lines[j]):
                    intersection_points_x_values.add(intersection.x)
        return intersection_points_x_values

    def build_lines_for_plot(self) -> Tuple[List[LineString], List[Callable]]:
        """
        Generate LineString objects where each dot represents the 1st player's gains
        based on the 2nd player's strategies.
        """
        lines = []
        line_functions = []
        matrix = self.game_model.transpose() if self.is_2n_game else self.game_model

        for row in matrix:
            y1, y2 = row
            lines.append(
                LineString(
                    coordinates=[[0, y1], [1, y2]]
                )
            )
            line_functions.append(
                lambda x, y_values=row:
                y_values[0] * (1 - x) + y_values[1] * x
            )
        return lines, line_functions

    def run(self, m: int = 5, n: int = 4, c1: int = 2, c2: int = 21) -> None:
        """
        Run the graph-analytical method to solve a 2xN / Mx2 matrix
        :param m: number of rows (opt)
        :param n: number of columns (opt)
        :param c1: lower limit for matrix values (opt)
        :param c2: upper limit for matrix values (opt)
        """
        self._validate_args(n, m, c1, c2)
        if self.game_model is None:
            print(f'Generating the matrix of size {n}x{m}...')
            self.generate_matrix(m, n, c1, c2)
        self.is_2n_game = self.game_model.shape[0] == 2
        Printer.print_matrix(self.game_model)

        lines, functions = self.build_lines_for_plot()
        y_values_at_interval_start_and_end = [
            (line_fn(0), line_fn(1))
            for line_fn in functions
        ]

        x_values = self.find_line_intersections(lines)

        best_strategy_x, best_strategy_y = (
            self.find_best_strategy_point(x_values, functions)
        )
        p_strategy = 1 - best_strategy_x, best_strategy_x

        Printer.print_results(
            game_type='2xN' if self.is_2n_game else 'Mx2',
            optimal_strategy=p_strategy,
            game_value=best_strategy_y,
        )

        Plotter.plot(
            is_2n_game=self.is_2n_game,
            best_strategy=(best_strategy_x, best_strategy_y),
            lines_y_coordinates=y_values_at_interval_start_and_end,
        )


if __name__ == '__main__':
    # 2xN, saddle point = (1, 1)
    GraphAnalyticalMethodRunner(
        game_model=np.array([
            [9, 15, 21, 13],
            [2, 6, 19, 17],
        ])
    ).run()
    print('=' * 50)

    # 2xN, no saddle point
    GraphAnalyticalMethodRunner(
        game_model=np.array([
            [7, 5, 8, 10],
            [10, 11, 21, 9],
        ])
    ).run()
    print('=' * 50)

    # Mx2, saddle point = (5, 1)
    GraphAnalyticalMethodRunner(
        game_model=np.array([
            [2, 18],
            [3, 5],
            [9, 7],
            [11, 21],
            [15, 16],
        ])
    ).run()
    print('=' * 50)

    # Mx2, no saddle point
    GraphAnalyticalMethodRunner(
        game_model=np.array([
            [14, 3],
            [2, 21],
            [9, 16],
            [4, 2],
            [18, 6],
        ])
    ).run()

from typing import Tuple

import numpy as np
import pulp as plp


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
    def print_solutions(cls, x: np.ndarray, y: np.ndarray) -> None:
        """
        Print game solutions for two players.
        """
        print(
            f'Player 1 problem solution: x = {[round(value, 3) for value in x]}'
        )
        print(
            f'Player 2 problem solution: y = {[round(value, 4) for value in y]}'
        )

    @classmethod
    def print_game_metrics(cls, metrics: Tuple[float, np.ndarray, np.ndarray]) -> None:
        """
        Print V, V*, p, and q.
        """
        game_value, p, q = metrics

        print(f'V = {game_value}')
        print(f'p = {[round(value, 3) for value in p]}')
        print(f'q = {[round(value, 3) for value in q]}')


class LinearProgrammingProblemSolver:
    """
    Solve a game model based on the linear programming approach, in which a linear
    function is maximized or minimized when subjected to various constraints.
    """

    def __init__(self, game_model: np.ndarray = None) -> None:
        self.game_model = game_model
        self.k = 0

    @classmethod
    def _validate_args(cls, rows: int, cols: int, c1: int, c2: int) -> None:
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

    def normalize_game_model(self) -> None:
        """
        Add the same number to each value in the matrix to get rid of all negative
        values, as linear programming requires all values to be positive.
        """
        min_element = np.min(self.game_model)
        if min_element <= 0:
            self.k = -min_element + 1
            self.game_model += self.k

    def run(self, n: int = 4, m: int = 5, c1: int = -4, c2: int = 5) -> None:
        """
        :param n: number of rows (opt)
        :param m: number of columns (opt)
        :param c1: lower limit for matrix values (opt)
        :param c2: upper limit for matrix values (opt)
        """
        self._validate_args(n, m, c1, c2)
        if self.game_model is None:
            print(f'Generating the matrix of size {n}x{m}...')
            self.generate_matrix(n, m, c1, c2)
        Printer.print_matrix(self.game_model)

        self.normalize_game_model()
        if self.k != 0:
            print(f'\nNormalized game model (k={self.k})')
            Printer.print_matrix(self.game_model)
        print('\n')

        x = self.find_1st_player_best_strategy()
        print('=' * 5)
        y = self.find_2nd_player_best_strategy()
        print('=' * 5)
        Printer.print_solutions(x, y)

        metrics = self.find_game_metrics(x, y)
        Printer.print_game_metrics(metrics)

    @classmethod
    def find_game_metrics(cls, x: np.ndarray, y: np.ndarray) \
            -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Find V from sum(x) = 1/V => V = 1/sum(x).
        Find p from xi = pi/V => pi = V * xi.
        Find q from yi = qi/V => qi = V * yi.
        """
        game_value = 1 / sum(x)
        strategy_vector_1st_player = game_value * x
        strategy_vector_2nd_player = game_value * y
        return (
            game_value,
            strategy_vector_1st_player,
            strategy_vector_2nd_player,
        )

    def find_1st_player_best_strategy(self) -> np.ndarray:
        """
        Find Z = x1 + ... + xn -> min where xi >= 0, i in 1:n.
        """
        num_strategies = self.game_model.shape[0]   # rows

        problem = plp.LpProblem('P1', plp.LpMinimize)
        x_var = [
            plp.LpVariable(f'x{i}', lowBound=0)
            for i in range(1, num_strategies + 1)
        ]
        problem += plp.lpSum(x_var)
        for strategy in self.game_model.transpose():
            problem += (
                plp.lpSum([strategy[i] * x_var[i] for i in range(len(x_var))]) >= 1
            )
        problem.solve(plp.PULP_CBC_CMD())

        return np.array([variable.varValue for variable in problem.variables()])

    def find_2nd_player_best_strategy(self) -> np.ndarray:
        """
        Find F = y1 + ... + ym -> max where yi >= 0, j in 1:m.
        """
        number_of_strategies = self.game_model.shape[1]   # cols

        problem = plp.LpProblem('P2', plp.LpMaximize)
        y_var = [
            plp.LpVariable(f'y{i}', lowBound=0)
            for i in range(1, number_of_strategies + 1)
        ]
        problem += plp.lpSum(y_var)
        for strategy in self.game_model:
            problem += (
                plp.lpSum([strategy[i] * y_var[i] for i in range(len(y_var))]) <= 1
            )
        problem.solve(plp.PULP_CBC_CMD())

        return np.array([variable.varValue for variable in problem.variables()])


if __name__ == "__main__":
    LinearProgrammingProblemSolver(
        game_model=np.array([
            [2,  0,  1, -3,  4],
            [0, -2, -1,  4,  3],
            [2,  0,  3,  2, -1],
            [0, -3,  1,  0,  1],
            [3,  0,  0, -2,  2]
        ])
    ).run()
    print('=' * 50)

    LinearProgrammingProblemSolver(
        game_model=np.array([
            [-1, -2, 5,  4, -4],
            [ 5,  0, 2,  1,  2],
            [ 2, -3, 5, -4,  4],
            [ 3,  0, 4, -1,  0]
        ])
    ).run()
    print('=' * 50)

    LinearProgrammingProblemSolver(
        game_model=np.array([
            [1, 4, 3, 5, 2],
            [2, 3, 4, 1, 5],
            [1, 2, 3, 4, 2],
            [5, 3, 5, 1, 5]
        ])
    ).run()

from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt


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
    def print_iteration_results(cls, iteration: int, p: np.ndarray, q: np.ndarray,
                                alpha: float = 0.0, beta: float = 0.0, v: float = 0.0,
                                i: int = 1, j: int = 1) -> None:
        """
        Print the results calculated for the current iteration.
        """
        print('=' * 10)
        print(f'STEP {iteration}')
        print('=' * 10)
        if iteration == 1:
            print(f'p = {p}')
            print(f'q = {q}')
        else:
            print(f'α (max gain) = {alpha}, i = {i}, p = {p}')
            print(f'β (min loss) = {beta}, j = {j}, q = {q}')
            print(f'v = {v}')


class Plotter:
    """
    Helper class used to plot a graph.
    """

    @classmethod
    def plot(cls, number_of_iterations: int, empirical_distribution: List) -> None:
        """
        Plot a line graph representing the dependence of the approximated game price
        value on the iteration.
        """
        i = range(1, number_of_iterations + 1)
        v = empirical_distribution

        plt.figure(1)
        plt.plot(i, v)
        plt.xticks(i)
        plt.title('Brown-Robinson Method')
        plt.xlabel('Iteration')
        plt.ylabel('Empirical distribution (V)')
        plt.grid()
        plt.show()


class BrownRobinsonMethodRunner:
    """
    Find a mixed-strategy equilibrium.
    """

    def __init__(self, game_model: np.ndarray = None) -> None:
        self.game_model = game_model

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

    @classmethod
    def get_empirical_distribution(cls, gain: float, loss: float) -> float:
        """
        The empirical distribution for the current iteration is an arithmetic average of
        the 1st player's max gain and the 2nd player's min loss.
        """
        return (gain + loss) / 2

    def find_max_gain_for_1st_player(self, opponent_vector: np.ndarray) \
            -> Tuple[float, int]:
        """
        Calculate the 1st player's best strategy (max gain) and its index using
        the 2nd player's probability vector.
        """
        gains = np.dot(self.game_model, opponent_vector)

        max_gain = np.max(gains)
        max_loss_index = np.argwhere(max_gain == gains).flatten()[0]

        return max_gain, max_loss_index

    def find_min_loss_for_2nd_player(self, opponent_vector: np.ndarray) \
            -> Tuple[float, int]:
        """
        Calculate the 2nd player's best strategy (min loss) and its index using
        the 1st player's probability vector.
        """
        losses = np.dot(self.game_model.transpose(), opponent_vector)

        min_loss = np.min(losses)
        min_loss_index = np.argwhere(min_loss == losses).flatten()[0]

        return min_loss, min_loss_index

    @classmethod
    def find_next_strategy_vector(cls, current_vector: np.ndarray,
                                  current_iteration: int, next_index: int) -> np.ndarray:
        """
        Frequency of strategies are calculated with the formula: (STEP * p) / (STEP + 1).
        """
        next_vector = current_iteration * current_vector / (current_iteration + 1)
        next_vector[next_index] += 1 / (current_iteration + 1)
        return next_vector

    def run_first_game_iteration(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        During the first game iteration, both players choose random strategies.
        """
        return np.max(self.game_model, axis=1) / 10, np.min(self.game_model, axis=0) / 10

    def run_game_iteration_for_1st_player(self, iteration: int, self_vector: np.ndarray,
                                          opponent_vector: np.ndarray) \
            -> Tuple[int, float, np.ndarray]:
        """
        During each iteration, the 1st player chooses the max gain based on the 2nd
        player's vector of probable strategies, and then re-calculates their own
        strategy vector.
        """
        max_gain_1st_player, max_gain_index = (
            self.find_max_gain_for_1st_player(opponent_vector)
        )
        strategy_vector_1st_player = self.find_next_strategy_vector(
            current_vector=self_vector,
            current_iteration=iteration,
            next_index=max_gain_index,
        )
        return max_gain_index, max_gain_1st_player, strategy_vector_1st_player

    def run_game_iteration_for_2nd_player(self, iteration: int, self_vector: np.ndarray,
                                          opponent_vector: np.ndarray) \
            -> Tuple[int, float, np.ndarray]:
        """
        During each iteration, the 2nd player chooses the min loss based on the 1st
        player's vector of probable strategies, and then re-calculates their own
        strategy vector.
        """
        min_loss_2nd_player, min_loss_index = (
            self.find_min_loss_for_2nd_player(opponent_vector)
        )
        strategy_vector_2nd_player = self.find_next_strategy_vector(
            current_vector=self_vector,
            current_iteration=iteration,
            next_index=min_loss_index,
        )
        return min_loss_index, min_loss_2nd_player, strategy_vector_2nd_player

    def run_game_iteration(self, iteration: int, strategy_vector_1st_player: np.ndarray,
                           strategy_vector_2nd_player: np.ndarray) \
            -> Tuple[Tuple[int, float, np.ndarray], Tuple[int, float, np.ndarray]]:
        """
        During each iteration, the 1st player chooses the max gain based on the 2nd
        player's vector of probable strategies, and the 2nd player chooses the min loss
        based on the 1st player's vector of probable strategies. Then both players
        re-calculate their probable strategies vectors for the next iteration.
        """
        return (
            self.run_game_iteration_for_1st_player(
                iteration, strategy_vector_1st_player, strategy_vector_2nd_player
            ),
            self.run_game_iteration_for_2nd_player(
                iteration, strategy_vector_2nd_player, strategy_vector_1st_player
            ),
        )

    def run(self, n: int = 5, m: int = 4, c1: int = -4, c2: int = 5,
            iterations: int = 10) -> None:
        """
        Run N iterations of the Brown-Robinson method for a game matrix
        :param n: number of columns (opt)
        :param m: number of rows (opt)
        :param c1: lower limit for matrix values (opt)
        :param c2: upper limit for matrix values (opt)
        :param iterations: number of iterations for the Brown-Robinson method (opt)
        """
        self._validate_args(n, m, c1, c2)
        if self.game_model is None:
            print(f'Generating the matrix of size {n}x{m}...')
            self.generate_matrix(m, n, c1, c2)
        Printer.print_matrix(self.game_model)

        empirical_distributions = []
        p = q = None
        for current_iteration in range(1, iterations + 1):
            if current_iteration == 1:
                p, q = self.run_first_game_iteration()
                max_gain = min_loss = max_gain_index = min_loss_index = v = 0.0
            else:
                iteration_results_1st_player, iteration_results_2nd_player = (
                    self.run_game_iteration(
                        iteration=current_iteration,
                        strategy_vector_1st_player=p,
                        strategy_vector_2nd_player=q,
                    )
                )
                max_gain_index, max_gain, p = iteration_results_1st_player
                min_loss_index, min_loss, q = iteration_results_2nd_player
                v = self.get_empirical_distribution(max_gain, min_loss)

            empirical_distributions.append(v)

            Printer.print_iteration_results(
                iteration=current_iteration,
                alpha=round(max_gain, 2), beta=round(min_loss, 2),
                i=max_gain_index, j=min_loss_index, p=p, q=q,
                v=round(v, 2),
            )

        Plotter.plot(iterations, empirical_distributions)


if __name__ == '__main__':
    # random
    BrownRobinsonMethodRunner().run()
    print('=' * 50)

    # own matrix
    BrownRobinsonMethodRunner(
        game_model=np.array([
          [-3, 5, -2, -1, -3],
          [0, 1, -1, 4, 5],
          [-3, -2, 4, 3, -4],
          [5, -4, 5, 2, 3],
        ])
    ).run()
    print('=' * 50)

import math
from typing import Dict, List


class CooperativeGameSolver:
    """
    Find a cooperative game solution using Shapley vector.
    """

    def __init__(self, characteristic_function_v: Dict[str, int]) -> None:
        self.V = characteristic_function_v
        self.normalized_V = dict()
        self.shapley_vector = dict()
        self.number_of_players = len(max(self.V, key=len))

    @property
    def all_coalitions(self) -> List[str]:
        return list(self.V.keys())

    @property
    def coalitions_of_players(self) -> List[str]:
        """
        All combinations of coalitions that could be formed by players, EXCLUDING
        the max coalition (see @prop players).
        """
        return [x for x in self.all_coalitions if self.number_of_players > len(x) > 1]

    @property
    def max_coalition(self) -> str:
        """
        Max coalition that could be formed by all the players.
        """
        return next(
            filter(
                lambda x: len(x) == self.number_of_players, self.all_coalitions
            )
        )

    @property
    def players(self) -> List[str]:
        """
        Return a list of players.
        """
        return [*self.max_coalition]

    @property
    def empty_coalition(self) -> str:
        """
        Returns ''.
        """
        return next(
            filter(
                lambda x: x == '', self.all_coalitions
            )
        )

    @property
    def is_super_additive(self) -> bool:
        """
        Check if the game is super additive.
        :return: True if true for every possible coalition, except the max coalition:
        \n '''
        \n V(COALITION) >= sum(V for all players in coalition)
        \n '''
        """
        additivity_for_coalition = list()
        for players in self.coalitions_of_players:
            additivity_for_coalition.append(
                is_super_additive :=
                self.V[players] >= self.sum_coalition(players)
            )
            if not is_super_additive:
                print(
                    'False because: '
                    f'V({players}) = {self.V[players]} < '
                    f'{" + ".join(f"V({player})" for player in players)} = '
                    f'{" + ".join(str(self.V[player]) for player in players)}'
                )
        return all(additivity_for_coalition)

    @property
    def is_essential(self) -> bool:
        """
        Check if game is essential.
        :return: True if:
        \n '''
        \n V(MAX_COALITION) < sum(V for all players)
        \n '''
        """
        is_essential = (
            self.V[self.max_coalition] < self.sum_coalition(self.max_coalition)
        )
        if not is_essential:
            print(
                'False because: '
                f'V({self.max_coalition}) = {self.V[self.max_coalition]} >= '
                f'{" + ".join(f"V({player})" for player in self.players)} = '
                f'{" + ".join(str(self.V[player]) for player in self.players)}'
            )
        return is_essential

    @property
    def is_core_empty(self) -> bool:
        """
        Check if the game core is empty.
        :return: True if true for every possible coalition:
        \n '''
        \n V'(COALITION) <= 1 / (NUM_OF_PLAYERS - NUM_IN_COALITION + 1)
        \n '''
        """
        is_core_per_coalition = list()

        for coalition in self.all_coalitions:
            if coalition:
                is_core_per_coalition.append(
                    is_core :=
                    self.normalized_V[coalition] <=
                    1 / (self.number_of_players - len(coalition) + 1)
                )
                print(
                    f"V'({''.join(coalition)}) = "
                    f"{self.normalized_V[coalition]:.2f} <= "
                    f"{1 / (self.number_of_players - len(coalition) + 1):.2f} "
                    f"{is_core}"
                )
        return not all(is_core_per_coalition)

    @property
    def is_shapley_vector_belongs_to_core(self) -> bool:
        """
        Check if Shapley vector belongs to the core.
        :return: True if true for every possible coalition:
        \n '''
        \n V(COALITION) <= sum(ф(V) for all players in coalition)
        \n '''
        """
        is_core_check_per_coalition = list()
        for coalition in self.all_coalitions:
            if coalition:
                sum_ = sum([self.shapley_vector[player] for player in coalition])
                is_core_check_per_coalition.append(
                    is_core :=
                    any(
                        [self.V[coalition] <= sum_, self.V[coalition] - sum_ < 0.1]
                    )
                )
                if not is_core:
                    print(
                        'False because: '
                        f'V({coalition}) = '
                        f'{self.V[coalition]} > '
                        f'{sum([self.shapley_vector[player] for player in coalition])} '
                    )
        return all(is_core_check_per_coalition)

    def get_normalized_form(self) -> None:
        """
        Find the normalized forms (V'). Calculated for every possible coalition as:
        \n '''
        \n V(COALITION) - sum(V for players in coalition) /
        \n V(MAX_COALITION) - sum(V for all players)
        \n '''
        \n V'() = 0, V'(player) = 0, V'(MAX_COALITION) = 1
        """
        self.normalized_V[self.empty_coalition] = 0
        self.normalized_V[self.max_coalition] = 1
        for player in self.players:
            self.normalized_V[player] = 0

        for coalition in self.coalitions_of_players:
            self.normalized_V[coalition] = (
                (self.V[coalition] - self.sum_coalition(coalition)) /
                (self.V[self.max_coalition] - self.sum_coalition(self.max_coalition))
            )

    def get_coalitions_by_player(self, player: str) -> List[str]:
        """
        Get all the coalitions where the player is.
        """
        return [x for x in self.all_coalitions if player in x]

    @classmethod
    def get_coalition_without_player(cls, coalition: str, player: str) -> str:
        """
        Remove the player from the coalition, e.g. ('123', '1') -> '23'.
        """
        return coalition.replace(player, '')

    def sum_coalition(self, coalition: str) -> int:
        """
        Get the sum of values for all the players in a coalition.
        """
        return sum([self.V[player] for player in coalition])

    def calculate_shapley_value(self, player: str) -> float:
        """
        Calculate Shapley value per player. For all the coalitions where the player is
        (including V(1)), sum values of:
        \n '''
        \n V(COALITION) - V(COALITION_WITHOUT_PLAYER) *
        \n (NUM_IN_COALITION - 1)! * (NUM_OF_PLAYERS - NUM_IN_COALITION)! /
        \n NUM_OF_PLAYERS!
        \n '''
        """
        return sum(
            (
                self.V[coalition] -
                self.V[self.get_coalition_without_player(coalition, player)]
            ) *
            (
                (
                    math.factorial(len(coalition) - 1) *
                    math.factorial(self.number_of_players - len(coalition))
                ) /
                (
                    math.factorial(self.number_of_players)
                )
            )
            for coalition in self.get_coalitions_by_player(player)
        )

    def run(self) -> None:
        """
        Solve the cooperative game.
        """
        print('Characteristic function:')
        print(self.V)

        print('\nSuper-additivity:')
        print('is_super_additive:', self.is_super_additive)

        print('\nGame essentiality:')
        print('is_essential:', self.is_essential)

        print('\nNormalized form:')
        self.get_normalized_form()

        for coalition in self.all_coalitions:
            print(f"V'({coalition}) = {self.normalized_V[coalition]:.2f}")

        print('\nCore:')
        print('is_core_empty', self.is_core_empty)

        print('\nShapley vector:')
        for player in self.players:
            self.shapley_vector[player] = (
                self.calculate_shapley_value(player)
            )
            print(f"ф{player}(V) = {self.shapley_vector[player]:.2f}")

        print('\nShapley vector belonging to the core:')
        print(
            'is_shapley_vector_belongs_to_core',
            self.is_shapley_vector_belongs_to_core
        )


if __name__ == "__main__":
    CooperativeGameSolver(
        characteristic_function_v={
            '': 0,
            '1': 2000,
            '2': 1800,
            '3': 1500,
            '12': 4200,
            '13': 4000,
            '23': 3700,
            '123': 6400,
        },
    ).run()

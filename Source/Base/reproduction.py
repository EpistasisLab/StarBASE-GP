#####################################################################################################
#
# Reproduction base class for generating offspring pipelines.
# A combination of mutation and crossover are used to generate pipelines (both user specified).
# Users must provide their own implementation of this class as each branch requires different data.
# Users must provide their own implementation of the HUB class as each branch requires different data.
#
#####################################################################################################

from typeguard import typechecked
from typing import List, Tuple, Set, Dict
from .pipeline import Pipeline
from .types import (rng_t, prob_t, int32_t, uint16_t, snp_t, uint32_t)
from abc import ABC, abstractmethod
from .hub import Hub
import time

@typechecked
class Reproduction(ABC):
    def __init__(self,
                 branch_max: uint16_t,
                 branch_min: uint16_t,
                 mut_prob: prob_t = prob_t(.5),
                 cross_prob: prob_t = prob_t(.5),
                 mut_selector_p: prob_t = prob_t(.5),
                 mut_ld_p: prob_t = prob_t(.5),
                 mut_ran_p: prob_t = prob_t(.45),
                 m_in_win_p: prob_t = prob_t(.1),
                 m_out_win_p: prob_t = prob_t(.45),
                 m_out_chr_p: prob_t = prob_t(.45),
                 window_distance: int32_t = int32_t(1000000),
                 timings_list: List[str] = ['in_window', 'out_window', 'out_chrom']) -> None:

        # save all the variables
        self.branch_max = branch_max
        self.branch_min = branch_min
        self.mut_prob = mut_prob
        self.cross_prob = cross_prob
        self.mut_selector_p = mut_selector_p
        self.mut_ld_p = mut_ld_p
        self.mut_ran_p = mut_ran_p
        self.m_in_win_p = m_in_win_p
        self.m_out_win_p = m_out_win_p
        self.m_out_chr_p = m_out_chr_p
        self.window_distance = window_distance

        # Dictionary to track mutation timing statistics
        self.mutation_timings: Dict[str, List[float]] = {var: [] for var in timings_list}
        print(f'mutation_timings initialized with keys: {list(self.mutation_timings.keys())}')

        return

    @abstractmethod
    def generate_random_pipeline(self, rng: rng_t, branches: Set, seed: int) -> Pipeline:
        """
        Function to generate a random pipeline during the initialization of the population.

        Parameters:
            rng (rng_t): A numpy random number generator from the evolver
            branches (Set): A set of branches to add to the pipeline (univariate snps or interactions (K2, K3, ...))
            seed (int): A seed to use for random_states within pipeline selector/ld nodes (if needed)
        """
        pass

    def variation_order(self, rng: rng_t, offspring_cnt: uint32_t) -> Tuple[List[snp_t], uint32_t]:
        """
        Generate the order of variation operators to be applied to generate offspring.
        The order is determined by the probabilities of mutation and crossover.
        We return a list with the names of the operators in the order they should be applied.
        E.g.: ['m', 'c', 'm', 'c', ...]

        Crossover means two parents are required
        Mutation means one parent is required

        Parameters:
            rng (rng_t): A numpy random number generator from the evolver
            offpring_cnt (pop_size_t): The number of offspring to generate

        Returns:
            List[snp_t]: A list of strings representing the order of variation operators to be applied
            pop_size_t: The number of parents needed to generate the offspring
        """
        # parents needed by variantion operators
        parent_count = {'m': 1, 'c': 2}

        # generate the order of variation operations
        order = rng.choice(['m', 'c'], offspring_cnt, p=[self.mut_prob, self.cross_prob])
        order = [snp_t(op) for op in order]

        # make sure we have the right number of offspring
        assert len(order) == offspring_cnt

        # return the order and number of parents needed
        return order, uint32_t(sum(parent_count[op] for op in order))

    def produce_offspring(self,
                          rng: rng_t,
                          hub: Hub,
                          offspring_cnt: uint32_t,
                          population: List[Pipeline],
                          parent_ids: List[uint32_t], # should be
                          order: List[snp_t]) -> List[Pipeline]:
        """
        Generate offspring pipelines based on the given order of variation operations.

        Parameters:
            rng (rng_t): A numpy random number generator from the evolver
            hub: An interface to a branch hub to get branch specific information
            offspring_cnt (uint16_t): The number of offspring to generate
            population (List[Pipeline]): The current population of pipelines
            parent_ids (List[uint16_t]): The list of parent IDs to use for generating offspring
            order (List[snp_t]): The order of variation operations to apply

        Returns:
            List[Pipeline]: The list of generated offspring pipelines
        """

        # quick checks
        assert len(parent_ids) > 0
        assert len(population) > 0
        assert offspring_cnt > 0

        # list to store the offspring
        offspring = []

        # go through the order of operators
        p_id = 0
        for op in order:
            # mutation only
            if op == 'm':
                off = self.mutate(rng, population[parent_ids[p_id]], hub)
                offspring.append(off)
                p_id += 1
            # crossover only
            elif op == 'c':
                off = self.crossover(rng, population[parent_ids[p_id]], population[parent_ids[p_id+1]], hub)

                # coin flip to decide if we should mutate the offspring
                if rng.choice([True, False], p=[self.mut_prob, 1.0-self.mut_prob]):
                    off = self.mutate_post_crossover(rng, off, hub)

                offspring.append(off)
                p_id += 2
            else:
                raise ValueError(f"Unknown operator: {op}")

        # make sure we have the right number of offspring
        assert len(offspring) == offspring_cnt
        assert p_id == len(parent_ids)

        # Print average timing statistics for mutation types
        if any(len(times) > 0 for times in self.mutation_timings.values()):
            print("\n=== Mutation Timing Statistics ===")
            for mutation_type, times in self.mutation_timings.items():
                if len(times) > 0:
                    avg_time = sum(times) / len(times)
                    print(f"{mutation_type}: {avg_time*1000:.4f} ms (n={len(times)})")
            print("==================================\n", flush=True)

        # set the mutation timings back to empty lists
        self.mutation_timings = {key: [] for key in self.mutation_timings}

        # clear out nearest neighbor cache in hub if applicable
        hub.consider.clear_nearest_neighbor()

        # return the offspring
        return offspring

    @abstractmethod
    def mutate(self, rng: rng_t, parent: Pipeline, hub: Hub) -> Pipeline:
        """
        Function to mutate a given parent pipeline.

        Parameters:
            rng (rng_t): A numpy random number generator from the evolver
            parent (Pipeline): The parent pipeline to be mutated
            hub: An interface to a branch hub to get branch specific information

        Returns:
            Pipeline: The mutated offspring pipeline
        """
        pass

    @abstractmethod
    def mutate_post_crossover(self, rng: rng_t, offspring: Pipeline, hub: Hub) -> Pipeline:
        """
        Function to mutate an offspring post being generated from crossover operation.
        Note that this mutation may be different from the standard mutation operation.
        I.e., offspring passed will not have trait features set.

        Parameters:
            rng (rng_t): A numpy random number generator from the evolver
            offspring (Pipeline): The offspring pipeline generated from crossover operation
            hub: An interface to a branch hub to get branch specific information

        Returns:
            Pipeline: The mutated offspring pipeline
        """
        pass

    @abstractmethod
    def crossover(self, rng: rng_t, parent1: Pipeline, parent2: Pipeline, hub: Hub) -> Pipeline:
        """
        Function to perform crossover between two parents to generate an offspring pipeline.
        Assuming that these parents have been evaluated so can use trait_feature_names branches.

        Parameters:
            rng (rng_t): A numpy random number generator from the evolver
            parent1 (Pipeline): The first parent pipeline
            parent2 (Pipeline): The second parent pipeline
            hub: An interface to a branch hub to get branch specific information

        Returns:
            Pipeline: The offspring pipeline generated from the two parents
        """
        pass

    def num_branches_to_add(self, rng: rng_t, branches: Set) -> uint16_t:
        """
        Calculate the number of branches to add within a mutated pipeline.
        Standardized for all derived classes to use

        Args:
            rng (rng_t): A numpy random number generator from the evolver
            branches (Set): A set of branches in the pipeline

        Returns:
            uint16_t: The number of branches to add
        """

        # quick checks
        assert len(branches) <= self.branch_max
        assert self.branch_max - len(branches) >= 0

        # get a number of interactions to add based on self.branch_max and self.branch_min
        if len(branches) < self.branch_min:
            return uint16_t(rng.integers(self.branch_min - len(branches), self.branch_max - len(branches), endpoint=False))
        else:
            num_add_range = uint16_t(max(self.branch_max - len(branches), 0))

        # if 0 or 1 just return the number
        if num_add_range == 0 or num_add_range == 1:
            return uint16_t(num_add_range)
        # else pick a number between the range and 1 (range < self.num_add_interactions)
        else:
            return uint16_t(rng.integers(1, num_add_range, endpoint=False))
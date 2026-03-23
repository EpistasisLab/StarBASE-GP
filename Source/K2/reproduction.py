#####################################################################################################
#
# Reproduction class that generating new pipelines for K1 regression.
# We use a combination of mutation and crossover to generate new pipelines (both user specified).
# We also initialize the first population of solutions (this is reproduction w/out parents).
#
# For coding purposes, switch Hub to K2_Hub once all functionality is done.
# This will ensure that all base functionality is correct and help fill K1 specific functionality.
# Switching back to Hub will remove incorrect dependencies on K1 specific code in the Base folder.
#
#####################################################################################################

# imports from Base
from ..Base.pipeline import Pipeline
from ..Base.types import (rng_t, prob_t, int32_t, uint16_t, snp_t, interaction_t)
from ..Base.reproduction import Reproduction
from ..Base.selectors import (VarianceThresholdNode, SelectPercentileNode, SelectFweNode, SelectFromModelLasso,
                              SelectFromModelTree, FeatureEncodingFrequencySelector)

# imports from K1
from .ld_selector import LDSelector
from .epi_hub import K2_Hub

# additional imports
from typeguard import typechecked
from typing import Set
import numpy as np
import copy as cp
# Start timing
import time


@typechecked
class K2_Reproduction(Reproduction):
    def __init__(self,
                 branch_max: uint16_t,
                 branch_min: uint16_t,
                 mut_prob: prob_t = prob_t(.5),
                 cross_prob: prob_t = prob_t(.5),
                 mut_selector_p: prob_t = prob_t(.5),
                 mut_ld_p: prob_t = prob_t(.5),
                 mut_ran_p: prob_t = prob_t(.3),
                 mut_neighbor_p: prob_t = prob_t(.5),
                 m_keep_left: prob_t = prob_t(.33),
                 m_keep_right: prob_t = prob_t(.33),
                 m_climb_both: prob_t = prob_t(.33),
                 mut_ioc_p: prob_t = prob_t(.2),
                 m_in_win_p: prob_t = prob_t(0.0),
                 m_out_win_p: prob_t = prob_t(.5),
                 m_out_chr_p: prob_t = prob_t(.5),
                 window_distance: int32_t = int32_t(1000000)) -> None:
        """
        K2 Reproduction class that extends the Base Reproduction class.

        Parameters:
            branch_max (uint16_t): The maximum number of branches in a pipeline
            branch_min (uint16_t): The minimum number of branches in a pipeline
            mut_prob (prob_t): The probability of mutation
            cross_prob (prob_t): The probability of crossover
            mut_selector_p (prob_t): The probability of mutating the selector node
            mut_ld_p (prob_t): The probability of mutating the ld node
            mut_regressor_p (prob_t): The probability of mutating the regressor node (not implemented yet)
            mut_ran_p (prob_t): The probability of mutating to a completely random interaction
            mut_neighbor_p (prob_t): The probability of mutating within the neighborhood of the current interaction
            m_keep_left (prob_t): The probability of keeping the left snp in an interaction and mutating the right snp
            m_keep_right (prob_t): The probability of keeping the right snp in an interaction and mutating the left snp
            m_climb_both (prob_t): The probability of mutating both snps in an interaction and keeping the interaction (i.e., climbing both snps)
            mut_ioc_p (prob_t): The probability of picking a new snp from either in or out of the chromosome to include within an interaction
            m_in_win_p (prob_t): The probability of picking a new snp from the same chromosome and within the window of a snp (NOT USED)
            m_out_win_p (prob_t): The probability of picking a new snp from the same chromosome
            m_out_chr_p (prob_t): The probability of picking a new snp from a different chromosome from the anchor snp when mutating within the neighborhood
            window_distance (int32_t): The distance in base pairs for the interaction window when mutating within the neighborhood
            mut_smt_p (prob_t): The probability of mutating to a completely random interaction from the hub
        """

        # additional probabilities for K2 specific mutation operations
        self.mut_neighbor_p = mut_neighbor_p
        self.m_keep_left = m_keep_left
        self.m_keep_right = m_keep_right
        self.m_climb_both = m_climb_both
        self.mut_ioc_p = mut_ioc_p


        # pass all variables to the Base Reproduction class
        super().__init__(branch_max=branch_max,
                         branch_min=branch_min,
                         mut_prob=mut_prob,
                         cross_prob=cross_prob,
                         mut_selector_p=mut_selector_p,
                         mut_ld_p=mut_ld_p,
                         mut_ran_p=mut_ran_p,
                         m_in_win_p=m_in_win_p,
                         m_out_win_p=m_out_win_p,
                         m_out_chr_p=m_out_chr_p,
                         window_distance=window_distance)

        # normalize the mut_ran_p, mut_neighbor_p, and mut_ioc_p to ensure they sum to 1
        total = mut_ran_p + mut_neighbor_p + mut_ioc_p
        self.mut_ran_p = mut_ran_p / total
        self.mut_neighbor_p = mut_neighbor_p / total
        self.mut_ioc_p = mut_ioc_p / total

        print(f'mut_ran_p: {self.mut_ran_p}, mut_neighbor_p: {self.mut_neighbor_p}, mut_ioc_p: {self.mut_ioc_p}')

        # normalize the m_keep_left, m_keep_right, and m_climb_both to ensure they sum to 1
        total = m_keep_left + m_keep_right + m_climb_both
        self.m_keep_left = m_keep_left / total
        self.m_keep_right = m_keep_right / total
        self.m_climb_both = m_climb_both / total

        # normalize the m_in_win_p, m_out_win_p, and m_out_chr_p to ensure they sum to 1
        total = m_in_win_p + m_out_win_p + m_out_chr_p
        self.m_in_win_p = m_in_win_p / total
        self.m_out_win_p = m_out_win_p / total
        self.m_out_chr_p = m_out_chr_p / total

        return

    def generate_random_pipeline(self, rng: rng_t, branches: Set, seed: int) -> Pipeline:
        """
        Function to generate a random pipeline during the initialization of the population.

        Parameters:
            rng (rng_t): A numpy random number generator from the evolver
            branches (Set): A set of branches to add to the pipeline (univariate snps or interactions (K2, K3, ...))
            seed (int): A seed to use for random_states within pipeline selector/ld nodes (if needed)

        Returns:
            Pipeline: A randomly generated pipeline
        """
        # quick checks
        assert len(branches) > 0, "Branches set cannot be empty."
        assert seed >= 0, "Seed must be non-negative."

        # selector options to choose from
        selector_choice = rng.choice([0,1,2,3,4,5])

        if selector_choice == 0: # variance threshold
            return Pipeline(branch_set=branches, ld_node=LDSelector(rng=rng), selector_node=VarianceThresholdNode(rng=rng))
        elif selector_choice == 1: # select percentile
            return Pipeline(branch_set=branches, ld_node=LDSelector(rng=rng), selector_node=SelectPercentileNode(rng=rng))
        elif selector_choice == 2: # select fwe
            return Pipeline(branch_set=branches, ld_node=LDSelector(rng=rng), selector_node=SelectFweNode(rng=rng))
        elif selector_choice == 3: # select from model lasso
            return Pipeline(branch_set=branches, ld_node=LDSelector(rng=rng), selector_node=SelectFromModelLasso(rng=rng, seed=seed))
        elif selector_choice == 4: # select from model tree
            return Pipeline(branch_set=branches, ld_node=LDSelector(rng=rng), selector_node=SelectFromModelTree(rng=rng, seed=seed))
        else: # feature encoding frequency selector
            return Pipeline(branch_set=branches, ld_node=LDSelector(rng=rng), selector_node=FeatureEncodingFrequencySelector(rng=rng))

    def mutate(self, rng: rng_t, parent: Pipeline, hub: K2_Hub) -> Pipeline:
        """
        Function to mutate a given parent pipeline.

        Parameters:
            rng (rng_t): A numpy random number generator from the evolver
            parent (Pipeline): The parent pipeline to be mutated
            hub: An interface to a branch hub to get branch specific information

        Returns:
            Pipeline: The mutated pipeline and the number of mutations applied
        """
        # quick checks: make sure parent has at least one branch
        assert len(parent.get_trait_feature_names()) > 0, "Parent pipeline must have at least one branch to mutate."

        # get branches from parent and make a deep copy
        parent_branches = cp.deepcopy(hub.remove_inactive_branches(parent.get_trait_feature_names()))
        assert len(parent_branches) > 0, "After removing inactive branches, parent pipeline must have at least one branch to mutate."

        # number of branches to add
        num_to_add = self.num_branches_to_add(rng, parent_branches)

        # list of anchor snps to append within the parent branches (could be duplicates)
        # Note: rng.choice returns a 2D numpy array when selecting from tuples, so we need to convert each row back to tuple
        anchor_snps = rng.choice(list(parent_branches), size=num_to_add, replace=True)
        for anchor in anchor_snps:
            anchor_tuple = tuple(anchor) if not isinstance(anchor, tuple) else anchor
            parent_branches.add(self.mutate_branch(rng, anchor_tuple, hub))

        # generate offspring with mutated parent branches and pass through selector and ld nodes from parent
        offspring = Pipeline(branch_set=parent_branches, ld_node=cp.deepcopy(parent.ld_node), selector_node=cp.deepcopy(parent.selector_node))

        # roll to mutate the ld node (use faster binary random)
        if rng.random() < self.mut_ld_p:
            offspring.mutate_ld_node(rng)
        # roll to mutate the selector node
        if rng.random() < self.mut_selector_p:
            offspring.mutate_selector_node(rng)

        return offspring

    def mutate_post_crossover(self, rng: rng_t, offspring: Pipeline, hub: K2_Hub) -> Pipeline:
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

        # quick checks: make sure offspring has at least one branch
        assert len(offspring.get_branch_set()) > 0, "Offspring pipeline must have at least one branch to mutate."
        # make sure none of the branches are inactive
        assert len(hub.remove_inactive_branches(offspring.get_branch_set())) == len(offspring.get_branch_set()), "Offspring pipeline must have all active branches at the start."

        # offspring branches
        offspring_new_branches = cp.deepcopy(offspring.get_branch_set())
        # number of branches to add
        num_to_add = self.num_branches_to_add(rng, offspring_new_branches)

        # list of anchor snps to append within the offspring branches (could be duplicates)
        # Note: rng.choice returns a 2D numpy array when selecting from tuples, so we need to convert each row back to tuple
        anchor_snps = rng.choice(list(offspring_new_branches), size=num_to_add, replace=True)
        for anchor in anchor_snps:
            anchor_tuple = tuple(anchor) if not isinstance(anchor, tuple) else anchor
            offspring_new_branches.add(self.mutate_branch(rng, anchor_tuple, hub))
        assert len(offspring.get_branch_set()) <= len(offspring_new_branches) <= len(offspring.get_branch_set()) + num_to_add \
            ,"Offspring branches after mutation must be correct size."

        # generate new offspring with mutated branches and pass through selector and ld nodes from original offspring
        offspring = Pipeline(branch_set=offspring_new_branches, ld_node=cp.deepcopy(offspring.ld_node), selector_node=cp.deepcopy(offspring.selector_node))

        # roll to mutate the ld node (use faster binary random)
        if rng.random() < self.mut_ld_p:
            offspring.mutate_ld_node(rng)
        # roll to mutate the selector node
        if rng.random() < self.mut_selector_p:
            offspring.mutate_selector_node(rng)

        return offspring

    def mutate_branch(self, rng: rng_t, branch: interaction_t, hub: K2_Hub) -> interaction_t:
        """
        Function to mutate a given anchor branch interaction.

        Parameters:
            rng (rng_t): A numpy random number generator from the evolver
            branch (interaction_t): The branch interaction to mutate
            hub: An interface to a branch hub to get branch specific information

        Returns:
            interaction_t: The mutated branch interaction
        """

        # quick checks
        assert hub.get_active_flag(branch), "Anchor branch must be active in the hub to mutate"

        mutation_type = None  # for timing purposes
        pair = None
        start_time = time.time()

        # roll to see if we do a neighborhood mutation
        mut_roll = rng.random()
        if mut_roll < self.mut_neighbor_p:
            mutation_type = 'in_window'
            roll = rng.random()
            # keep left and mutate right via neighborhood
            if roll < self.m_keep_left:
                replace = hub.get_ran_snp_in_window(branch[1], rng)
                pair = (branch[0], replace) if branch[0] < replace else (replace, branch[0])
            # keep right and mutate left via neighborhood
            elif roll < self.m_keep_left + self.m_keep_right:
                replace = hub.get_ran_snp_in_window(branch[0], rng)
                pair = (branch[1], replace) if branch[1] < replace else (replace, branch[1])
            # replace both via neighborhood (i.e., climb both)
            else: # climb both
                left_replace = hub.get_ran_snp_in_window(branch[0], rng)
                right_replace = hub.get_ran_snp_in_window(branch[1], rng)
                pair = (left_replace, right_replace) if left_replace < right_replace else (right_replace, left_replace)

            # if pair consists of the same snp get a random interaction
            # this can only happen in the keep left or keep right scenarios if the snp we are mutating is in a very tight cluster of snps
            if pair[0] == pair[1]:
                mutation_type = 'new_pair'
                print(f"Warning: Mutated pair {pair} consists of the same SNP. This can happen in tight clusters when mutating within the neighborhood. Getting a random interaction from the hub instead.")
                for _ in range(hub.mutation_tries):
                    # will automatically ensure that the same snp is not returned as a pair
                    new_pair = hub.get_ran_interaction(rng, branch)

                    # if pair and new_pair are the same, try again
                    if new_pair == branch:
                        continue

                    # have we seen this new_pair before in the hub
                    if hub.does_interaction_exist(new_pair):
                        # if so and not active, try again
                        if hub.get_active_flag(new_pair) == False:
                            continue
                        # if so and active, we can roll with it
                        else:
                            assert new_pair[0] != new_pair[1], f"Mutated SNPs in neighborhood mutation should not be the same. Got new_pair: {new_pair} from branch: {branch}"
                            pair = new_pair
                            break
                    # if not seen before, we can add it to the hub and return it
                    else:
                        assert new_pair[0] != new_pair[1], f"Mutated SNPs in neighborhood mutation should not be the same. Got new_pair: {new_pair} from branch: {branch}"
                        pair = new_pair
                        break

        # roll to see if we are doing an in/out chromosome mutation
        elif mut_roll < self.mut_neighbor_p + self.mut_ioc_p:
            # roll to pick which snp in the interaction we want to keep
            anchor_snp = rng.choice(list(branch))

            # perform mutation based on type
            roll = rng.random()
            if roll < self.m_out_win_p:
                result = hub.get_ran_snp_in_chrm(anchor_snp, rng)
                mutation_type = 'out_window'
            else: # out_chrom
                result = hub.get_ran_snp_out_chrm(anchor_snp, rng)
                mutation_type = 'out_chrom'

            assert result != anchor_snp, f"Mutated SNP should not be the same as the anchor SNP. Got result: {result} and anchor_snp: {anchor_snp}"
            pair = (anchor_snp, result) if anchor_snp < result else (result, anchor_snp)
            assert pair[0] != pair[1], f"Mutated SNPs in in/out chromosome mutation should not be the same. Got pair: {pair} from branch: {branch} with anchor_snp: {anchor_snp}"

        else:
            # return a completely random interaction from the hub
            pair = hub.get_ran_interaction(rng, branch)
            mutation_type = 'new_pair'
            assert pair[0] != pair[1], f"Mutated SNPs in random interaction mutation should not be the same. Got pair: {pair} from branch: {branch}"

        # if we have a pair that we have seen before in the hub but is not active, get random interaction
        if hub.does_interaction_exist(pair):
            if hub.get_active_flag(pair) == False:
                mutation_type = 'new_pair'
                for _ in range(hub.mutation_tries):
                    new_pair = hub.get_ran_interaction(rng, branch)

                    # if pair and new_pair are the same, try again
                    if new_pair == pair:
                        continue

                    # have we seen this new_pair before in the hub
                    if hub.does_interaction_exist(new_pair):
                        # if so and not active, try again
                        if hub.get_active_flag(new_pair) == False:
                            continue
                        # if so and active, we can roll with it
                        else:
                            assert new_pair[0] != new_pair[1], f"Mutated SNPs in random interaction mutation should not be the same. Got new_pair: {new_pair} from branch: {branch}"
                            pair = new_pair
                    # if not seen before, we can add it to the hub and return it
                    else:
                        assert new_pair[0] != new_pair[1], f"Mutated SNPs in random interaction mutation should not be the same. Got new_pair: {new_pair} from branch: {branch}"
                        pair = new_pair

        # Record timing
        elapsed_time = time.time() - start_time
        self.mutation_timings[mutation_type].append(elapsed_time)
        assert pair[0] != pair[1], f"Mutated SNPs should not be the same. Got pair: {pair} from branch: {branch}"
        return pair

    def crossover(self, rng: rng_t, parent1: Pipeline, parent2: Pipeline, hub: K2_Hub) -> Pipeline:
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
        # quick checks: make sure both parents have at least one branch that are active
        assert len(hub.remove_inactive_branches(parent1.get_trait_feature_names())) > 0, "Parent 1 pipeline must have at least one active branch to perform crossover."
        assert len(hub.remove_inactive_branches(parent2.get_trait_feature_names())) > 0, "Parent 2 pipeline must have at least one active branch to perform crossover."

        # get random number between self.branch_min and self.branch_max for number of branches in offspring
        num_branches = rng.integers(self.branch_min, self.branch_max, endpoint=True)

        # get active branches from both parents
        parent1_branches = cp.deepcopy(hub.remove_inactive_branches(parent1.get_trait_feature_names()))
        parent2_branches = cp.deepcopy(hub.remove_inactive_branches(parent2.get_trait_feature_names()))
        # combine branches
        combined_branches = parent1_branches.union(parent2_branches)
        assert len(combined_branches) > 0, "Combined branches from both parents must have at least one branch to perform crossover."

        # if combined branches less than num_branches, just use all combined branches
        if len(combined_branches) <= num_branches:
            return Pipeline(branch_set=combined_branches,
                            ld_node=cp.deepcopy(parent1.ld_node if rng.random() < 0.5 else parent2.ld_node),
                            selector_node=cp.deepcopy(parent1.selector_node if rng.random() < 0.5 else parent2.selector_node))

        # convert to list once (cached for both random and smart crossover)
        combined_list = list(combined_branches)

        # Note: rng.choice returns a 2D numpy array when selecting from tuples, so we need to convert each row back to tuple
        selected_branches = rng.choice(combined_list, size=num_branches, replace=False)
        selected_branches_tuples = set(tuple(branch) if not isinstance(branch, tuple) else branch for branch in selected_branches)

        return Pipeline(branch_set=selected_branches_tuples,
                            ld_node=cp.deepcopy(parent1.ld_node if rng.random() < 0.5 else parent2.ld_node),
                            selector_node=cp.deepcopy(parent1.selector_node if rng.random() < 0.5 else parent2.selector_node))

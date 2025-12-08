#####################################################################################################
#
# Reproduction class that generating new pipelines for K1 regression.
# We use a combination of mutation and crossover to generate new pipelines (both user specified).
# We also initialize the first population of solutions (this is reproduction w/out parents).
#
# For coding purposes, switch Hub to K1_Hub once all functionality is done.
# This will ensure that all base functionality is correct and help fill K1 specific functionality.
# Switching back to Hub will remove incorrect dependencies on K1 specific code in the Base folder.
#
#####################################################################################################

# imports from Base
from ..Base.pipeline import Pipeline
from ..Base.types import (rng_t, prob_t, int32_t, uint16_t, snp_t)
from ..Base.reproduction import Reproduction
from ..Base.selectors import (VarianceThresholdNode, SelectPercentileNode, SelectFweNode, SelectFromModelLasso,
                              SelectFromModelTree, FeatureEncodingFrequencySelector)

# imports from K1
from .ld_selector import LDSelector
from .snp_hub import K1_Hub

# additional imports
from typeguard import typechecked
from typing import Set
import numpy as np
import copy as cp

@typechecked
class K1_Reproduction(Reproduction):
    def __init__(self,
                 branch_max: uint16_t,
                 branch_min: uint16_t,
                 mut_prob: prob_t = prob_t(.5),
                 cross_prob: prob_t = prob_t(.5),
                 mut_selector_p: prob_t = prob_t(.5),
                 mut_ld_p: prob_t = prob_t(.5),
                 mut_regressor_p: prob_t = prob_t(.5),
                 mut_ran_p: prob_t = prob_t(.45),
                 mut_smt_p: prob_t = prob_t(.45),
                 m_in_win_p: prob_t = prob_t(.1),
                 m_out_win_p: prob_t = prob_t(.45),
                 m_out_chr_p: prob_t = prob_t(.45),
                 window_distance: int32_t = int32_t(1000000)) -> None:
        """
        K1 Reproduction class that extends the Base Reproduction class.
        """

        # pass all variables to the Base Reproduction class
        super().__init__(branch_max=branch_max,
                         branch_min=branch_min,
                         mut_prob=mut_prob,
                         cross_prob=cross_prob,
                         mut_selector_p=mut_selector_p,
                         mut_ld_p=mut_ld_p,
                         mut_regressor_p=mut_regressor_p,
                         mut_ran_p=mut_ran_p,
                         mut_smt_p=mut_smt_p,
                         m_in_win_p=m_in_win_p,
                         m_out_win_p=m_out_win_p,
                         m_out_chr_p=m_out_chr_p,
                         window_distance=window_distance)
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

    def mutate(self, rng: rng_t, parent: Pipeline, hub: K1_Hub) -> Pipeline:
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
        anchor_snps = rng.choice(list(parent_branches), size=num_to_add, replace=True)
        for anchor in anchor_snps:
            parent_branches.add(self.mutate_branch(rng, anchor, hub))

        # generate offspring with mutated parent branches and pass through selector and ld nodes from parent
        offspring = Pipeline(branch_set=parent_branches, ld_node=cp.deepcopy(parent.ld_node), selector_node=cp.deepcopy(parent.selector_node))

        # roll to mutate the ld node
        if rng.choice([True, False], p=[self.mut_ld_p, 1.0 - self.mut_ld_p]):
            offspring.mutate_ld_node(rng)
        # roll to mutate the selector node
        if rng.choice([True, False], p=[self.mut_selector_p, 1.0 - self.mut_selector_p]):
            offspring.mutate_selector_node(rng)

        return offspring

    def mutate_post_crossover(self, rng: rng_t, offspring: Pipeline, hub: K1_Hub) -> Pipeline:
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
        anchor_snps = rng.choice(list(offspring_new_branches), size=num_to_add, replace=True)
        for anchor in anchor_snps:
            offspring_new_branches.add(self.mutate_branch(rng, anchor, hub))
        assert len(offspring.get_branch_set()) <= len(offspring_new_branches) <= len(offspring.get_branch_set()) + num_to_add \
            ,"Offspring branches after mutation must be correct size."

        # generate new offspring with mutated branches and pass through selector and ld nodes from original offspring
        offspring = Pipeline(branch_set=offspring_new_branches, ld_node=cp.deepcopy(offspring.ld_node), selector_node=cp.deepcopy(offspring.selector_node))

        # roll to mutate the ld node
        if rng.choice([True, False], p=[self.mut_ld_p, 1.0 - self.mut_ld_p]):
            offspring.mutate_ld_node(rng)
        # roll to mutate the selector node
        if rng.choice([True, False], p=[self.mut_selector_p, 1.0 - self.mut_selector_p]):
            offspring.mutate_selector_node(rng)

        return offspring

    def mutate_branch(self, rng: rng_t, branch: snp_t, hub: K1_Hub) -> snp_t:
        """
        Function to mutate a given anchor branch SNP.

        Parameters:
            rng (rng_t): A numpy random number generator from the evolver
            branch (snp_t): The branch SNP to mutate
            hub: An interface to a branch hub to get branch specific information

        Returns:
            snp_t: The mutated branch SNP
        """

        # quick checks
        assert hub.get_active_flag(branch), "Anchor branch must be active in the hub to mutate"
        assert '.' in branch, "Anchor branch SNP must be in the format 'chrom.pos'"

        # randomly choose a anchor mutation type
        mutation_type = rng.choice(['in_window', 'out_window', 'out_chrom'], p = [self.m_in_win_p / (self.m_in_win_p + self.m_out_win_p + self.m_out_chr_p),
                                                                                self.m_out_win_p / (self.m_in_win_p + self.m_out_win_p + self.m_out_chr_p),
                                                                                self.m_out_chr_p / (self.m_in_win_p + self.m_out_win_p + self.m_out_chr_p)])
        # smart or random mutation roll
        ran_roll = rng.choice([True, False], p=[self.mut_ran_p / (self.mut_ran_p + self.mut_smt_p), self.mut_smt_p / (self.mut_ran_p + self.mut_smt_p)])

        # perform mutation based on type
        if mutation_type == 'in_window':
            if ran_roll:
                return hub.get_ran_snp_in_window(branch, rng, hub.get_in_window_positions(branch))
            else:
                return hub.get_smt_snp_in_window(branch, rng, hub.get_in_window_positions(branch))
        elif mutation_type == 'out_window':
            if ran_roll:
                return hub.get_ran_snp_in_chrm(branch, rng, hub.get_out_of_window_positions(branch))
            else:
                return hub.get_smt_snp_in_chrm(branch, rng, hub.get_out_of_window_positions(branch))
        else: # out_chrom
            if ran_roll:
                return hub.get_ran_snp_out_chrm(branch, rng)
            else:
                return hub.get_smt_snp_out_chrm(branch, rng)

    def crossover(self, rng: rng_t, parent1: Pipeline, parent2: Pipeline, hub: K1_Hub) -> Pipeline:
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
                            ld_node=cp.deepcopy(parent1.ld_node if rng.choice([True, False]) else parent2.ld_node),
                            selector_node=cp.deepcopy(parent1.selector_node if rng.choice([True, False]) else parent2.selector_node))

        # smart or random crossover roll
        ran_roll = rng.choice([True, False], p=[self.mut_ran_p / (self.mut_ran_p + self.mut_smt_p),
                                                       self.mut_smt_p / (self.mut_ran_p + self.mut_smt_p)])

        if ran_roll: # random crossover
            return Pipeline(branch_set=set(rng.choice(list(combined_branches), size=num_branches, replace=False)),
                            ld_node=cp.deepcopy(parent1.ld_node if rng.choice([True, False]) else parent2.ld_node),
                            selector_node=cp.deepcopy(parent1.selector_node if rng.choice([True, False]) else parent2.selector_node))
        else: # smart crossover
            r2 = np.array([hub.get_r2(branch) for branch in combined_branches])
            return Pipeline(branch_set=set(rng.choice(list(combined_branches), size=num_branches, replace=False, p=r2 / np.sum(r2))),
                            ld_node=cp.deepcopy(parent1.ld_node if rng.choice([True, False]) else parent2.ld_node),
                            selector_node=cp.deepcopy(parent1.selector_node if rng.choice([True, False]) else parent2.selector_node))
#####################################################################################################
#
# Reproduction class that generating new pipelines.
# We use a combination of mutation and crossover to generate new pipelines (both user specified).
# We also initialize the first population of solutions (this is reproduction w/out parents).
#
#####################################################################################################

import numpy as np
from typeguard import typechecked
from typing import List, Tuple, Set

from .pipeline import Pipeline
from .snp_hub import SnpHub

import copy as cp

# feature selectors
from .scikit_node import VarianceThresholdNode, SelectPercentileNode, SelectFweNode, SelectFromModelLasso, SelectFromModelTree, SequentialFeatureSelectorNode, LDSelector, FeatureEncodingFrequencySelector
# regressors
from .scikit_node import LinearRegressionNode, RandomForestRegressorNode, SGDRegressorNode, DecisionTreeRegressorNode, ElasticNetNode, SVRNode, GradientBoostingRegressorNode

rng_t = np.random.Generator
pop_size_t = np.uint16
# probability type
prob_t = np.float64
# snp type
snp_t = np.str_
#YF snps type
snps_t = Set

@typechecked
class Reproduction:
    def __init__(self,
                 uni_cnt_max: pop_size_t,
                 uni_cnt_min: pop_size_t,
                 mut_prob: prob_t = prob_t(.5),
                 cross_prob: prob_t = prob_t(.5),
                 mut_selector_p: prob_t = prob_t(.5),
                 mut_ld_p: prob_t = prob_t(.5),
                 mut_regressor_p: prob_t = prob_t(.5),
                 mut_ran_p: prob_t = prob_t(.45),
                 mut_smt_p: prob_t = prob_t(.45),
                 smt_in_in_p: prob_t = prob_t(.1),
                 smt_in_out_p: prob_t = prob_t(.45),
                 smt_out_out_p: prob_t = prob_t(.45)) -> None:

        # save all the variables
        self.uni_cnt_max = uni_cnt_max
        self.uni_cnt_min = uni_cnt_min
        self.mut_prob = mut_prob
        self.cross_prob = cross_prob
        self.mut_selector_p = mut_selector_p
        self.mut_ld_p = mut_ld_p
        self.mut_regressor_p = mut_regressor_p
        self.mut_ran_p = mut_ran_p
        self.mut_smt_p = mut_smt_p
        self.smt_in_in_p = smt_in_in_p
        self.smt_in_out_p = smt_in_out_p
        self.smt_out_out_p = smt_out_out_p

        return

    # method to generate the initial population
    def generate_random_pipeline(self, rng_: rng_t, snps: snps_t, seed: int,) -> Pipeline:
        # quick checks
        assert len(snps) > 0

        # set rng
        rng = np.random.default_rng(rng_)

        # randomly select selector node
        selector_node = rng.choice([VarianceThresholdNode(rng_=rng),
                                    SelectPercentileNode(rng_=rng),
                                    SelectFweNode(rng_=rng),
                                    SelectFromModelLasso(rng_=rng, seed=seed),
                                    SelectFromModelTree(rng_=rng, seed=seed),
                                    SequentialFeatureSelectorNode(rng_=rng, seed=seed),
                                    FeatureEncodingFrequencySelector(rng_=rng),
                                ])
        # randomly select root node
        root_node = rng.choice([LinearRegressionNode(rng_=rng),
                                RandomForestRegressorNode(rng_=rng, seed=seed),
                                SGDRegressorNode(rng_=rng, seed=seed),
                                DecisionTreeRegressorNode(rng_=rng, seed=seed),
                                ElasticNetNode(rng_=rng, seed=seed),
                                SVRNode(rng_=rng),
                                GradientBoostingRegressorNode(rng_=rng, seed=seed),
                            ])

        return Pipeline(ld_node=LDSelector(rng_=rng, seed=seed),
                        selector_node=selector_node,
                        root_node=root_node,
                        uni_snps=snps,
                        traits=[])

    def variation_order(self, rng_: rng_t, offpring_cnt: pop_size_t) -> Tuple[List[str], pop_size_t]:
        """
        Function to generate the order of variation operators to be applied to generate offspring.
        The order is determined by the probabilities of mutation and crossover.
        We return a list with the names of the operators in the order they should be applied.
        E.g.: ['m', 'c', 'm', 'c', ...]

        Crossover means two parents are required
        Mutation means one parent is required

        Parameters:
        rng (rng_t): A numpy random number generator from the evolver
        offpring_cnt (pop_size_t): The number of offspring to generate

        Returns:
        List[np.str_]: A list of strings representing the order of variation operators to be applied
        pop_size_t: The number of parents needed to generate the offspring
        """
        # set the random number generator
        rng = np.random.default_rng(rng_)

        # parents needed by variantion operators
        parent_count = {'m': 1, 'c': 2}

        # generate the order of variation operations
        order = rng.choice(['m', 'c'], offpring_cnt, p=[self.mut_prob, self.cross_prob]).tolist()

        # make sure we have the right number of offspring
        assert len(order) == offpring_cnt

        # return the order and number of parents needed
        return order, pop_size_t(sum(parent_count[op] for op in order))

    # method to generate offspring
    def produce_offspring(self,
                          rng_: rng_t,
                          hub: SnpHub,
                          offspring_cnt: pop_size_t,
                          population: List[Pipeline],
                          parent_ids: List[pop_size_t],
                          order: List[str]) -> List[Pipeline]:
        # quick checks
        assert len(parent_ids) > 0
        assert len(population) > 0
        assert offspring_cnt > 0

        # set the random number generator
        rng = np.random.default_rng(rng_)

        # list to store the offspring
        offspring = []

        # go through the order of operators
        p_id = 0
        for op in order:
            # mutation only
            if op == 'm':
                offspring.append(self.mutate(rng, population[parent_ids[p_id]], hub))
                p_id += 1
            # crossover only
            elif op == 'c':
                offspring.append(self.crossover(rng, population[parent_ids[p_id]], population[parent_ids[p_id+1]], hub))
                p_id += 2
            else:
                raise ValueError(f"Unknown operator: {op}")
        # make sure we have the right number of offspring
        assert len(offspring) == offspring_cnt
        assert p_id == len(parent_ids)

        # return the offspring
        return offspring

    # what mutation are we applying to the pipeline
    def mutate(self,
               rng_: rng_t,
               parent: Pipeline,
               hub: SnpHub) -> Pipeline:

        # set the random number generator
        rng = np.random.default_rng(rng_)

        # clone the pipeline
        offspring = Pipeline(uni_snps=set(),
                            selector_node=parent.get_selector_node(),
                            ld_node=parent.get_ld_node(),
                            root_node=parent.get_root_node(),
                            traits=[])

        # get parent uni snps
        parent_uni_snps = cp.deepcopy(parent.get_uni_snps())

        # delete snps that have been pruned before
        # possible to have less than the minimum
        parent_uni_snps = self.delete_pruned_snps(parent_uni_snps, hub)

        # get the number of snps to add
        snps_to_add = self.num_snps_to_add(rng, parent_uni_snps)

        # sample a set of snps from parent_uni_snps with replacement to append to the offspring
        mutated_snps = rng.choice(list(parent_uni_snps), snps_to_add, replace=True)

        # go through mutated_snps, mutate them, and append them to parent_uni_snps
        for uni_snp in mutated_snps:
            parent_uni_snps.add(self.get_ran_snp_mut(rng, uni_snp, hub))

        # update the epi pairs + new interactions
        offspring.set_uni_snps(parent_uni_snps)

        # mutate the selector node
        if rng.choice([True, False], p=[self.mut_selector_p, 1.0-self.mut_selector_p]):
            offspring.mutate_selector_node(rng)

        # mutate the ld node
        if rng.choice([True, False], p=[self.mut_ld_p, 1.0-self.mut_ld_p]):
            offspring.mutate_ld_node(rng)

        # mutate the regressor node
        if rng.choice([True, False], p=[self.mut_regressor_p, 1.0-self.mut_regressor_p]):
            offspring.mutate_root_node(rng)

        return offspring

    # delete snps that have been pruned before
    def delete_pruned_snps(self,
                        uni_snps: snps_t,
                        hub: SnpHub) -> snps_t:

        # get the set of snps that have not been pruned
        not_pruned_snps = set()

        for snp in uni_snps:
            if hub.has_been_prunned(snp) == False:
                not_pruned_snps.add(snp)

        return not_pruned_snps

    # add a set of snps: smart or random addition depends on the probabilities
    def num_snps_to_add(self,
                    rng: rng_t,
                    uni_snps: snps_t) -> np.uint16:
        # quick checks
        assert len(uni_snps) <= self.uni_cnt_max
        assert self.uni_cnt_max - len(uni_snps) >= 0

        # get a number of interactions to add based on self.epi_cnt_max and self.epi_cnt_min
        if len(uni_snps) < self.uni_cnt_min:
            num_add_range = np.uint16(self.uni_cnt_min - len(uni_snps))
        else:
            num_add_range = np.uint16(max(self.uni_cnt_max - len(uni_snps), 0))

        # if 0 or 1 just return the number
        if num_add_range == 0 or num_add_range == 1:
            return np.uint16(num_add_range)
        # else pick a number between the range and 1 (range < self.num_add_interactions)
        else:
            return np.uint16(rng.integers(1, num_add_range, endpoint=True))

    def get_ran_snp_mut(self, rng_: rng_t, snp_name: snp_t, hub: SnpHub) -> snp_t:
        # set the random number generator
        rng = np.random.default_rng(rng_)

        # randomly select one of the mutations to perform
        mut_fun = rng.choice([0,1,2], p=[self.smt_in_in_p / (self.smt_in_in_p + self.smt_in_out_p + self.smt_out_out_p),
                                         self.smt_in_out_p / (self.smt_in_in_p + self.smt_in_out_p + self.smt_out_out_p),
                                         self.smt_out_out_p / (self.smt_in_in_p + self.smt_in_out_p + self.smt_out_out_p)])

        roll = rng.choice([True,False], p=[self.mut_smt_p / (self.mut_smt_p + self.mut_ran_p),
                                           self.mut_ran_p / (self.mut_smt_p + self.mut_ran_p)])

        # in chromosome and in bin
        if mut_fun == 0:
            if roll:
                return hub.get_smt_snp_in_bin(snp=snp_name, rng_=rng)
            else:
                return hub.get_ran_snp_in_bin(snp=snp_name, rng_=rng)
        # in chromosome and out of bin
        elif mut_fun == 1:
            if roll:
                return hub.get_smt_snp_in_chrm(snp=snp_name, rng_=rng)
            else:
                return hub.get_ran_snp_in_chrm(snp=snp_name, rng_=rng)
        # out of chromosome
        elif mut_fun == 2:
            if roll:
                return hub.get_smt_snp_out_chrm(snp=snp_name, rng_=rng)
            else:
                return hub.get_ran_snp_out_chrm(snp=snp_name, rng_=rng)
        else:
            exit("Unknown mutation function", -1)

    # execute a crossover between two pipelines
    def crossover(self,
                  rng: np.random.Generator,
                  parent1: Pipeline,
                  parent2: Pipeline,
                  hub: SnpHub) -> Pipeline:

        # combine the univariate snps from both parents
        p1_snps = cp.deepcopy(parent1.get_uni_snps())
        p2_snps = cp.deepcopy(parent2.get_uni_snps())
        combined_snps = p1_snps.union(p2_snps)
        snps = None

        # if the combined snps is greater than the maximum allowed, we smaple the maximum allowed
        if len(combined_snps) > self.uni_cnt_max:
            # roll to see if we should randomly sample or use r2 to sample
            if rng.choice([True, False], p=[self.mut_smt_p / (self.mut_smt_p + self.mut_ran_p), self.mut_ran_p / (self.mut_smt_p + self.mut_ran_p)]):
                # collect all snps r2 to sample based of that
                r2 = np.array([hub.get_uni_res(snp) for snp in combined_snps], dtype=np.float32)
                r2 = r2 / np.sum(r2, dtype=np.float32)

                # sample the snps
                snps = rng.choice(list(combined_snps), self.uni_cnt_max, replace=False, p=r2)
            else:
                snps = rng.choice(list(combined_snps), self.uni_cnt_max, replace=False)
        else:
            snps = combined_snps

        return Pipeline(uni_snps=set(snps),
                        selector_node=cp.deepcopy(parent1.get_selector_node()) if rng.choice([True, False]) else cp.deepcopy(parent2.get_selector_node()),
                        ld_node=cp.deepcopy(parent1.get_ld_node()) if rng.choice([True, False]) else cp.deepcopy(parent2.get_ld_node()),
                        root_node=cp.deepcopy(parent1.get_root_node()) if rng.choice([True, False]) else cp.deepcopy(parent2.get_root_node()),
                        traits=[])
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
# wiggle range (step) type
step_t = np.uint16

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
                 smt_out_out_p: prob_t = prob_t(.45),

                 wiggle_mut_p: prob_t = prob_t(.45), #YF
                 wiggle_rand_p: prob_t= prob_t(.5),
                 wiggle_smrt_p: prob_t= prob_t(.5),
                 step: step_t = step_t(5),

                 num_add_interactions: pop_size_t = pop_size_t(10),
                 num_del_interactions: pop_size_t = pop_size_t(10),
                 num_add_snps: pop_size_t = pop_size_t(10), #YF
                 num_del_snps: pop_size_t = pop_size_t(10)) -> None:

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
        self.wiggle_mut_p = wiggle_mut_p
        self.wiggle_rand_p = wiggle_rand_p
        self.wiggle_smrt_p = wiggle_smrt_p
        self.step = step
        self.num_add_interactions = num_add_interactions
        self.num_del_interactions = num_del_interactions
        self.num_add_snps = num_add_snps
        self.num_del_snps = num_del_snps

        return

    # method to generate the initial population
    def generate_random_pipeline(self, rng_: rng_t, snps: snps_t, seed: int) -> Pipeline:
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
        # create the pipeline
        return Pipeline(ld_node=LDSelector(rng_=rng),
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

        # generate half of the operators
        # we do this so that in the worse case (all crossover) we don't need to generate more than needed
        order = rng.choice(['m', 'c'], offpring_cnt // 2, p=[self.mut_prob, self.cross_prob]).tolist()

        # how many more offspring do we need
        left = offpring_cnt -  pop_size_t(sum(parent_count[op] for op in order))

        # get that many more operators
        while left > 0:
            op = rng.choice(['m', 'c'], 1, p=[self.mut_prob, self.cross_prob])[0]

            # check if we can add the operator
            if parent_count[op] <= left:
                order.append(op)
                left -= parent_count[op]

            # if left == 1, we can only add mutation
            elif left == 1:
                order.append('m')
                left -= 1

        # make sure we have the right number of offspring
        assert sum(parent_count[op] for op in order) == offpring_cnt

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
            # crossover + mutation
            elif op == 'c':
                off1, off2 = self.crossover(rng, population[parent_ids[p_id]], population[parent_ids[p_id+1]])
                offspring.append(self.mutate(rng, off1, hub))
                offspring.append(self.mutate(rng, off2, hub))
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

        # delete snps if we have more than the minimum
        # possible to have less than the minimum
        if len(parent_uni_snps) > self.uni_cnt_min:
            # delete uni snps
            parent_uni_snps = self.delete_uni_snps(rng, parent_uni_snps, hub)

        if len(parent_uni_snps) < self.uni_cnt_max:
            # get new set of uni snps
            new_snps_list = self.add_uni_snps(rng, hub, parent_uni_snps)

        # mutate the offspring
        uni_snps = set()

        # go through the univariate snp and mutate them
        for uni_snp in parent_uni_snps:
            # mutate via wiggle or random replacement
            if rng.choice([True, False], p=[self.wiggle_mut_p, 1.0-self.wiggle_mut_p]):
                # smart wiggle
                if rng.choice([True, False], p=[self.mut_smt_p / (self.mut_smt_p + self.mut_ran_p), self.mut_ran_p / (self.mut_smt_p + self.mut_ran_p)]):
                    uni_snps.add(self.mutate_uni_node_wiggle_smrt(rng,hub,uni_snp))
                else: # dumb wiggle
                    uni_snps.add(self.mutate_uni_node_wiggle_rand(rng,hub,uni_snp))
            else:
                # replace snp with a random one
                uni_snps.add(self.get_ran_snp_mut(rng, uni_snp, hub))

        # update the epi pairs + new interactions
        offspring.set_uni_snps(uni_snps.union(new_snps_list))

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

    # delete a set of snps: smart or random deletion depends on the probabilities
    def delete_uni_snps(self,
                        rng: rng_t,
                        uni_snps: snps_t,
                        hub: SnpHub) -> snps_t:
        # quick checks
        assert len(uni_snps) - self.uni_cnt_min > 0

        num_snps = len(uni_snps)

        # get a number of snps to delete based on self.uni_cnt_min
        num_del_range = np.uint16(max(len(uni_snps) - self.uni_cnt_min, 0))

        # if nothing to do return snps
        if num_del_range == 0 or num_snps == 0:
            return uni_snps
        # if range is 1, delete one snp
        elif num_del_range == 1:
            num_deletions = 1
        # if the range is greater than self.num_del_interactions
        elif num_del_range >= self.num_del_snps:
            # get a random number between 1 and num_del_range
            num_deletions = rng.integers(1, self.num_del_snps)
        # else pick a number between the range and 1 (range < self.num_del_interactions)
        else:
            num_deletions = rng.integers(1, num_del_range)


        # delete random snps
        if rng.choice([True, False], p=[self.mut_ran_p / (self.mut_smt_p + self.mut_ran_p), self.mut_smt_p / (self.mut_smt_p + self.mut_ran_p)]):
            # get a random set of snps to delete
            del_snps = rng.choice(np.array(list(uni_snps)), num_deletions, replace=False)
            return uni_snps.difference(del_snps)
        # delete snps based on r2
        else:
            # collect all the results
            r2_results = []
            for snp in uni_snps:
                assert hub.get_uni_res(snp) >= 0.0
                assert hub.does_snp_exist(snp) == True
                r2_results.append(np.float32(1.0) - hub.get_uni_res(snp))

            # normalize the results with respect to the sum of r2 results
            r2_results = np.array(r2_results, dtype=np.float32) / np.sum(r2_results, dtype=np.float32)

            # get a set of interactions to delete
            del_snps = rng.choice(np.array(list(uni_snps)), num_deletions, p=r2_results, replace=False)

            # convert to sets
            del_snps = set(del_snps)

            # return the interactions without the deleted ones
            return uni_snps.difference(del_snps)

    # add a set of snps: smart or random addition depends on the probabilities
    def add_uni_snps(self,
                    rng: rng_t,
                    hub: SnpHub,
                    uni_snps: snps_t) -> snps_t:
        # quick checks
        assert len(uni_snps) <= self.uni_cnt_max
        assert self.uni_cnt_max - len(uni_snps) >= 0

        # get a number of interactions to add based on self.epi_cnt_max
        num_add_range = np.uint16(max(self.uni_cnt_max - len(uni_snps), 0))

        # if nothing to do return interactions
        if num_add_range == 0:
            return uni_snps
        # if range is 1, add one interaction
        elif num_add_range == 1:
            num_additions = 1
        # if the range is greater than self.num_add_interactions
        elif num_add_range >= self.num_add_snps:
            # get a random number between 1 and num_add_interactions
            num_additions = rng.integers(1, self.num_add_snps)
        # else pick a number between the range and 1 (range < self.num_add_interactions)
        else:
            num_additions = rng.integers(1, num_add_range)

        # collect all new snps
        new_snps = set()
        while len(new_snps) < num_additions:
            # roll to get the snp
            new_snp_name = None

            # get the snp randomly
            if rng.choice([True, False], p=[self.mut_ran_p / (self.mut_smt_p + self.mut_ran_p), self.mut_smt_p / (self.mut_smt_p + self.mut_ran_p)]):
                new_snp_name = hub.get_ran_snp(rng)
            else:
                # get new snp based on R2
                new_snp_name = hub.get_smt_snp(rng)

            assert new_snp_name != None
           # make sure the new snps are not already in the snps set
            if new_snp_name in new_snps:
                continue
            else:
                new_snps.add(new_snp_name)
        # return the interactions
        return new_snps

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

    def mutate_uni_node_wiggle_rand(self,
                               rng: rng_t,
                               hub: SnpHub,
                               uni_snp: snp_t) -> snp_t:
        # will hold new snp
        new_snp_name = None

        # randomly select snp within wiggle range from current snp
        new_snp_name = hub.get_ran_snp_in_bin_wiggle(uni_snp,rng,self.step)

        # make sure the new snps are set and are in the same chromosome and bin
        assert new_snp_name != None
        assert new_snp_name != uni_snp
        return new_snp_name

    def mutate_uni_node_wiggle_smrt(self,
                               rng: rng_t,
                               hub: SnpHub,
                               uni_snp: snp_t) -> snp_t:
        # will hold new snp
        new_snp_name = None

        # smartly select snp based on R2 as probability within wiggle range from current snp
        new_snp_name = hub.get_smt_snp_in_bin_wiggle(uni_snp,rng,self.step)

         # make sure the new snps are set and are in the same chromosome and bin
        assert new_snp_name != None
        assert new_snp_name != uni_snp
        return new_snp_name

    # execute a crossover between two pipelines
    def crossover(self,
                  rng: np.random.Generator,
                  parent1: Pipeline,
                  parent2: Pipeline) -> Tuple[Pipeline, Pipeline]:

        p1_uni_snps = list(parent1.get_uni_snps())
        p2_uni_snps = list(parent2.get_uni_snps())

        # get smallest half length from both
        half_len_uni = min(len(p1_uni_snps), len(p2_uni_snps)) // 2

        # randomly select indecies from both parents epi branches
        p1_idx_uni = rng.choice(len(p1_uni_snps), half_len_uni, replace=False)
        p2_idx_uni = rng.choice(len(p2_uni_snps), half_len_uni, replace=False)

        for i1, i2 in zip(p1_idx_uni, p2_idx_uni):
            p1_uni_snps[i1], p2_uni_snps[i2] = p2_uni_snps[i2], p1_uni_snps[i1]

        # make sure the univariate snps set is the correct size
        assert 0 <= len(p1_uni_snps) <= self.uni_cnt_max
        assert 0 <= len(p2_uni_snps) <= self.uni_cnt_max

        # create one offspring per parent
        offspring_1 = Pipeline(uni_snps=set(p1_uni_snps),
                               selector_node=parent1.get_selector_node(),
                               ld_node=parent1.get_ld_node(),
                               root_node=parent1.get_root_node(),
                               traits=[])

        offsprint_2 = Pipeline(uni_snps=set(p2_uni_snps),
                               selector_node=parent2.get_selector_node(),
                               ld_node=parent2.get_ld_node(),
                               root_node=parent2.get_root_node(),
                               traits=[])

        return offspring_1, offsprint_2
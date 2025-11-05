#####################################################################################################
#
# Abstract class for a pipeline that holds all relevant information within a pipeline instance:
# - Set containing the SNPs (single SNPs or Tuples of SNPs (k2,k3,...)) that the pipeline will use.
# - List of traits that a pipeline is given after evaluation.
# - the LD node that a pipeline  will use.
# - the selector node that a pipeline  will use.
# - the methods to get and mutate the LD node.
# - the methods to get and mutate the selector node.
#
# This class is used in the evolver to hold individual pipelines that will be evolved.
# Note that derived evolver classes MUST add checks/guards to ensure that the correct things are
# being passed to the pipeline class.
#
#####################################################################################################


from typing import List, Set
from typeguard import typechecked
import copy as cp
from .types import (rng_t, int16_t, float32_t)
from .selectors import SelectorNode

@typechecked
class Pipeline:
    def __init__(self,
                 branch_set: Set,
                 ld_node: SelectorNode,
                 selector_node: SelectorNode) -> None:
        """
        - branch set:
            Set containing the SNPs (single SNPs or Tuples of SNPs (k2,k3,...)) that the pipeline will use.
        - ld_node:
            the LD node that a pipeline  will use.
        - selector_node:
            the selector node that a pipeline  will use.
        """
        # make sure branch_set is not empty
        assert len(branch_set) > 0

        # holds pipeline's set of branch nodes
        self.branch_set = cp.deepcopy(branch_set)
        # holds pipeline's set of traits: r2 (traits[0]) and feature_cnt (traits[1]) and actual feature names (traits[2])
        self.traits = []
        # holds the selector node
        self.selector_node = cp.deepcopy(selector_node)
        # holds the LD node
        self.ld_node = cp.deepcopy(ld_node)

    # set traits
    def set_traits(self, traits: List) -> None:
        """
        Set the traits for the pipeline.
        Args:
            traits (List): List containing the traits to set for the pipeline.
                traits[0]: float32_t - r2 value
                traits[1]: int16_t - feature count
                traits[2]: Set - feature names
        """

        # check that internal traits is empty
        assert len(self.traits) == 0
        # make sure that the traits are not empty
        assert len(traits) == 3
        # make sure correct types
        assert isinstance(traits[0], float32_t)
        assert isinstance(traits[1], int16_t)
        assert isinstance(traits[2], Set)
        # make sure they are the correct length
        assert len(traits[2]) == traits[1]

        # make sure we have a non-negative number of features
        assert traits[1] >= 0

        # update the traits
        self.traits = [cp.deepcopy(traits[0]),
                        cp.deepcopy(traits[1]),
                        cp.deepcopy(traits[2])]
        return

    def get_trait_r2(self) -> float32_t:
        assert len(self.traits) == 3
        return self.traits[0]

    def get_trait_feature_cnt(self) -> int16_t:
        assert len(self.traits) == 3
        assert self.traits[1] >= 0 # make sure we have a non-negative number of features
        assert self.traits[1] == len(self.traits[2]) # make sure feature count matches feature names length
        return self.traits[1]

    def get_trait_feature_names(self) -> Set:
        assert len(self.traits) == 3
        assert len(self.traits[2]) == self.traits[1]
        assert len(self.traits[2]) > 0 # make sure we have at least one feature name
        return self.traits[2]

    def get_branch_set(self) -> Set:
        assert len(self.branch_set) > 0
        return self.branch_set

    def get_ld_node(self):
        return self.ld_node

    def get_selector_node(self):
        return self.selector_node

    def mutate_ld_node(self, rng: rng_t) -> None:
        self.ld_node.mutate(rng)

    def mutate_selector_node(self, rng: rng_t) -> None:
        self.selector_node.mutate(rng)
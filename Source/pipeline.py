# Will contain the definition of Pipeline class which will be used by univariate branch.

from typing import List, Set
from .scikit_node import ScikitNode, LDSelector
import numpy as np
from typeguard import typechecked
import numpy.typing as npt
import copy as cp

# numpy random number generator type
gen_rng_t = np.random.Generator
gen_snp_arr_t = npt.NDArray[np.str_]
uni_snps_t = Set
uniq_chrombin_t = Set
traits_t = List
r2_t = np.float32
feature_cnt_t = np.int16
feature_name_t = Set
div_score_t = np.float32


@typechecked
class Pipeline:
    # initialize the pipeline with a set of univariate snps, an LD node, a selector node and a root node
    def __init__(self,
                 uni_snps: uni_snps_t,
                 ld_node: LDSelector,
                 selector_node: ScikitNode,
                 root_node: ScikitNode) -> None:

        # make sure uni_snps is not empty
        assert len(uni_snps) > 0

        # holds pipeline's set of univariate snps
        self.univariate_snps = cp.deepcopy(uni_snps)
        # holds pipeline's set of traits: r2 (traits[0]) and feature_cnt (traits[1]) and actual feature names (traits[2])
        self.traits = []
        # holds the selector node
        self.selector_node = cp.deepcopy(selector_node)
        # holds the LD node
        self.ld_node = cp.deepcopy(ld_node)
        # holds the root node
        self.root_node = cp.deepcopy(root_node)

    # set traits
    def set_traits(self, traits: traits_t) -> None:
        # check that internal traits is empty
        assert len(self.traits) == 0
        # make sure that the traits are not empty
        assert len(traits) == 3
        # make sure correct types
        assert isinstance(traits[0], r2_t)
        assert isinstance(traits[1], feature_cnt_t)
        assert isinstance(traits[2], feature_name_t)
        # make sure they are the correct length
        assert len(traits[2]) == traits[1]

        # make sure we have a non-negative number of features
        assert traits[1] >= 0

        # update the traits
        self.traits = [cp.deepcopy(traits[0]),
                        cp.deepcopy(traits[1]),
                        cp.deepcopy(traits[2])]
        return

    # get r2 from trait set
    def get_trait_r2(self) -> r2_t:
        assert len(self.traits) == 3
        return self.traits[0]

    # get feature count from trait set
    def get_trait_feature_cnt(self) -> feature_cnt_t:
        assert len(self.traits) == 3
        assert self.traits[1] >= 0 # make sure we have a non-negative number of features
        return self.traits[1]

    # get feature names that made it to regressor from trait set
    def get_trait_feature_names(self) -> feature_name_t:
        assert len(self.traits) == 3
        assert len(self.traits[2]) == self.traits[1]
        assert len(self.traits[2]) > 0 # make sure we have at least one feature name
        return self.traits[2]

    # get the traits from the pipeline
    def get_traits(self) -> traits_t:
        assert len(self.traits) == 3
        return self.traits

    # get the univariate snps from the pipeline
    def get_uni_snps(self) -> uni_snps_t:
        assert len(self.univariate_snps) > 0
        return self.univariate_snps

    # method to get the LD node
    def get_ld_node(self):
        return self.ld_node

    # method to get the selector node
    def get_selector_node(self):
        return self.selector_node

    # method to get the root node
    def get_root_node(self):
        return self.root_node

    # print the pipeline
    def print_pipeline(self) -> None:
        print("Pipeline:")
        print('Traits:', self.traits)
        print("SNP set:")
        print('len(self.univariate_snps):', len(self.univariate_snps))
        for uni_node in self.univariate_snps:
            print(uni_node)
        print("LD Node:")
        print(self.ld_node.selector)
        print("Selector Node:")
        print(self.selector_node.selector)
        print("Root Node:")
        print(self.root_node.regressor)
        return

    # call the LD nodes mutation functions to mutate the LD node
    def mutate_ld_node(self, rng_: gen_rng_t) -> None:
        rng = np.random.default_rng(rng_)
        self.ld_node.mutate(rng)

    # call the selector nodes mutation functions to mutate the selector node
    def mutate_selector_node(self, rng_: gen_rng_t) -> None:
        rng = np.random.default_rng(rng_)
        self.selector_node.mutate(rng)

    # call the root nodes mutation functions to mutate the root node
    def mutate_root_node(self, rng_: gen_rng_t) -> None:
        rng = np.random.default_rng(rng_)
        self.root_node.mutate(rng)
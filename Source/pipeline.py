# Will contain the definition of Pipeline class which will be used by both the Epistatic branch and the univariate branch.

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
    def __init__(self,
                 uni_snps: uni_snps_t,
                 ld_node: LDSelector,
                 selector_node: ScikitNode,
                 root_node: ScikitNode,
                 traits: traits_t) -> None:

        # holds pipeline's set of univariate snps
        self.univariate_snps = cp.deepcopy(uni_snps)
        # holds pipeline's set of traits: r2 (traits[0]) and feature_cnt (traits[1]) and actual feature names (traits[2])
        self.traits = cp.deepcopy(traits)
        # holds the selector node
        self.selector_node = cp.deepcopy(selector_node)
        # holds the LD node
        self.ld_node = cp.deepcopy(ld_node)
        # holds the root node
        self.root_node = cp.deepcopy(root_node)

    # set univariate snps
    def set_uni_snps(self, uni_snps: uni_snps_t) -> None:
        # check that internal uni_snps is empty
        assert len(self.univariate_snps) == 0
        # make sure that the uni_snps are not empty
        assert len(uni_snps) > 0
        # update
        self.univariate_snps = cp.deepcopy(uni_snps)
        return

    # set traits
    def set_traits(self, traits: traits_t) -> None:
        # check that internal traits is empty
        assert len(self.traits) == 0
        # make sure that the traits are not empty
        assert len(traits) == 3
        # make sure correct types
        assert isinstance(traits[0], r2_t)
        assert isinstance(traits[1], feature_cnt_t)
        # print("Type of traits[2]:", type(traits[2]))
        assert isinstance(traits[2], feature_name_t)
        # make sure we have a non-negative number of features
        assert traits[1] >= 0

        # update the traits
        self.traits = cp.deepcopy(traits)
        return

    def get_trait_r2(self) -> r2_t:
        assert len(self.traits) == 3
        return self.traits[0]

    def get_trait_feature_cnt(self) -> feature_cnt_t:
        assert len(self.traits) == 3
        assert self.traits[1] >= 0 # make sure we have a non-negative number of features
        return self.traits[1]

    def get_trait_feature_names(self) -> feature_name_t:
        assert len(self.traits) == 3
        assert self.traits[2] is not None
        return self.traits[2]

    def get_traits(self) -> traits_t:
        return self.traits

    def get_uni_snps(self) -> uni_snps_t:
        return self.univariate_snps

    # method to get the number of nodes in the pipeline
    def get_uni_count(self):
        return len(self.univariate_snps)

    # method to get the number of features/SNPs in the pipeline
    def get_feature_count(self):
        return self.selector_node.get_feature_count()

    def get_ld_snp_details_after_ld(self):
        return self.ld_node.snp_details_after_ld

    def get_ld_name_of_selected_feature(self):
        return self.ld_node.name_of_selected_features

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
        print("UnivariateNodes:")
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

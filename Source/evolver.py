#####################################################################################################
#
# Evolutionary algorithm class that evolves pipelines.
# We use the NSGA-II algorithm to evolve pipelines.
# Pipelines consist of a set of epistatic interactions, feature selector, and a final regressor.
#
#####################################################################################################

import numpy as np
from sklearn.inspection import permutation_importance
from typeguard import typechecked
from typing import List, Dict
import pandas as pd
import os
import ray
from .pipeline import Pipeline
from sklearn.model_selection import train_test_split
from .snp_hub import SnpHub
from typing import List, Tuple, Set
import copy as cp
from sklearn.metrics import r2_score

from .uni_node import UniNode
from .uni_node import UniAdditiveNode, UniDominantNode, UniRecessiveNode, UniHeterosisNode, UniUnderDominantNode, UniSubadditiveNode, UniSuperadditiveNode, UniPAGERNode

from .scikit_node import ScikitNode, LDSelector
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from .reproduction import Reproduction
import numpy.typing as npt
from . import nsga_tool as nsga
import logging
import warnings
from sklearn.exceptions import NotFittedError, ConvergenceWarning
import matplotlib.pyplot as plt
from .poster import Poster
import time
import warnings

# don't show runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# snp name type
snp_name_t = np.str_
# snp hub position type
snp_hub_pos_t = np.uint32
# uni node list type
uni_node_list_t = List[UniNode]
# probability type: needed to avoid rounding errors with probabilities
prob_t = np.float64
# r2 type
r2_t = np.float32
# node type/logcial operation type
nodelo_t = np.str_
# feature count type
feature_cnt_t = np.int16
# list of feature names type (for the LD node)
feature_names_t = List
# population id type
pop_id_t = np.uint16
# diversity score type
div_t = np.float32
# snp hub generation type
snp_hub_gen_t = np.int32

# evaluate unseen snps
@ray.remote
def ray_uni_eval(x_train,
                y_train,
                x_val,
                y_val,
                snp_name: snp_name_t,
                snp_pos: snp_hub_pos_t) -> Tuple[np.float32, np.str_, np.str_]:
    # hold results
    best_uni = ''
    best_res = -1.0

    # holds all lo's we are going to evaluate
    unis = {np.str_('additive'): UniAdditiveNode,
            np.str_('dominant'): UniDominantNode,
            np.str_('recessive'): UniRecessiveNode,
            np.str_('heterosis'): UniHeterosisNode,
            np.str_('underdominant'): UniUnderDominantNode,
            np.str_('subadd'): UniSubadditiveNode,
            np.str_('superadd'): UniSuperadditiveNode,
            np.str_('pager'): UniPAGERNode,
            }

    # iterate over the uni node types and create sklearn pipeline
    for lo, uni in unis.items():
        steps = []
        # create the epi node
        uni_node = uni(name=lo, snp_name=snp_name, snp_pos=snp_pos)
        steps.append((lo, uni_node))

        # add random forrest regressor
        steps.append(('regressor', LinearRegression()))

        # create the pipeline
        skl_pipeline = SklearnPipeline(steps=steps)

        # Fit the pipeline
        skl_pipeline_fitted = skl_pipeline.fit(x_train, y_train)

        # get score
        r2 = skl_pipeline_fitted.score(x_val, y_val)

        # check if this is the best lo
        if r2 > best_res:
            best_res = r2
            best_uni = lo

    return r2_t(best_res), nodelo_t(best_uni), snp_name

@ray.remote
# LD first then FS
def ray_eval_pipeline(x_train,
                      y_train,
                      x_val,
                      y_val,
                      uni_nodes: uni_node_list_t,
                      selector_node: ScikitNode,
                      ld_node: LDSelector,
                      root_node: ScikitNode,
                      pop_id: np.int16,        #    r2, feature count, pop_id, one_snp_only_pipeline, pruned, snp_name_after_ld
                      snp_r2_set: Set) -> Tuple[np.float32, np.uint16, np.int16, snp_name_t, List[np.str_], List[np.str_]]:

    # make dictionary to hold the snp r2 scores
    snp_r2_dict = {p[0]: p[1] for p in snp_r2_set}

    # create the pipeline
    steps = []
    # uni nodes into one sklearn union
    steps.append(('snp_union', FeatureUnion([(uni_node.name, uni_node) for uni_node in uni_nodes])))
    # make a list of uni node names
    uni_node_names = [uni_node.get_snp_name() for uni_node in uni_nodes]

    # fit the pipeline to get the selected features
    pipeline = SklearnPipeline(steps=steps)
    try:
        pipeline_fitted = pipeline.fit(x_train, y_train)
    except Exception as e:
        # Catch all other exceptions and log error with relevant context
        logging.error(f"Exception while fitting SNP union step: {e}")
        return r2_t(-1.0), feature_cnt_t(0), pop_id, False, (), []

    # transform the dataset using snp_union
    x_train_transformed = pipeline_fitted.transform(x_train)
    x_val_transformed = pipeline_fitted.transform(x_val)
    x_train_transformed_df = pd.DataFrame(x_train_transformed, columns=uni_node_names)
    x_val_transformed_df = pd.DataFrame(x_val_transformed, columns=uni_node_names)
    if x_train_transformed_df.empty:
        return r2_t(-1.0), feature_cnt_t(0), pop_id, False, (), []

    x_train_original_df = pd.DataFrame(x_train, columns=uni_node_names)

    # Fit the LD node
    try:
        ld_node.fit(x_train_original_df, x_train_transformed_df, y_train, snp_r2_dict)
        selected_features_after_ld = ld_node.selected_features_
        # keeping only the selected features after the LD node
        x_train_transformed_df = pd.DataFrame(x_train_transformed_df[selected_features_after_ld], columns=selected_features_after_ld)
        if x_train_transformed_df.empty:
            return r2_t(-1.0), feature_cnt_t(0), pop_id, False, (), []
        x_val_transformed_df = pd.DataFrame(x_val_transformed_df[selected_features_after_ld], columns=selected_features_after_ld)
    except Exception as e:
        logging.error(f"Exception while fitting LD node: {e}")
        return r2_t(-1.0), feature_cnt_t(0), pop_id, False, (), []

    # adding the selector and regressor nodes
    try:
        # create the pipeline
        steps = []
        # add the selector node
        steps.append(('selector', selector_node))
        # pass to regressor
        steps.append(('regressor', root_node.regressor))
        # create the pipeline without refitting the regressor
        pipeline = SklearnPipeline(steps=steps)
        pipeline.fit(x_train_transformed_df, y_train)
    except Exception as e:
        logging.error(f"Exception while fitting pipeline after LD: {e}")
        return r2_t(-1.0), feature_cnt_t(0), pop_id, False, (), []

    try:

        r2_score = pipeline.score(x_val_transformed_df, y_val)
        feature_count = pipeline.named_steps['selector'].get_feature_count() # number of selected features after the selector node
        features_final = (pipeline.named_steps['selector'].get_feature_names(selected_features_after_ld)) # get the names of the features after the selector node by sending the selected features after the LD node
        # if features_final is not a list, convert it to a list
        if not isinstance(features_final, list):
            features_final = features_final.tolist()
    except Exception as e:
        logging.error(f"Error while scoring or getting feature count: {e}")
        return r2_t(-1.0), feature_cnt_t(0), pop_id, False, (), []

    one_snp_only_pipeline = 'N/A'
    if features_final != None:
        one_snp_only_pipeline = features_final

    # return the pipeline
    return r2_t(r2_score), feature_cnt_t(feature_count), pop_id, snp_name_t(one_snp_only_pipeline), [snp_name_t(k) for k,v in ld_node.snp_details_after_ld.items() if v == True], features_final

@typechecked # for debugging purposes
class EA:
    def __init__(self,
                 seed: np.uint16,
                 pop_size: np.uint16,
                 uni_cnt_max: np.uint16,
                 uni_cnt_min: np.uint16,
                 cores: int,
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
                 save_directory: str = ""
                 ) -> None:
        """
        Main class for the evolutionary algorithm.

        Parameters:
        seed: np.uint16
            Seed for the random number generator.
        pop_size: np.uint16
            Population size.
        epi_cnt_max: np.uint16
            Maximum number of epistatic interactions (nodes).
        uni_cnt_max: np.uint16
            Maximum number of univariate snps (nodes).
        uni_cnt_min: np.uint16
            Minimum number of univariate snps (nodes).
        mut_ran_p: prob_t
            Probability for random mutation.
        mut_smt_p: prob_t
            Probability for smart mutation.
        smt_in_in_p: prob_t
            Probability for smart mutation, new snp comes from the same chromosome and assigned bin.
        smt_in_out_p: prob_t
            Probability for smart mutation, new snp comes from the same chromosome but different bin.
        smt_out_out_p: prob_t
            Probability for smart mutation, new snp comes from a different chromosome (different bin by definition).
        mut_prob: prob_t
            Probability for mutation ocurring.
        cross_prob: prob_t
            Probability for crossover ocurring.
        """

        # arguments needed to run
        self.seed = seed
        self.pop_size = pop_size
        self.rng = np.random.default_rng(seed) # random number generator to be passed to all other stocastic functions
        self.uni_cnt_max = uni_cnt_max #YF uni arguments
        self.uni_cnt_min = uni_cnt_min
        self.mut_ran_p = mut_ran_p
        self.mut_smt_p = mut_smt_p
        self.mut_prob = mut_prob
        self.cross_prob = cross_prob
        self.smt_in_in_p = smt_in_in_p
        self.smt_in_out_p = smt_in_out_p
        self.smt_out_out_p = smt_out_out_p
        self.population = [] # will hold all the pipelines
        self.repoduction = Reproduction(uni_cnt_max=uni_cnt_max,
                                        uni_cnt_min=uni_cnt_min,
                                        mut_prob=mut_prob,
                                        cross_prob=cross_prob,
                                        mut_selector_p=mut_selector_p,
                                        mut_ld_p=mut_ld_p,
                                        mut_regressor_p=mut_regressor_p,
                                        mut_ran_p=mut_ran_p,
                                        mut_smt_p=mut_smt_p,
                                        smt_in_in_p=smt_in_in_p,
                                        smt_in_out_p=smt_in_out_p,
                                        smt_out_out_p=smt_out_out_p)
        self.save_directory = save_directory


        # Initialize Ray: Will have to specify when running on hpc
        ray.init(num_cpus=cores, include_dashboard=True)
        print(flush=True)

    # data loader
    def data_loader(self, path: str, target_label: str = "y", split: float = 0.50) -> None:
        """
        Function to load data from a csv file into a pandas dataframe.
        We assume that the target label is 'y', unless otherwise specified.
        At the end of the function, we partition the data into training and validation sets.
        Additionally, we load the data into the ray object store and intialize hubs.

        Parameters:
        path: str
            Path to the file.
        target_label: str
            Name of the target label.
        """

        print('Loading data...', flush=True)
        print('Path:', path, flush=True)

        # check if the path is valid
        if os.path.isfile(path) == False:
            # load the data
            exit('Error: The path provided is not valid. Please provide a valid path to the data file.', -1)

        # data = pd.read_csv(path)
        # print('Data loaded successfully.', flush=True)
        # print("Data shape:", data.shape, flush=True)

        # get pandas dataframe snp names without loading all data
        self.snp_labels = pd.read_csv(path, nrows=0).columns.tolist()

        # check if the target label is valid
        if target_label not in self.snp_labels:
            exit('Error: The target label provided is not valid. Please provide a valid target label.', -1)

        # remove target label from snp labels
        self.snp_labels.remove(target_label)

        # convert python strings into numpy strings
        self.snp_labels = np.array(self.snp_labels, dtype=np.str_)
        self.target_label = np.str_(target_label)

        # load the data
        all_x = pd.read_csv(filepath_or_buffer=path, usecols=self.snp_labels)
        all_y = pd.read_csv(filepath_or_buffer=path, usecols=[self.target_label]).values.ravel()

        # check if the data was loaded correctly
        all_x, all_y = self.check_dataset(all_x, all_y)
        print('X_data.shape:', all_x.shape, flush=True)
        print('y_data.shape:', all_y.shape, flush=True)
        print(flush=True)

        # change all the 1 in all_x to 0.5, all 2 to 1 in all_x - changing the additive encoding from 0,1,2 to 0,0.5,1
        all_x = all_x.replace(1, 0.5)
        all_x = all_x.replace(2, 1)

        # checking the encoding
        print("Genotype data: ", all_x, flush=True)

        # partition data based splits
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(all_x, all_y, test_size=split, random_state=self.seed)

        # check if the data was partitioned correctly
        self.X_train, self.y_train = self.check_dataset(self.X_train, self.y_train)
        self.X_val, self.y_val = self.check_dataset(self.X_val, self.y_val)

        # load data into ray object store
        self.X_train_id = ray.put(self.X_train)
        self.y_train_id = ray.put(self.y_train)
        self.X_val_id = ray.put(self.X_val)
        self.y_val_id = ray.put(self.y_val)

        print('X_train_new.shape:', self.X_train.shape, flush=True)
        print("X_train values: ", self.X_train, flush=True)
        print('y_train_new.shape:', self.y_train.shape, flush=True)
        print(flush=True)
        print('X_val_new.shape:', self.X_val.shape, flush=True)
        print('y_val_new.shape:', self.y_val.shape, flush=True)
        print(flush=True)
        print('Data loaded successfully.', flush=True)
        print(flush=True)
        return

    # data checker to check for validity of dataset
    def check_dataset(self, features, target):
        """
        Check if a dataset has a valid feature set and labels.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        target: array-like {n_samples} or None
            List of class labels for prediction
        sample_weight: array-like {n_samples} (optional)
            List of weights indicating relative importance
        """
        # # check for missing values
        # if isinstance(features, pd.DataFrame):
        #         for col in features.columns:
        #             #if features[col].isnull().values.any():
        #                features[col].fillna(features[col].mode()[0], inplace=True)

        # Check if features is a DataFrame and handle missing values
        if isinstance(features, pd.DataFrame):
            for col in features.columns:
                if features[col].isnull().any():
                    # Calculate mode and handle edge cases
                    mode_values = features[col].mode()
                    if not mode_values.empty:
                        features[col] = features[col].fillna(mode_values[0])
                        print(f"Column '{col}' contains missing values. Imputed with mode value: {mode_values[0]}.")
                    else:
                        raise ValueError(f"Cannot calculate mode for column '{col}' due to missing or ambiguous data.")

        # check for target
        try:
            if target is not None:
                return features, target
            else:
                return features
        except (AssertionError, ValueError):
            raise ValueError(
                "Error: Input data is not in a valid format. Please confirm "
                "that the input data is scikit-learn compatible. For example, "
                "the features must be a 2-D array and target labels must be a "
                "1-D array."
            )

    def initialize_hubs(self, bin_size:int) -> None:
        """
        Function to initialize the hubs for the evolutionary algorithm.
        This function must be called after the data has been loaded.

        Parameters
        ----------
        hub_size: int
            Number of hubs to initialize.
        """
        # initialize the hubs
        self.hubs = SnpHub(snps=self.snp_labels, bin_size=np.uint16(bin_size))

    # will evolve a population of pipelines for a specified number of generations
    def evolve(self, gens: int) -> None:
        """
        Function to evovle pipelines using the NSGA-II algorithm for a user specified number of generations.
        We also take in the training and validation data to evaluate the pipelines.

        Parameters:
        gens: np.uint16
            Number of generations to run the algorithm.

        """
        # create the initial population
        print('Initializing population...', flush=True)
        start_time = time.time()
        self.initialize_population()
        print(f"Population initialized in {(time.time() - start_time) / 60 / 60} hours", flush=True)
        print('Entering evolutionary proccess.\n', flush=True)

        # run the algorithm for the specified number of generations
        for g in range(gens):
            # make sure we have the correct number of pipelines
            print('Population size:', len(self.population), flush=True)
            assert(0 < len(self.population) <= self.pop_size)

            print('Generation:', g, flush=True)
            start_time = time.time()

            # how many extra pipeline offspring are needed to reach 2*N potentially surviving solutions
            extra_offspring = self.pop_size - len(self.population)
            # get order of mutation/crossover to do with the extra offspring
            var_order, parent_cnt = self.repoduction.variation_order(self.rng, np.uint16(extra_offspring + self.pop_size))

            # get the parent scores by position
            parent_ids = self.parent_selection(parent_cnt)

            # generate offspring
            offspring = self.repoduction.produce_offspring(rng_ = self.rng,
                                                           hub = self.hubs,
                                                           offspring_cnt=np.uint16(extra_offspring + self.pop_size),
                                                           parent_ids=parent_ids,
                                                           population=self.population,
                                                           order=var_order)
            # make sure we have the correct number of competing solutions
            assert len(offspring) + len(self.population) == 2 * self.pop_size

            # process offspring: evaluation interactions and remove bad interactions
            offspring = self.process_offspring(offspring, snp_hub_gen_t(g))

            # evaluate the offspring
            offspring = self.evaluation(offspring, snp_hub_gen_t(g))

            # remove bad snps from the offspring
            offspring = self.clean_pipeline_post_eval(offspring)

            # must be less than or equal bc of potential negative r2 offspring pipelines
            assert (0 < len(offspring) + len(self.population) <= 2 * self.pop_size)

            # survival selection
            self.population = self.survival_selection(self.population, offspring)

            # make sure we have the correct number of pipelines
            assert len(self.population) == self.pop_size

            print(f"Time to finish generation: {(time.time() - start_time) / 60} minutes", flush=True)

        # plot the pareto front
        self.plot_pareto_front() # calling the plotting function at the end to get the final pareto plot
        # save the epi_hub to a csv file in the save directory
        self.hubs.save_hubs(self.save_directory)

    # get list of pipeline scores (r2, complexity) by position
    def get_pipeline_scores(self, pipelines: List[Pipeline], weights: Tuple[r2_t, feature_cnt_t]) -> npt.NDArray:
        """
        Function to get the pipeline scores (r2, complexity) by position.
        Will also apply weights to the scores, so that we can use NSGA-II to get the pareto front.
        """
        scores = np.empty(len(pipelines), dtype=object)
        for i, pipeline in enumerate(pipelines):
            scores[i] = (pipeline.get_trait_r2() * weights[0], pipeline.get_trait_feature_cnt() * weights[1])

        return scores

    # survival selection
    def survival_selection(self, pop1: List[Pipeline], pop2: List[Pipeline]) -> List[Pipeline]:
        """
        Function to select the survivors from the current population and offspring.

        Parameters:
        pop1: List[Pipeline]
            First list of pipelines
        pop1_scores: List[Tuple[np.float32, np.uint16]]
            First list of pipeline scores
        pop2: List[Pipeline]
            Second list of pipelines
        pop2_scores: List[Tuple[np.float32, np.uint16]]
            Second list of pipeline scores
        """
        # make sure all population scores are positive
        assert all(pipeline.get_trait_r2() > 0.0 for pipeline in pop1)
        assert all(pipeline.get_trait_r2() > 0.0 for pipeline in pop2)

        # combine both the population and offspring lists into one
        pipelines_original = pop1 + pop2

        # iterate through the combined pipelines and remove duplicates with the same get_trait_feature_names
        best_pipelines = {}
        for pipeline in pipelines_original:
            # Convert features to a frozenset so it can be used as a dict key
            feats = frozenset(pipeline.get_trait_feature_names())

            if feats not in best_pipelines:
                best_pipelines[feats] = pipeline
            else:
                current_best = best_pipelines[feats]
                if pipeline.get_trait_r2() > current_best.get_trait_r2():
                    # Found a strictly better pipeline for this feature set
                    best_pipelines[feats] = pipeline
                elif pipeline.get_trait_r2() == current_best.get_trait_r2():
                    # Tie: pick randomly
                    if self.rng.choice([True, False]):
                        best_pipelines[feats] = pipeline

        # get the best pipelines
        non_dup_pipelines = list(best_pipelines.values())

        # get the fronts and rank
        fronts, _ = nsga.non_dominated_sorting(obj_scores=self.get_pipeline_scores(non_dup_pipelines, (r2_t(1.0), feature_cnt_t(-1))))

        # get crowding distance for each solution
        crowding_distance = nsga.crowding_distance(self.get_pipeline_scores(non_dup_pipelines, (r2_t(1.0), feature_cnt_t(1))), np.int32(2))

        # truncate the population to the population size with nsga ii
        survivor_ids = nsga.non_dominated_truncate(fronts, crowding_distance, self.pop_size)
        # make sure that the number of survivors is correct
        assert len(survivor_ids) == self.pop_size

        # subset the candidates to only include the survivors
        new_pop = []

        for i in survivor_ids:
            # make sure we are within the bounds of the candidates
            assert 0 <= i < len(non_dup_pipelines)
            new_pop.append(non_dup_pipelines[i])

        return new_pop

    # initialize the starting population
    def initialize_population(self) -> None:
        """
        Initialize the population of pipelines with their set of epistatic interactions.
        We start by creating (2 * pop_size) random set of interactions and find their best lo from the 9 we use.
        After, we remove bad interactions (negative r2) for a given set and create pipelines from the good interactions.
        We then evaluate the pipelines and keep only the pipelines with positive r2 scores.

        If the number of pipelines with positive r2 scores is less than the population size, we keep the same population.
        If the number of pipelines with positive r2 scores is greater than the population size, we use NSGA-II to get the pareto front.
        """

        # will hold a set of snps for each pipeline in the population
        pop_univariate_sets = []
        # will hold unseen snps -- snps whose best encoder type is empty
        unseen_snps = set()

        # create the initial population
        for _ in range(self.pop_size):
            # holds all interactions we are doing
            # set to make sure we don't have duplicates
            snps = set()

            # get a random number of univariate snps
            uni_cnt = self.rng.integers(low=self.uni_cnt_min, high=self.uni_cnt_max, endpoint=True)

            while len(snps) <= uni_cnt:
                # get random snp and add to snps
                snp = self.hubs.get_ran_snp(self.rng)
                # add snp to the snps set
                snps.add(snp)

            new_snp = set(snp for snp in snps
                            if not self.hubs.is_encoder_in_hub(snp))
            unseen_snps.update(new_snp)
            # add to the population
            pop_univariate_sets.append(snps)

        # make sure we have the correct number of interactions
        assert len(pop_univariate_sets) == self.pop_size

        # evaluate all unseen interactions
        self.evaluate_unseen_snps(unseen_snps, snp_hub_gen_t(0))

        # remove bad snps for each pipeline's set of snps
        for snps in pop_univariate_sets:
            good_snps = self.remove_bad_snps(snps)

            # make sure we have the correct number of good interactions
            assert len(good_snps) <= len(snps)

            # make sure we have more than 0 good interactions
            if len(good_snps) == 0:
                # skip this iteration if there are no good snps
                continue

            # create pipeline and add to the population
            self.population.append(self.repoduction.generate_random_pipeline(self.rng, good_snps, int(self.seed)))

        # make sure we have the correct number of pipelines
        assert (0 < len(self.population) <= self.pop_size)

        # evaluate the initial population
        self.population = self.evaluation(self.population, snp_hub_gen_t(0))

        # remove bad snps from the population
        self.population = self.clean_pipeline_post_eval(self.population)

        # make sure we have the correct number of pipelines
        assert (0 < len(self.population) <= self.pop_size)

        return

    # evaluate all unevaluated snps and update
    def evaluate_unseen_snps(self, unseen_snps: Set, gen_seen: snp_hub_gen_t) -> None:
        """
        Function to evaluate all unseen snps and add their best R2 and Encoder type to the SnpHub.
        All of this should be done in asyncronous parallel jobs.
        We update the SnpHub with the results as they come in.

        Parameters:
        unseen_snps: Set
            Unseen snps in a set to evaluate.
        """
        ray_jobs = []
        # collect all ray jobs for evaluation
        for snp_name in unseen_snps:
            ray_jobs.append(ray_uni_eval.remote(x_train = self.X_train_id,
                                                y_train = self.y_train_id,
                                                x_val = self.X_val_id,
                                                y_val = self.y_val_id,
                                                snp_name = snp_name,
                                                snp_pos = self.hubs.get_snp_pos(snp_name)))
        assert len(ray_jobs) == len(unseen_snps)

        # process results as they come in
        while len(ray_jobs) > 0:
            finished, ray_jobs = ray.wait(ray_jobs)
            r2, type, snp_name = ray.get(finished)[0]
            self.hubs.update_snp_hub(snp_name, r2, type, gen_seen)

    # remove bad snps: r2 < 0 and snp has been prunned
    def remove_bad_snps(self, snps: Set) -> Set:
        """
        Function to remove bad snps with r2<0 for a given set of snps

        Parameters:
        snps: Set of snps
        """
        good_snps = set()
        for snp_name in snps:
            # check if r2 is positive
            if self.hubs.get_uni_res(snp_name) > np.float32(0.0) and self.hubs.has_been_prunned(snp_name) == False:
                # add to good snps
                good_snps.add(snp_name)
        # return the good snps
        return good_snps

    # print the population
    def print_population(self) -> None:
        """
        Function to print the population.
        """
        print('Population:', flush=True)
        for p in self.population:
            p.print_pipeline()

    # evaluate the population                                   # r2 , feature count, pop_id
    def evaluation(self, pop: List[Pipeline], gen_pruned: snp_hub_gen_t) -> List[Pipeline]:
        """
        Function to evaluate entire pipelines.
        We will evaluate all unseen interactions and snps and add them to the SnpHub.
        All of this is done in asyncronous parallel jobs.
        We update the SnpHub with the results as they come in.

        Parameters:
        unseen_interactions: Set[Tuple]
            Set of unseen interactions to evaluate.
        """

        # collect all parallel jobs
        ray_jobs = []
        # go through each pipeline in the population and evaluate
        # collect all ray jobs for evaluation
        for i, pipeline in enumerate(pop):
            ray_jobs.append(ray_eval_pipeline.remote(self.X_train_id,
                                                self.y_train_id,
                                                self.X_val_id,
                                                self.y_val_id,
                                                self.construct_uni_nodes(pipeline.get_uni_snps()),
                                                pipeline.get_selector_node(),
                                                pipeline.get_ld_node(),
                                                pipeline.get_root_node(),
                                                np.int16(i),
                                                snp_r2_set=self.hubs.generate_r2_dict(pipeline.get_uni_snps())))

        assert len(ray_jobs) == len(pop)

        # keep track of prunned snps
        prunned_snps = set()

        # process results as they come in
        while len(ray_jobs) > 0:
            finished, ray_jobs = ray.wait(ray_jobs)
            r2, feature_count, pop_id, one_snp_only_pipeline, pruned, feature_names = ray.get(finished)[0]
            # update the pipeline
            pop[pop_id].set_traits([r2, feature_count, set(feature_names)])

            new_snp = set(snp for snp in pruned
                            if not self.hubs.has_been_prunned(snp))
            prunned_snps.update(new_snp)

        # process prunned snps
        self.hubs.process_prunned_snps(prunned_snps, gen_pruned)

        # collect only pipelines that do not consist of only prunned snps
        new_pop = []

        for pipeline in pop:
            if self.hubs.all_snps_prunned(pipeline.get_uni_snps()) == False and pipeline.get_trait_r2() > 0.0:
                new_pop.append(pipeline)

        print('non pruned hub size:' , self.hubs.pruned_hub_size())

        return new_pop

    # construct uni_nodes for a pipeline's set of individual snps
    def construct_uni_nodes(self, uni_snps: Set) -> List[UniNode]:
        """
        Function to construct uni nodes for a pipeline's set of univariates.

        Parameters:
        uni_snps: Set
            Set of snps.
        """
        # make sure there are snps to construct
        assert len(uni_snps) > 0

        uni_nodes = []
        id = 0
        for snp_name in uni_snps:
            # get the uni snp type
            uni_type = self.hubs.get_uni_encoding(snp_name)
            # get each snps position in the hub
            snp_pos = self.hubs.get_snp_pos(snp_name)

            if uni_type == np.str_('additive'):
                uni_nodes.append(UniAdditiveNode(name=f"UniAdditiveNode_{id}", snp_name=snp_name, snp_pos=snp_pos))
            elif uni_type == np.str_('dominant'):
                uni_nodes.append(UniDominantNode(name=f"UniDominantNode_{id}", snp_name=snp_name, snp_pos=snp_pos))
            elif uni_type == np.str_('recessive'):
                uni_nodes.append(UniRecessiveNode(name=f"UniRecessiveNode_{id}", snp_name=snp_name, snp_pos=snp_pos))
            elif uni_type == np.str_('heterosis'):
                uni_nodes.append(UniHeterosisNode(name=f"UniHeterosisNode_{id}", snp_name=snp_name, snp_pos=snp_pos))
            elif uni_type == np.str_('underdominant'):
                uni_nodes.append(UniUnderDominantNode(name=f"UniUnderDominantNode_{id}", snp_name=snp_name, snp_pos=snp_pos))
            elif uni_type == np.str_('subadd'):
                uni_nodes.append(UniSubadditiveNode(name=f"UniSubadditiveNode_{id}", snp_name=snp_name, snp_pos=snp_pos))
            elif uni_type == np.str_('superadd'):
                uni_nodes.append(UniSuperadditiveNode(name=f"UniSuperadditiveNode_{id}", snp_name=snp_name, snp_pos=snp_pos))
            elif uni_type == np.str_('pager'):
                uni_nodes.append(UniPAGERNode(name=f"UniPAGERNode_{id}", snp_name=snp_name, snp_pos=snp_pos))
            else:
                exit('Error: The univariate snp type is not valid. Please provide a valid type.', -1)

            id += 1
        # return the list of epi nodes
        return uni_nodes

    # parent selection
    def parent_selection(self, parent_cnt: pop_id_t) -> List[pop_id_t]:
        """
        Function to return a specified number of parent ids based on the scores of the pipelines.

        Parameters:
        scores: List[Tuple[np.float32, np.int16, np.int16]] # r2 , feature count, pop_id
            Scores of the pipelines (scores: np.array([r2,complexity])).
        """
        # will hold the parent ids
        parent_ids = []

        # get the fronts and rank
        fronts, ranks = nsga.non_dominated_sorting(obj_scores=self.get_pipeline_scores(self.population, (r2_t(1.0), feature_cnt_t(-1))))
        # make sure that the number of fronts is correct
        assert sum([len(f) for f in fronts]) == len(ranks)

        # get crowding distance for each solution
        crowding_distance = nsga.crowding_distance(self.get_pipeline_scores(self.population, weights=(r2_t(1.0), feature_cnt_t(1))), np.int32(2))

        # get parent_cnt number of parents
        for _ in range(parent_cnt):
            parent_ids.append(nsga.non_dominated_binary_tournament(rng_=self.rng, ranks=ranks, distances=crowding_distance))
        # make sure that the number of parents is correct
        assert len(parent_ids) == parent_cnt

        return parent_ids

    # process offspring: evaluate new snps, remove bad snps, and create pipelines with good snps
    def process_offspring(self, pipelines: List[Pipeline], gen_found: snp_hub_gen_t) -> List[Pipeline]:
        # get unseen interactions
        unseen_snps = self.get_unseen_univariates(pipelines)

        # evaluate all unseen interactions
        self.evaluate_unseen_snps(unseen_snps, gen_found)

        # remove bad interactions for each pipeline's set of interactions
        updated_pipelines = []
        for pipeline in pipelines:
            # remove bad snps and make sure more than 0 snps are left
            good_snps = self.remove_bad_snps(pipeline.get_uni_snps())

            if len(good_snps) == 0:
                # skip this iteration if there are no good snps
                continue

            updated_pipelines.append(Pipeline(uni_snps=good_snps,
                                              selector_node=pipeline.get_selector_node(),
                                              ld_node=pipeline.get_ld_node(),
                                              root_node=pipeline.get_root_node(),
                                              traits=[]))
        return updated_pipelines

    def clean_pipeline_post_eval(self, pipelines: List[Pipeline]) -> List[Pipeline]:
        # remove bad interactions for each pipeline's set of interactions
        updated_pipelines = []
        for pipeline in pipelines:
            # remove bad snps and make sure more than 0 snps are left
            good_snps = self.remove_bad_snps(pipeline.get_uni_snps())

            if len(good_snps) == 0:
                # skip this iteration if there are no good snps
                continue

            updated_pipelines.append(Pipeline(uni_snps=good_snps,
                                              selector_node=pipeline.get_selector_node(),
                                              ld_node=pipeline.get_ld_node(),
                                              root_node=pipeline.get_root_node(),
                                              traits=pipeline.get_traits()))
        return updated_pipelines

    def get_unseen_univariates(self, pipelines: List[Pipeline]) -> Set[snp_name_t]:
        """
        Function to get all unseen univariates from the pipelines.

        Parameters:
        pipelines: List[Pipeline]
            List of pipelines.
        """
        unseen_univariates = set()
        for pipeline in pipelines:
            for snp_name in pipeline.get_uni_snps():
                if not self.hubs.is_encoder_in_hub(snp_name):
                    unseen_univariates.add(snp_name)
        return unseen_univariates

    # plot the current pareto front from the population with complexity and r2 scores
    def plot_pareto_front(self) -> None:
        """
        Function to plot the current pareto front from the population with complexity and r2 scores.
        """

        # get all scores from the current population
        pop_scores = self.get_pipeline_scores(self.population, weights=(r2_t(1.0), feature_cnt_t(1)))

        # get the fronts and rank
        fronts, rank = nsga.non_dominated_sorting(obj_scores=self.get_pipeline_scores(self.population, weights=(r2_t(1.0), feature_cnt_t(-1))))

        # remove scores that are not of rank 0
        pareto_front = pop_scores[rank == 0]
        print('pareto front:', pareto_front, flush=True)

        # plot the pareto front
        plt.scatter([t[1] for t in pareto_front], [t[0] for t in pareto_front])
        plt.xlabel('Feature Count')
        plt.ylabel('R2 Score')
        plt.title('Final Pareto Front')

        # Annotate the points with pipeline numbers (indexes in pareto front)
        for i, (r2_score, feature_count) in enumerate(pareto_front):
            plt.annotate(
                str(i + 1),  # Text label (pipeline number)
                (feature_count, r2_score),  # The point where the annotation should be
                textcoords="offset points",  # Use offset for better readability
                xytext=(5, 5),  # Offset position (x, y)
                ha='center',  # Horizontal alignment
                fontsize=9,
                color='red'
            )

        # show grid
        plt.grid(True)
        # save the plot
        plt.savefig(self.save_directory + 'pareto_front.png')
        plt.clf()

    # Create a object of Poster class to check the post analyis of the pipelines
    # to do the feature importance analysis
    def post_analysis(self) -> None:
        """
        Function to perform post analysis of the pipelines.
        """

        #epi_nodes = [self.construct_epi_nodes(pipeline.get_epi_pairs()) for pipeline in self.population]

        ###################### create the pareto front #######################

        # get the fronts and rank
        fronts, rank = nsga.non_dominated_sorting(obj_scores=self.get_pipeline_scores(self.population, weights=(r2_t(1.0), feature_cnt_t(-1))))


        pareto_front = []
        # get rank == 0 pipelines
        for i, r in enumerate(rank):
            if r == 0:
                pareto_front.append(self.population[i])

        print('Size of Pareto Front:', len(pareto_front), flush=True)


        # create a poster object
        #poster = Poster(self.X_train_id, self.y_train_id, self.X_val_id, self.y_val_id, hub=self.hubs)

        # Initialize an empty DataFrame to save all shap_values_df
        all_perm_imp_df = pd.DataFrame()

        # print the pipelines in the population
        ray_jobs = []
        for i, pipeline in enumerate(pareto_front): # think this as pareto front pipelines
            # # printing pipeline details from evolver
            # print("Pipeline ID from EVOLVER function:", i, flush=True)
            # print("Pipeline R2 Score from EVOLVER function:", pipeline.get_trait_r2(), flush=True)
            # print("Pipeline Feature Count from EVOLVER function:", pipeline.get_trait_feature_cnt(), flush=True)
            # print("Pipeline feature names after LD node from EVOLVER function:", pipeline.get_trait_feature_names(), flush=True)
            # get the uni_nodes - has the SNP Name, SNP position and the best encoder type
            uni_nodes = self.construct_uni_nodes(pipeline.get_uni_snps())
            # make a uni_snps_df - which will have the feature and the best inheritence type
            uni_snps_list = []
            uni_snps = pipeline.get_uni_snps()
            # print('uni_snps:',uni_snps, flush=True)
            for snp_name in uni_snps:
                best_lo = self.hubs.get_uni_encoding(snp_name)
                uni_snps_list.append({'feature': snp_name, 'inheritence': best_lo})
            uni_snps_df = pd.DataFrame(uni_snps_list)

            #uni_snps_df = poster.get_uni_snp(pipeline=pipeline)
            features_final = pipeline.get_trait_feature_names()
            # filter uni_nodes to only include the snps in the pipeline
            uni_nodes = [uni_node for uni_node in uni_nodes if uni_node.get_snp_name() in features_final]
            # print('uni_nodes:', uni_nodes, flush=True)
            # filter uni_snps_df to only include the snps in the pipeline
            uni_snps_df = uni_snps_df[uni_snps_df['feature'].isin(features_final)]

            # change the feature names to have the inheritance information - from the filtered uni_snps_df
            new_column_names = [ f'{row["feature"]}_{row["inheritence"]}' for _, row in uni_snps_df.iterrows()]

            # Construct the snp_union FeatureUnion - to transform the training and test datasets to have the encoded SNPs
            snp_union = FeatureUnion([(uni_node.name, uni_node) for uni_node in uni_nodes])
            steps = []
            steps.append(('snp_union',snp_union))
            pipe = SklearnPipeline(steps=steps)
            pipe.fit(self.X_train, self.y_train)

            # Transform the training and test datasets using snp_union
            uni_features_train = pipe.transform(self.X_train)
            uni_features_test = pipe.transform(self.X_val)

            # Get R2 and Feature Count for this specific pipeline
            pipeline_r2 = pipeline.get_trait_r2()
            pipeline_feature_count = pipeline.get_trait_feature_cnt()
            ############ FOR PERM IMP ############
            # create the pipeline without refitting the regressor
            model = pipeline.get_root_node().regressor
            #pipeline_fitted = pipeline.fit(self.X_train, self.y_train)
            fitted_model = model.fit(uni_features_train, self.y_train)
            # get permutation importance
            # get a random state using the self.rng to get a number between 1 to 100000
            random_state = self.rng.integers(low=1, high=100000)
            # perm_imp = self.permutation_importance_inhouse(fitted_model, uni_features_test, self.y_val, n_repeats=500, random_state=random_state)
            perm_imp = permutation_importance(fitted_model, uni_features_test, self.y_val, n_repeats=100, random_state=random_state, n_jobs=-1)
            # make a sorted dataframe
            perm_imp_df = pd.DataFrame({'Feature': new_column_names, 'PFI_importance': perm_imp['importances_mean']})
            perm_imp_df = perm_imp_df.sort_values(by='PFI_importance', ascending=False)
            perm_imp_df['Rank'] = range(1, len(perm_imp_df) + 1)
            # Add pipeline details to the PFI DataFrame
            perm_imp_df['Pipeline_No'] = i + 1
            perm_imp_df['Pipeline_R2'] = pipeline_r2
            perm_imp_df['Pipeline_Feature_Count'] = pipeline_feature_count
            perm_imp_df['Pipeline_Selector'] = pipeline.get_selector_node().name
            perm_imp_df['Pipeline_Root'] = pipeline.get_root_node().name
            # # print the perm_imp_df
            # print("Permutation Importance for Pipeline", i, ":", perm_imp_df, flush=True)
            # # save the perm_imp_df to a csv file
            # perm_imp_df.to_csv(self.save_directory + f'perm_imp_pipeline_{i}.csv', index=False)
            # adding the individual pipeline PFI to the all_perm_imp_df
            all_perm_imp_df = pd.concat([all_perm_imp_df, perm_imp_df], ignore_index=True)

        # sort the all_shap_values_df by Pipeline_No and then by shap_value
        all_perm_imp_df = all_perm_imp_df.sort_values(by=['Pipeline_No', 'PFI_importance'], ascending=[True, False])
        # reset the index after sorting
        all_perm_imp_df = all_perm_imp_df.reset_index(drop=True)
        all_perm_imp_df.to_csv(self.save_directory + "all_pipelines_PFI_values.csv", index=False)
        print("All PFI values saved to all_pipelines_PFI_values.csv", flush=True)

        # create overall feature importance for all features seen in all_perm_imp_df by using the formula: 1/mean_rank * percentage of times appeared in a pipeline
        # get the mean rank of each feature
        mean_rank = all_perm_imp_df.groupby('Feature').agg({'Rank': 'mean'}).reset_index()
        # get the count of each feature
        feature_count = all_perm_imp_df.groupby('Feature').agg({'Rank': 'count'}).reset_index()
        # Merge the mean_rank and feature_count dataframes
        mean_rank = mean_rank.merge(feature_count, on='Feature', how='left')
        mean_rank.rename(columns={'Rank_x': 'Mean_Rank', 'Rank_y': 'Feature_Count'}, inplace=True)
        #print("Mean Rank and Feature Count: ", mean_rank, flush=True)

        # Calculate the percentage of times each feature appeared in a pipeline
        total_pipelines = all_perm_imp_df['Pipeline_No'].nunique()
        mean_rank['Appearance_Percentage'] = mean_rank['Feature_Count'] / total_pipelines

        # Calculate the Overall Feature Importance
        mean_rank['Overall_Feature_Importance'] = (1 / mean_rank['Mean_Rank']) * mean_rank['Appearance_Percentage']

        # Sort by Overall_Feature_Importance
        mean_rank = mean_rank.sort_values(by='Overall_Feature_Importance', ascending=False)
        # save the mean_rank to a csv file
        mean_rank.to_csv(self.save_directory + "overall_feature_importance.csv", index=False)


        # plot a bar graph of the top 20 features
        top_20_features = mean_rank.head(20)
        plt.figure(figsize=(10, 6))
        plt.barh(top_20_features['Feature'], top_20_features['Overall_Feature_Importance'], color='skyblue')
        plt.title('Top 20 Features by Overall FI')
        plt.xlabel('Overall Feature Importance')
        plt.ylabel('Feature')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.save_directory + 'top_20_features.png')

        plt.clf()
#####################################################################################################
#
# Evolutionary algorithm class that evolves pipelines.
# We use the NSGA-II algorithm to evolve pipelines.
# Pipelines consist of a set of epistatic interactions, feature selector, and a final regressor.
#
#####################################################################################################

import numpy as np
from typeguard import typechecked
from typing import List, Dict
import pandas as pd
import os
import ray
from .pipeline import Pipeline
from sklearn.model_selection import train_test_split
from .snp_hub import SnpHub
from typing import List, Tuple, Set

from .uni_node import UniNode
from .uni_node import UniDominantNode, UniRecessiveNode, UniHeterosisNode, UniUnderDominantNode, UniSubadditiveNode, UniSuperadditiveNode, UniPAGERNode

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
# population id type
pop_id_t = np.uint16
# diversity score type
div_t = np.float32

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
    unis = {np.str_('dominant'): UniDominantNode,
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
def ray_eval_pipeline(x_train,
                      y_train,
                      x_val,
                      y_val,
                      uni_nodes: uni_node_list_t,
                      selector_node: ScikitNode,
                      ld_node: LDSelector,
                      root_node: ScikitNode,
                      pop_id: np.int16,
                      snp_r2_set: Set) -> Tuple[np.float32, np.uint16, np.int16]:
    # make dictionary to hold the snp r2 scores
    snp_r2_dict = {p[0]: p[1] for p in snp_r2_set}

    # create the pipeline
    steps = []
    # uni nodes into one sklearn union
    steps.append(('snp_union', FeatureUnion([(uni_node.name, uni_node) for uni_node in uni_nodes])))
    # make a list of uni node names
    uni_node_names = [uni_node.get_snp_name() for uni_node in uni_nodes]
    # print("Uni node names: ", uni_node_names, flush=True)
    # add the selector node
    steps.append(('selector', selector_node))

    # fit the pipeline to get the selected features
    pipeline = SklearnPipeline(steps=steps)
    try:
        pipeline_fitted = pipeline.fit(x_train, y_train)
    except Exception as e:
        # Catch all other exceptions and log error with relevant context
        logging.error(f"Exception while fitting model: {e}")
        logging.error(f"selector_node: {selector_node.name}")
        return r2_t(-1.0), feature_cnt_t(0), pop_id

    selected_features = pipeline_fitted.named_steps['selector'].get_feature_names(uni_node_names)
    # print("Selected features: ", selected_features, flush=True)
    x_train_transformed = pipeline_fitted.transform(x_train)
    x_train_transformed_df = pd.DataFrame(x_train_transformed, columns=selected_features)
    if x_train_transformed_df.empty:
        return r2_t(-1.0), feature_cnt_t(0), pop_id

    # x_train original dataframe
    x_train_original_df = pd.DataFrame(x_train, columns=selected_features)

    # # reinitialize the LD node with the selected features, and the other features already in the LD node
    ld_node.fit(x_train_original_df, x_train_transformed_df, y_train, snp_r2_dict)

    # data after LD node
    features_final = ld_node.selected_features_
    # print("Features after LD node: ", features_final, flush=True)
    x_final = pd.DataFrame(x_train_transformed_df[features_final], columns=features_final)
    # print("Shape of x_final: ", x_final.shape, flush=True)

    try:
        # Fit the pipeline with warnings captured as exceptions
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=ConvergenceWarning)

            # create the pipeline
            steps = []
            # uni nodes into one sklearn union
            steps.append(('snp_union', FeatureUnion([(uni_node.name, uni_node) for uni_node in uni_nodes if uni_node.get_snp_name() in features_final])))
            # pass to regressor
            steps.append(('regressor', root_node.regressor))

            # create the pipeline without refitting the regressor
            pipeline = SklearnPipeline(steps=steps)

            pipeline.fit(x_train, y_train)

            # print the number of features seen during the fitting
            # Access the fitted regressor from the pipeline
            # fitted_regressor = pipeline.named_steps['regressor']

            # Check if the fitted regressor has the attribute n_features_in_
            # if hasattr(fitted_regressor, 'n_features_in_'):
            #     features_seen_by_regressor = fitted_regressor.n_features_in_
            #     print("Number of features seen by regressor: ", features_seen_by_regressor, flush=True)
            # else:
            #     print("The regressor does not have the attribute 'n_features_in_'", flush=True)


    except ConvergenceWarning as cw:
        logging.error(f"ConvergenceWarning while fitting model: {cw}")
        logging.error(f"selector_node: {selector_node.name}")
        logging.error(f"selector_node.params: {selector_node.params}")
        logging.error(f"feature_uni_nodes: {len(uni_nodes)}")
        logging.error(f"LD node: {ld_node.name}")
        return r2_t(-1.0), feature_cnt_t(0), pop_id
    except NotFittedError as nfe:
        logging.error(f"NotFittedError occurred: {nfe}")
        return r2_t(-1.0), feature_cnt_t(0), pop_id
    except Exception as e:
        # Catch all other exceptions and log error with relevant context
        logging.error(f"Exception while fitting model: {e}")
        logging.error(f"selector_node: {selector_node.name}")
        logging.error(f"selector_node.params: {selector_node.params}")
        logging.error(f"feature_uni_nodes: {len(uni_nodes)}")
        logging.error(f"Shapes -> X_train: {x_train.shape}, Y_train: {y_train.shape}")
        logging.error(f"LD node: {ld_node.name}")
        return r2_t(-1.0), feature_cnt_t(0), pop_id

    try:
        # print('type of pipeline: ', type(pipeline), flush=True)
        # # print('type of root: ', type(root_node), flush=True)
        # # print('root node name: ', root_node.name, flush=True)
        # print('pipeline: ', pipeline, flush=True)

        r2_score = pipeline.score(x_val, y_val)
        feature_count = len(features_final) # get the number of features after the LD node
    except Exception as e:
        logging.error(f"Error while scoring or getting feature count: {e}")
        return r2_t(-1.0), feature_cnt_t(0), pop_id

    # return the pipeline
    return r2_t(r2_score), feature_cnt_t(feature_count), pop_id

@typechecked # for debugging purposes
class EA:
    def __init__(self,
                 seed: np.uint16,
                 pop_size: np.uint16,
                 uni_cnt_max: np.uint16,
                 uni_cnt_min: np.uint16,
                 cores: int,
                 uni_start_cnt: int = -1,
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
                 num_add_interactions: np.uint16 = np.uint16(10),
                 num_del_interactions: np.uint16 = np.uint16(10),
                 save_directory: str = "") -> None:
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
        num_add_interactions: np.uint16
            Number of interactions to add within a pipeline.
        num_del_interactions: np.uint16
            Number of interactions to delete within a pipeline.
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
        self.num_add_interactions = num_add_interactions
        self.num_del_interactions = num_del_interactions
        self.population = [] # will hold all the pipelines
        self.uni_start_cnt = uni_start_cnt
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
                                        smt_out_out_p=smt_out_out_p,
                                        num_add_interactions=num_add_interactions,
                                        num_del_interactions=num_del_interactions)
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
            offspring = self.process_offspring(offspring)

            # evaluate the offspring
            self.evaluation(offspring)

            # subset the population to only include pipelines with positive r2 scores
            off = []
            for pipeline in offspring:
                if pipeline.get_trait_r2() > 0.0:
                    off.append(pipeline)
            offspring = off

            # must be less than or equal bc of potential negative r2 offspring pipelines
            assert len(offspring) + len(self.population) <= 2 * self.pop_size

            # survival selection
            self.population = self.survival_selection(self.population, offspring)

            # make sure we have the correct number of pipelines
            assert len(self.population) == self.pop_size

            print(f"Time to finish generation: {(time.time() - start_time) / 60} minutes", flush=True)

        # plot the pareto front
        self.plot_pareto_front() # calling the plotting function at the end to get the final pareto plot
        # save the epi_hub to a csv file in the save directory
        self.hubs.save_hubs(self.save_directory+"snp_hub.csv")

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
        combined_pipelines = pop1 + pop2

        # get the fronts and rank
        fronts, _ = nsga.non_dominated_sorting(obj_scores=self.get_pipeline_scores(combined_pipelines, (r2_t(1.0), feature_cnt_t(-1))))

        # get crowding distance for each solution
        crowding_distance = nsga.crowding_distance(self.get_pipeline_scores(combined_pipelines, (r2_t(1.0), feature_cnt_t(1))), np.int32(2))

        # truncate the population to the population size with nsga ii
        survivor_ids = nsga.non_dominated_truncate(fronts, crowding_distance, self.pop_size)
        # make sure that the number of survivors is correct
        assert len(survivor_ids) == self.pop_size

        # combine the population and offspring
        candidates = pop1 + pop2

        # subset the candidates to only include the survivors
        new_pop = []

        for i in survivor_ids:
            # make sure we are within the bounds of the candidates
            assert 0 <= i < len(candidates)
            new_pop.append(candidates[i])

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

            if 0 < self.uni_start_cnt:
                uni_cnt = self.uni_start_cnt
            else:
                # add a random number of snps to the set
                uni_cnt = self.rng.integers(low=self.uni_cnt_min, high=self.uni_cnt_max + 1)

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
        self.evaluate_unseen_snps(unseen_snps)

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
            self.population.append(self.repoduction.generate_random_pipeline(self.rng, good_snps, int(self.seed), self.X_train, self.y_train, self.hubs))

        # make sure we have the correct number of pipelines
        assert len(self.population) == self.pop_size

        # evaluate the initial population
        self.evaluation(self.population)

        # subset the population to only include pipelines with positive r2 scores
        pop = []
        for pipeline in self.population:
            # print("Pipeline r2 score: ", pipeline.get_trait_r2())
            # print("Pipeline feature count: ", pipeline.get_trait_feature_cnt())
            # print("SNPs selected by the LD node: ", len(pipeline.get_ld_node().selected_features_))
            if pipeline.get_trait_r2() > 0.0:
                pop.append(pipeline)
        self.population = pop

        return

    # evaluate all unevaluated snps and update
    def evaluate_unseen_snps(self, unseen_snps: Set) -> None:
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
            self.hubs.update_snp_hub(snp_name, r2, type)

    # remove bad snps (r2 < 0)
    def remove_bad_snps(self, snps: Set) -> Set:
        """
        Function to remove bad snps with r2<0 for a given set of snps

        Parameters:
        snps: Set of snps
        """
        good_snps = set()
        for snp_name in snps:
            # check if r2 is positive
            if self.hubs.get_uni_res(snp_name) > np.float32(0.0):
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
    def evaluation(self, pop: List[Pipeline]) -> None:
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
        for i, pipeline in enumerate(pop):
            # add ray job
            # uni_snps, snps_pos = [],[]
            # self.construct_uni_nodes(pipeline.get_uni_snps())


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

        # process results as they come in
        while len(ray_jobs) > 0:
            finished, ray_jobs = ray.wait(ray_jobs)
            r2, feature_count, pop_id = ray.get(finished)[0]
            # update the pipeline
            pop[pop_id].set_traits([r2, feature_count])
        return

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

            if uni_type == np.str_('dominant'):
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
    def process_offspring(self, pipelines: List[Pipeline]) -> List[Pipeline]:
        # get unseen interactions
        unseen_snps = self.get_unseen_univariates(pipelines)

        # evaluate all unseen interactions
        self.evaluate_unseen_snps(unseen_snps)

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

    # Create a object of Poster class to check the post analyis of the pipelines
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
        poster = Poster(self.X_train_id, self.y_train_id, self.X_val_id, self.y_val_id, hub=self.hubs)

        # Initialize an empty DataFrame to save all shap_values_df
        all_shap_values_df = pd.DataFrame()

        # print the pipelines in the population
        ray_jobs = []
        for i, pipeline in enumerate(pareto_front): # think this as pareto front pipelines
            uni_nodes = self.construct_uni_nodes(pipeline.get_uni_snps())

            # Get R2 and Feature Count for this specific pipeline
            pipeline_r2 = pipeline.get_trait_r2()
            pipeline_feature_count = pipeline.get_trait_feature_cnt()
            results_refs = poster.run_poster(pipeline,  uni_nodes, self.X_train_id, self.y_train_id, self.X_val_id, self.y_val_id, id = i)
            ray_jobs.append((results_refs, pipeline_r2, pipeline_feature_count))  # Save the refs along with R2 and Feature Count

        assert len(ray_jobs) == len(pareto_front)

        uni_feature_datasets = []

        # getting the results from ray
        while len(ray_jobs) > 0:
            # Wait for the next job to complete (extracting the refs only)
            finished_refs, remaining_jobs = ray.wait([job[0] for job in ray_jobs])  # Wait for the first job to finish

            # Find the corresponding job in ray_jobs
            for i, (ref, r2_value, feature_count) in enumerate(ray_jobs):
                if ref == finished_refs[0]:  # Match the finished job reference
                    # Get the results of the finished job
                    uni_feature_dataset, shap_values_df, selector_name, root_name,  pipeline_id = ray.get(ref)

                    # Add pipeline details to the SHAP DataFrame
                    shap_values_df['Pipeline_No'] = pipeline_id + 1
                    shap_values_df['Pipeline_R2'] = r2_value
                    shap_values_df['Pipeline_Feature_Count'] = feature_count
                    shap_values_df['Pipeline_Selector'] = selector_name
                    shap_values_df['Pipeline_Root'] = root_name

                    # Store the shap_values_df in the list for later processing or concatenation
                    all_shap_values_df = pd.concat([all_shap_values_df, shap_values_df], ignore_index=True)

                    # print the shape of the epi feature dataset with the pipeline number
                    print(f"Pipeline {pipeline_id + 1} Uni Feature Dataset Shape:", uni_feature_dataset.shape, flush=True)

                    # Add the uni_feature_dataset DataFrame to the list
                    uni_feature_datasets.append(uni_feature_dataset)

                    # Remove the processed job from ray_jobs
                    ray_jobs.pop(i)
                    break

        # sort the all_shap_values_df by Pipeline_No and then by shap_value
        all_shap_values_df = all_shap_values_df.sort_values(by=['Pipeline_No', 'shap_value'], ascending=[True, False])
        # reset the index after sorting
        all_shap_values_df = all_shap_values_df.reset_index(drop=True)
        all_shap_values_df.to_csv(self.save_directory + "combined_shap_values.csv", index=False)
        print("All SHAP values saved to combined_shap_values.csv", flush=True)

        # create the average SHAP values for each SNP
        # add a column added OVERALL_FEATURE_IMP, which will be the multiplication value of shap_value and R2
        all_shap_values_df['OVERALL_FEATURE_IMP'] = all_shap_values_df['shap_value'] * all_shap_values_df['Pipeline_R2']

        # for the each unique feature in feature column add up all the OVERALL_FEATURE_IMP values
        all_shap_values_df = all_shap_values_df.groupby('feature').agg({'OVERALL_FEATURE_IMP': 'sum'}).reset_index()

        # sort the dataframe by the OVERALL_FEATURE_IMP
        all_shap_values_df = all_shap_values_df.sort_values(by='OVERALL_FEATURE_IMP', ascending=False)

        # save the all_shap_values_df to a csv file
        all_shap_values_df.to_csv(self.save_directory + "avg_shap_values.csv", index=False)

        # plot a bar graph of the top 20 features
        top_20_features = all_shap_values_df.head(20)
        plt.figure(figsize=(10, 6))
        plt.barh(top_20_features['feature'], top_20_features['OVERALL_FEATURE_IMP'], color='skyblue')
        plt.title('Top 20 Features by Overall FI')
        plt.xlabel('Overall Feature Importance')
        plt.ylabel('Feature')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.save_directory + 'top_20_features.png')
#####################################################################################################
#
# Evolutionary algorithm base class that evolves pipelines.
# We use the NSGA-II algorithm to evolve pipelines.
#
#####################################################################################################

import numpy as np
from typeguard import typechecked
from typing import List, Dict, Set
import pandas as pd
import os
import ray
from typing import List, Tuple
import numpy.typing as npt
from . import nsga_tool as nsga
import matplotlib.pyplot as plt
import datatable as dt
from datatable import f
import warnings
from .types import (float32_t, int16_t, prob_t, int32_t, snp_t, uint16_t)
from abc import ABC, abstractmethod
from .pipeline import Pipeline

# to not show runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

@typechecked
class EA(ABC):
    def __init__(self,
                 seed: int,
                 pop_size: uint16_t,
                 branch_max: uint16_t,
                 branch_min: uint16_t,
                 cores: int,
                 mut_prob: prob_t = prob_t(.5), # probability of mutation
                 cross_prob: prob_t = prob_t(.5), # probability of crossover
                 mut_selector_p: prob_t = prob_t(.5), # probability of mutating the feature selector
                 mut_ld_p: prob_t = prob_t(.5), # probability of mutating the ld pruner
                 mut_root_p: prob_t = prob_t(.5), # probability of mutating the root node (regressor or classifier)
                 mut_ran_p: prob_t = prob_t(.5), # probability of random mutation
                 mut_smt_p: prob_t = prob_t(.5), # probability of smart mutation
                 m_in_win_p: prob_t = prob_t(.33), # probability for smart in window mutation
                 m_out_win_p: prob_t = prob_t(.33), # probability for smart out window mutation
                 m_out_chr_p: prob_t = prob_t(.33), # probability for smart out of chromosome mutation
                 save_directory: str = "",
                 window_distance: int32_t = int32_t(1000000),
                 branch_explainability_threshold: float32_t = float32_t(0.0),
                 ld_flag: bool = True,
                 ) -> None:
        """
        Main class for the evolutionary algorithm.

        Parameters:
        seed: int32_t
            Seed for the random number generator.
        pop_size: int32_t
            Population size.
        branch_max: int32_t
            Maximum number of branch nodes.
        branch_min: int32_t
            Minimum number of branch nodes.
        cores: int
            Number of cores to use for parallel processing.
        mut_prob: prob_t
            Probability for mutation occurring.
        cross_prob: prob_t
            Probability for crossover occurring.
        mut_selector_p: prob_t
            Probability for mutating the feature selector.
        mut_ld_p: prob_t
            Probability for mutating the LD pruner.
        mut_root_p: prob_t
            Probability for mutating the root node (regressor or classifier).
        mut_ran_p: prob_t
            Probability for random mutation.
        mut_smt_p: prob_t
            Probability for smart mutation.
        m_in_win_p: prob_t
            Probability for smart in window mutation.
        m_out_win_p: prob_t
            Probability for smart out window mutation.
        m_out_chr_p: prob_t
            Probability for smart out of chromosome mutation.
        save_directory: str
            Directory to save the results.
        branch_explainability_threshold: float32_t
            Threshold for branch explainability for a branch to be considered important.
        ld_flag: bool
            Flag to indicate whether to use LD pruning or not.
        """
        # initial population
        self.population: List[Pipeline] = []

        # base arguments to use across derived classes
        self.seed = seed

        assert 0 < pop_size
        self.pop_size = pop_size

        assert 0 <= branch_min <= branch_max
        self.branch_max = branch_max
        self.branch_min = branch_min

        assert os.path.isdir(save_directory), "Save directory does not exist."
        self.save_directory = save_directory

        assert 0.0 <= mut_prob <= 1.0, "Mutation probability must be between 0 and 1."
        self.mut_prob = mut_prob

        assert 0.0 <= cross_prob <= 1.0, "Crossover probability must be between 0 and 1."
        self.cross_prob = cross_prob

        assert 0.0 <= mut_selector_p <= 1.0, "mut_selector_p must be between 0 and 1."
        self.mut_selector_p = mut_selector_p

        assert 0.0 <= mut_ld_p <= 1.0, "mut_ld_p must be between 0 and 1."
        self.mut_ld_p = mut_ld_p

        assert 0.0 <= mut_root_p <= 1.0, "mut_root_p must be between 0 and 1."
        self.mut_root_p = mut_root_p

        assert 0.0 <= mut_ran_p <= 1.0, "mut_ran_p must be between 0 and 1."
        self.mut_ran_p = mut_ran_p

        assert 0.0 <= mut_smt_p <= 1.0, "mut_smt_p must be between 0 and 1."
        self.mut_smt_p = mut_smt_p

        assert 0.0 <= m_in_win_p <= 1.0, "m_in_win_p must be between 0 and 1."
        self.m_in_win_p = m_in_win_p

        assert 0.0 <= m_out_win_p <= 1.0, "m_out_win_p must be between 0 and 1."
        self.m_out_win_p = m_out_win_p

        assert 0.0 <= m_out_chr_p <= 1.0, "m_out_chr_p must be between 0 and 1."
        self.m_out_chr_p = m_out_chr_p

        assert 0 <= window_distance, "window_distance must be non-negative."
        self.window_distance = window_distance

        assert 0.0 <= branch_explainability_threshold <= 1.0, "branch_explainability_threshold must be between 0 and 1."
        self.branch_explainability_threshold = branch_explainability_threshold

        assert isinstance(ld_flag, bool), "ld_flag must be a boolean."
        self.ld_flag = ld_flag

        # random number generator to be passed to all other stochastic functions
        self.rng = np.random.default_rng(seed)

        # initialize ray
        ray.init(num_cpus=cores, include_dashboard=True)
        print(flush=True)

    # data splitter: will return indices for train/val/test splits
    def split_dataset_indices(self,
                              n_samples: int,
                              train_ratio: float = 0.7,
                              val_ratio: float = 0.15,          # train idx,  val idx,    test idx
                              test_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns disjoint index arrays for train/val/test that index into the ORIGINAL dataset.
        If y is provided, splits are stratified. If groups is provided (and y is None),
        a grouped split is used for the first split (test) and then for train/val.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1."

        print("Splitting data into train/val/test setspass", flush=True)

        # shuffle indices
        idx = np.arange(n_samples)
        self.rng.shuffle(idx)

        # how many samples for each split
        train_size = int(train_ratio * n_samples)
        val_size = int(val_ratio * n_samples)
        test_size = int(test_ratio * n_samples)
        remaining = n_samples - (train_size + val_size + test_size)
        assert 0 <= remaining, "Remaining samples must be greater than 0."

        # randomly pick which split gets the remaining samples, incrementally add one sample to that split until none are left
        for _ in range(remaining):
            pick = self.rng.integers(0, 3)
            if pick == 0:
                train_size += 1
            elif pick == 1:
                val_size += 1
            else:
                test_size += 1

        assert train_size + val_size + test_size == n_samples, "Sizes do not add up."
        print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}", flush=True)
        print(f'Total samples in each split: {train_size + val_size + test_size}', flush=True)

        # sample indices for training split
        train_idx = idx[:train_size]
        val_idx = idx[train_size:train_size + val_size]
        test_idx = idx[train_size + val_size:]

        assert len(train_idx) == train_size, "Train size does not match."
        assert len(val_idx) == val_size, "Val size does not match."
        assert len(test_idx) == test_size, "Test size does not match."

        assert len(set(train_idx).intersection(set(val_idx))) == 0, "Train and Val sets are not disjoint."
        assert len(set(train_idx).intersection(set(test_idx))) == 0, "Train and Test sets are not disjoint."
        assert len(set(val_idx).intersection(set(test_idx))) == 0, "Val and Test sets are not disjoint."

        assert len(train_idx) + len(val_idx) + len(test_idx) == n_samples, "Sizes do not add up."

        assert 0 <= min(train_idx) and max(train_idx) < n_samples, "Train indices are out of bounds."
        assert 0 <= min(val_idx) and max(val_idx) < n_samples, "Val indices are out of bounds."
        assert 0 <= min(test_idx) and max(test_idx) < n_samples, "Test indices are out of bounds."

        return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)

    # function to generate k-folds for a list of indices
    def make_cv_splits_for_subset(self,
                                  subset_idx: np.ndarray, # fold           train    validation
                                  n_splits: int = 3) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Build CV folds on a SUBSET (e.g., training) but RETURN GLOBAL indices.
        If y/groups are provided, they should be arrays aligned to the ORIGINAL dataset.
        """
        assert n_splits >= 2, "n_splits must be at least 2."
        assert subset_idx.ndim == 1, "subset_idx must be 1-dimensional."
        assert np.all(subset_idx >= 0), "subset_idx must be non-negative."

        # shuffle subset_idx
        self.rng.shuffle(subset_idx)
        # percentage for validation set
        cv_proportion = 1.0 / float(n_splits)
        # number of samples in the subset
        n_samples = len(subset_idx)

        # how big will each validation set be?
        cv_sizes = [int(cv_proportion * n_samples) for _ in range(n_splits)]
        remaining = n_samples - sum(cv_sizes)
        assert 0 <= remaining < n_splits, "Remaining samples must be between 0 and n_splits-1 bc math."

        # randomly pick which splits from cv_sizes gets the remaining samples, incrementally add one sample to that split until none are left
        cv_size_idx = list(range(n_splits))
        self.rng.shuffle(cv_size_idx)
        for i in range(remaining):
            cv_sizes[cv_size_idx[i]] += 1

        # dictionary to hold all splits {'fold_i': (train_idx, val_idx)}
        splits = {}
        for fold in range(n_splits):
            # Get train/val indices for this fold
            val_start = sum(cv_sizes[:fold])
            val_end = val_start + cv_sizes[fold]
            valid_idx = subset_idx[val_start:val_end]
            train_idx = np.concatenate([subset_idx[:val_start], subset_idx[val_end:]])

            # save split info
            splits[f'fold_{fold}'] = (np.sort(train_idx), np.sort(valid_idx))

            assert len(train_idx) + len(valid_idx) == n_samples, "Sizes do not add up."
            assert len(set(train_idx).intersection(set(valid_idx))) == 0, "Train and Val sets are not disjoint."

        # make sure the union of all validation sets is the entire subset
        assert len(set().union(*[set(splits[f'fold_{i}'][1]) for i in range(n_splits)])) == n_samples, "Validation sets are not disjoint."
        # make sure the union of all training sets is the entire subset
        assert len(set().union(*[set(splits[f'fold_{i}'][0]) for i in range(n_splits)])) == n_samples, "Training sets are not disjoint."
        # return new set of indeces
        return splits

    # data loader
    # todo: add seperate seeds for data shuffling, may be different from evolver seed
    def data_loader(self,
                    path: str,
                    target_label: str = "y",
                    train_split: float = 0.50,
                    valid_split: float = 0.25,
                    test_split: float = 0.25,
                    k: int = 3) -> None:
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
        train_split: float
            Proportion of data to be used for training.
        valid_split: float
            Proportion of data to be used for validation.
        test_split: float
            Proportion of data to be used for testing.
        k: int
            Number of folds for cross-validation.
        """

        print('Loading datapass', flush=True)
        print('Path:', path, flush=True)

        self.k = k

        # check if the path is valid
        if os.path.isfile(path) == False:
            print(f"Error: The file {path} does not exist.", flush=True)
            exit(-1)

        # load the entire dataset using datatable
        data_dt = dt.fread(path)

        # check if the column names contain 'chr' and remove it
        if any("chr" in name for name in data_dt.names):
            print("Detected 'chr' in column names. Removing 'chr' from column namespass", flush=True)
            # remove 'chr' from column names
            data_dt.names = [name.replace("chr", "") for name in data_dt.names]

        # check if the target label is valid
        if target_label not in data_dt.names:
            print(f"Error: The target label '{target_label}' is not in the dataset columns.", flush=True)
            exit(-1)

        # Extract target column and SNP columns
        self.target_label = snp_t(target_label)
        self.snp_labels = [snp_t(name) for name in data_dt.names if name != target_label]

        # Convert to pandas DataFrame for compatibility with downstream logic
        all_x = data_dt[:, [f[name] for name in self.snp_labels]].to_pandas()
        all_y = data_dt[:, f[self.target_label]].to_pandas().values.ravel()

        # check if the data was loaded correctly
        all_x, all_y = self.check_dataset(all_x, all_y)
        print('X_data.shape:', all_x.shape, flush=True)
        print('y_data.shape:', all_y.shape, flush=True)
        print(flush=True)

        # checking the additive encoding of the data, if 0,1,2 detected, we will transform it to dosage encoding (0,0.5,1)
        # Check unique values across all SNP columns
        unique_vals = pd.unique(all_x.values.ravel())

        # Convert to a set for comparison
        unique_vals_set = set(unique_vals)

        # Check if the data is in {0, 1, 2} encoding
        if unique_vals_set.issubset({0, 1, 2}):
            print("Detected additive encoding (0,1,2). Transforming to dosage encoding (0,0.5,1)pass", flush=True)
            all_x = all_x.replace(1, 0.5)
            all_x = all_x.replace(2, 1)
        else:
            print("Detected additive encoding (0,0.5,1). No transformation applied.", flush=True)

        self.all_x = all_x
        self.all_y = np.array(all_y, dtype=float32_t)

        # print the data after changing the encoding
        print("Genotype data: ", all_x, flush=True)

        # get number of samples from all_x
        n_samples = all_x.shape[0]
        print("Number of samples:", n_samples, flush=True)

        # get the indices for train/val/test splits
        self.train_idx, self.val_idx, self.test_idx = self.split_dataset_indices(n_samples=n_samples,
                                                                                 train_ratio=train_split,
                                                                                 val_ratio=valid_split,
                                                                                 test_ratio=test_split)
        # save the indices for ray calls
        self.train_idx_ray = ray.put(self.train_idx)
        self.val_idx_ray = ray.put(self.val_idx)
        self.test_idx_ray = ray.put(self.test_idx)

        # k-fold cross validation on the training set
        self.train_fold_dict = self.make_cv_splits_for_subset(subset_idx=self.train_idx, n_splits=k)
        print("Data partitioning complete.", flush=True)

        self.train_fold_dict_ray = {}
        for fold, (train_index, val_index) in self.train_fold_dict.items():
            self.train_fold_dict_ray[fold] = {'train_idx': ray.put(train_index),'val_idx': ray.put(val_index)}
            print(f"{fold} => Train size: {len(train_index)}, Val size: {len(val_index)}", flush=True)
        print(flush=True)
        return

    # data checker to check for validity of dataset
    def check_dataset(self, features, target):
        """
        Check if a dataset has a valid feature set and labels. If there are missing values, we will impute them with the mode of the column.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        target: array-like {n_samples} or None
            List of class labels for prediction
        sample_weight: array-like {n_samples} (optional)
            List of weights indicating relative importance
        """

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

    @abstractmethod
    def initialize_hubs(self) -> None:
        """
        Initialize the hubs needed for the run.
        These specific hub classes must be implemented in the derived class folders.
        """

    # evolve a population of pipelines for 'gens' generations
    # must specify how to initialize the population and implement NSGA steps
    @abstractmethod
    def evolve(self, gens: uint16_t) -> None:
        """
        Evolve the population of pipelines for a given number of generations.
        Should follow the NSGA-II algorithm steps:
        1. Initialize the population.
        2. Evaluate the population.
        3. While generation < gens:
            a. Select parents.
            b. Generate offspring through variation (crossover and mutation).
            c. Evaluate offspring.
            d. Select survivors to form the new population.
            e. Go to step 3.

        Args:
            gens (int): Number of generations to evolve the population.
        """
        pass

    def collapse_population_to_front_0(self):
        front_0 = []
        for i in nsga.front_zero(obj_scores=self.get_pipeline_scores(self.population, weights=(float32_t(1.0), int32_t(-1)))):
            front_0.append(self.population[i])
        self.population = front_0  # keep only the front 0 pipelines

        # group all pipelines in self.population that have the same complexity
        complexity_groups = {}
        for i, pipeline in enumerate(self.population):
            complexity = pipeline.get_trait_feature_cnt()
            if complexity not in complexity_groups:
                complexity_groups[complexity] = []
            complexity_groups[complexity].append(i)

        # for each complexity group, randomly sample one representative to keep
        winner_ids = []
        for group in complexity_groups.values():
            winner_ids.append(self.rng.choice(group))

        self.population = [self.population[i] for i in winner_ids]
        return

    def save_total_runtime(self, total_runtime: float) -> None:
        """
        Function to save the total runtime in minutes of the algorithm to a file.

        Parameters:
        total_runtime: float
            Total runtime of the algorithm.
        """
        with open(os.path.join(self.save_directory, 'total_runtime.csv'), 'w') as f:
            f.write(str(total_runtime))

    # get list of pipeline scores (r2, complexity) by position
    def get_pipeline_scores(self, pipelines: List[Pipeline], weights: Tuple[float32_t, int32_t]) -> npt.NDArray:
        """
        Function to get the pipeline scores (r2, complexity) by position.
        Will also apply weights to the scores, so that we can use NSGA-II to get the pareto front.
        """
        scores = np.empty(len(pipelines), dtype=object)
        for i, pipeline in enumerate(pipelines):
            scores[i] = (float32_t(pipeline.get_trait_r2() * weights[0]), int32_t(pipeline.get_trait_feature_cnt() * weights[1]))

        return scores

    # survival selection
    def survival_selection(self, offspring_pipelines: List[Pipeline]) -> List[Pipeline]:
        """
        Function to select the survivors from the offspring pipelines provided.

        Parameters:
        offspring_pipelines: List[Pipeline]
            List of offspring pipelines to select survivors from.

        Returns:
        List[Pipeline]: List of survivor pipelines.
        """
        # make sure all population scores are positive
        assert all(pipeline.get_trait_r2() > 0.0 for pipeline in offspring_pipelines), "All offspring r2 scores must be positive."
        assert all(pipeline.get_trait_feature_cnt() > 0 for pipeline in offspring_pipelines), "All population complexity scores must be positive."

        # combine both the population and offspring lists into one
        pipelines_original = offspring_pipelines

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
        fronts, _ = nsga.non_dominated_sorting(obj_scores=self.get_pipeline_scores(non_dup_pipelines, (float32_t(1.0), int32_t(-1))))

        # get crowding distance for each solution
        crowding_distance = nsga.crowding_distance(self.get_pipeline_scores(non_dup_pipelines, (float32_t(1.0), int32_t(1))), fronts)

        # truncate the population to the population size with nsga ii
        survivor_ids = nsga.non_dominated_truncate(fronts, crowding_distance, int16_t(self.pop_size))
        # make sure that the number of survivors is correct
        assert len(survivor_ids) <= self.pop_size

        # subset the candidates to only include the survivors
        new_pop = []

        for i in survivor_ids:
            # make sure we are within the bounds of the candidates
            assert 0 <= i < len(non_dup_pipelines)
            new_pop.append(non_dup_pipelines[i])

        return new_pop

    @abstractmethod
    def initialize_population(self) -> None:
        """
        Function to initialize the population of pipelines for self.population.
        Size of self.population must be self.pop_size.
        """
        pass

    @abstractmethod
    def remove_bad_pipelines(self, pipelines: List[Pipeline]) -> List[Pipeline]:
        """
        Function to remove pipelines that fail to meet certain criteria.

        Args:
            pipelines (List[Pipeline]): List of pipelines to evaluate.

        Returns:
            List[Pipeline]: List of pipelines that passed the evaluation.
        """
        pass

    @abstractmethod
    def evaluation(self, pipelines: List[Pipeline], gen_info: int16_t) -> List[Pipeline]:
        """
        Function to evaluate entire pipelines.
        All of this should be done in asyncronous parallel jobs for maximum efficiency.

        Parameters:
        pipelines: List[Pipeline]
            List of pipelines to evaluate.
        gen_info: int16_t
            Generation number for logging purposes.

        Returns:
        List[Pipeline]: List of evaluated pipelines (pipelines updated with evaluation results).
        """
        pass

    @abstractmethod
    # evaluate all unevaluated branches and update
    def evaluate_unseen_branches(self, unseen_branches: Set, gen_seen: int16_t) -> None:
        """
        Function to evaluate all unseen branches and add their best R2 and Encoder type to the Hub.
        All of this should be done in asyncronous parallel jobs.
        We update the Hub with the results as they come in.

        Parameters:
        unseen_branches: Set
            Unseen branches in a set to evaluate.
        """
        pass

    # parent selection
    def parent_selection(self, parent_cnt: uint16_t) -> List[uint16_t]:
        """
        Function to return a List of parent ids based on Pareto dominance.
        Parent selection only considers pipelines in the current population.
        Size of list must be parent_cnt.

        Parameters:
        parent_cnt: uint16_t
            Number of parents to select.

        Returns:
        List[uint16_t]: List of parent ids
        """

        # will hold the parent ids
        parent_ids = []

        # get the fronts and rank
        fronts, ranks = nsga.non_dominated_sorting(obj_scores=self.get_pipeline_scores(self.population, (float32_t(1.0), int32_t(-1))))
        # make sure that the number of fronts is correct
        assert sum([len(f) for f in fronts]) == len(ranks)

        # get crowding distance for each solution
        crowding_distance = nsga.crowding_distance(self.get_pipeline_scores(self.population, weights=(float32_t(1.0), int32_t(1))), fronts)

        # get parent_cnt number of parents
        for _ in range(parent_cnt):
            parent = nsga.non_dominated_binary_tournament(rng=self.rng, ranks=ranks, distances=crowding_distance)
            # make sure we are within the bounds of the candidates
            assert 0 <= parent < len(self.population)
            parent_ids.append(parent)
        # make sure that the number of parents is correct
        assert len(parent_ids) == parent_cnt

        return parent_ids

    @abstractmethod
    def process_offspring(self, pipelines: List[Pipeline], gen_info: int16_t) -> List[Pipeline]:
        """
        Function to process the offspring pipelines after they have been generated.
        This should include branch set updates, pipeline evaluation, removal of bad pipelines, etc.

        Args:
            pipelines (List[Pipeline]): List of offspring pipelines to process.
            gen_info (int16_t): Generation information for logging purposes.

        Returns:
            List[Pipeline]: List of processed pipelines.
        """
        pass

    # record the pipeline r2 and complexity scores for the pareto front from the final population
    def record_final_pareto_front(self) -> None:
        """
        Function to record the final pareto front from the population with complexity and r2 scores.
        Must run collapse_population_to_front_0 before this function to ensure that the population is reduced to only front 0 pipelines.
        """

        # get all scores from the current population
        # should only be front 0 pipelines
        pop_scores = self.get_pipeline_scores(self.population, weights=(float32_t(1.0), int32_t(1)))

        # sort front by feature count
        pareto_front = sorted(pop_scores, key=lambda x: x[1])

        # save the pareto front to a csv file and enumerate the pipelines
        pareto_front_df = pd.DataFrame(pareto_front, columns=['R2', 'Feature Count'])
        pareto_front_df.to_csv(self.save_directory + 'final_pareto_front.csv', index=False)
        return

    # plot the current pareto front from the population with complexity and r2 scores
    def plot_pareto_front(self) -> None:
        """
        Function to plot the current pareto front from the population with complexity and r2 scores.
        Must run collapse_population_to_front_0 before this function to ensure that the population is reduced to only front 0 pipelines.
        """

        # get all scores from the current population
        pop_scores = self.get_pipeline_scores(self.population, weights=(float32_t(1.0), int32_t(1)))

        # sort front by feature count
        pareto_front = sorted(pop_scores, key=lambda x: x[1])

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

    @abstractmethod
    def post_analysis_with_good_snps(self) -> None:
        """
        Function to perform post analysis of the pipelines.
        Must include
        """

    @abstractmethod
    def final_pipeline_test(self, pipeline_data: Dict, file_name: str, train_r2: float32_t, validation_r2: float32_t, size: int16_t) -> None:
        """
        Function to perform the final validation on the test dataset.
        Combine the training and validation datasets to fit a linear regression model.
        Then, evaluate the model on the test dataset and save the results to a csv file.
        Also, perform permutation importance on the test dataset and save the results to a csv file.
        """
        pass
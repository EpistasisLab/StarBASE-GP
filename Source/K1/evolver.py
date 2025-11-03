
# import base EA class and types
from ..Base.evovler import EA
from ..Base.types import (float32_t, int16_t, prob_t, int32_t, snp_t, uint16_t)
from ..Base.pipeline import Pipeline
from ..Base.utils import snp_chrm_pos
from ..Base import nsga_tool as nsga
from ..Base.selectors import OLSRegressor

# import K1 specific classes
from .snp_hub import K1_Hub
from . import ray_utils
from .reproduction import K1_Reproduction

# import other necessary libraries
import numpy as np
from typeguard import typechecked
from typing import List, Dict, Set, Tuple
import pandas as pd
import os
import ray
from typing import List, Tuple
import numpy.typing as npt
import matplotlib.pyplot as plt
import time
from sklearn.metrics import r2_score
import statsmodels.api as sm

@typechecked
class K1_Evolver(EA):
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
                 regression: bool = True
                 ) -> None:
        """
        K1 Evolver class that extends the EA base class.
        """

        # pass all variables to the EA base class
        super().__init__(seed=seed,
                         pop_size=pop_size,
                         branch_max=branch_max,
                         branch_min=branch_min,
                         cores=cores,
                         mut_prob=mut_prob,
                         cross_prob=cross_prob,
                         mut_selector_p=mut_selector_p,
                         mut_ld_p=mut_ld_p,
                         mut_root_p=mut_root_p,
                         mut_ran_p=mut_ran_p,
                         mut_smt_p=mut_smt_p,
                         m_in_win_p=m_in_win_p,
                         m_out_win_p=m_out_win_p,
                         m_out_chr_p=m_out_chr_p,
                         save_directory=save_directory,
                         window_distance=window_distance,
                         branch_explainability_threshold=branch_explainability_threshold,
                         ld_flag=ld_flag)
        self.regression = regression

        # todo: add pager encoder once done
        self.encoder_types = [ snp_t('additive'), snp_t('dominant'), snp_t('recessive'),
                              snp_t('heterosis'), snp_t('underdominant'), snp_t('overdominant'),
                              snp_t('subadditive'), snp_t('superadditive'),
                              snp_t('pager')
                              ]
        # initialize reproduction class
        self.reproduction = K1_Reproduction(branch_max=self.branch_max,
                                           branch_min=self.branch_min,
                                           mut_prob=self.mut_prob,
                                           cross_prob=self.cross_prob,
                                           mut_selector_p=self.mut_selector_p,
                                           mut_ld_p=self.mut_ld_p,
                                           mut_regressor_p=self.mut_root_p,
                                           mut_ran_p=self.mut_ran_p,
                                           mut_smt_p=self.mut_smt_p,
                                           m_in_win_p=self.m_in_win_p,
                                           m_out_win_p=self.m_out_win_p,
                                           m_out_chr_p=self.m_out_chr_p,
                                           window_distance=self.window_distance)

        return

    def initialize_hubs(self) -> None:
        """
        Initialize the hubs needed for the run.
        These specific hub classes must be implemented in the derived class folders.
        """
        # dictionary of feature names and ray_ids for each specific column put into ray
        feature_ray_ids = {}
        for feature in self.snp_labels:
            feature_ray_ids[feature] = ray.put(self.all_x[feature].to_numpy(dtype=float32_t))
        self.all_y_ray_id = ray.put(self.all_y)

        # initialize the hubs
        self.hub = K1_Hub(snp_list=self.snp_labels, snps_ray_ids=feature_ray_ids)
        return

    def evolve(self, gens: uint16_t) -> None:
        # list to store the generation details - front zero size, still consider snp set size, number of snps pruned
        generation_details = []
        # start the timer for the entire process
        total_gp_run = time.time()
        # create the initial population
        print('Initializing population...', flush=True)
        start_time = time.time()
        self.initialize_population()
        print(f"Population initialized in {(time.time() - start_time) / 60 / 60} hours", flush=True)
        print('Entering evolutionary proccess.\n', flush=True)

        # run the algorithm for the specified number of generations
        for g in range(gens):
            print('Generation:', g, flush=True)

            # how many pipelines are in the population
            print('Population size:', len(self.population), flush=True)
            assert(0 < len(self.population) <= self.pop_size)

            # get the size of the front 0 after each generation
            count = 0
            if len(self.population) >= 2:
                _, rank = nsga.non_dominated_sorting(obj_scores=self.get_pipeline_scores(self.population, weights=(float32_t(1.0), int32_t(-1))))
                count = 0
                for r in rank:
                    if r == 0:
                        count += 1
            else:
                count = 1
            print('Size of Pareto Front:', count, flush=True)
            self.hub.seen_snps_proportion()  # count the number of unseen snps after each generation

            # Initialize generation stats dictionary (will be updated after evaluation)
            gen_stats = {
                'generation': g,
                'front_zero_size': count,
                'consideration_set_size': self.hub.consideration_hub_size()
            }

            start_time = time.time()

            # get order of mutation/crossover to do with the extra offspring
            var_order = None
            parent_cnt = None
            parent_ids = None
            if len(self.population) == 1:
                var_order, parent_cnt = [snp_t('m')] * uint16_t(2*self.pop_size), uint16_t(2*self.pop_size)
                parent_ids = [uint16_t(0)] * parent_cnt
            else:
                var_order, parent_cnt = self.reproduction.variation_order(self.rng, uint16_t(2*self.pop_size))
                parent_ids = self.parent_selection(parent_cnt)

            # generate offspring
            offspring = self.reproduction.produce_offspring(rng = self.rng,
                                                           hub = self.hub,
                                                           offspring_cnt = uint16_t(2*self.pop_size),
                                                           parent_ids = parent_ids,
                                                           population = self.population,
                                                           order = var_order)
            # make sure we have the correct number of competing solutions
            assert len(offspring) + len(self.population) <= 3 * self.pop_size

            # process offspring: evaluation interactions and remove bad interactions
            offspring = self.process_offspring(offspring, int16_t(g))

            # evaluate the offspring
            print('Evaluating offspring pipelines...', flush=True)
            offspring, eval_stats = self.evaluation(offspring, int16_t(g))

            # must be less than or equal because of potential negative r2 offspring pipelines
            assert (0 < len(offspring) + len(self.population) <= 3 * self.pop_size)

            # survival selection
            self.population = self.survival_selection(offspring)

            # make sure we have the correct number of pipelines
            assert len(self.population) <= self.pop_size

            # Calculate total generation time
            gen_time = (time.time() - start_time) / 60  # in minutes
            print(f"Time to finish generation: {gen_time} minutes", flush=True)
            
            # Add evaluation stats and timing to generation details
            gen_stats.update({
                'total_time_mins': gen_time,
                'fs_only_count': eval_stats['fs_only_count'],
                'ld_fs_count': eval_stats['ld_fs_count'],
                'sequential_selector_count': eval_stats['sequential_selector_count'],
                'fs_ld_time_all_pipelines_mins': eval_stats['fs_ld_time_all_pipelines']
            })
            generation_details.append(gen_stats)
            
            print('-'*50, flush=True)

        # prints for the end of a run and the final population
        print('Final run/population details')
        print('Final population size:', len(self.population), flush=True)

        # save Pareto front details and plot Pareto front
        self.save_and_plot_pareto_front()
        # save the hubs details
        self.hub.save_hubs(self.save_directory)
        
        # save generation details to CSV
        if len(generation_details) > 0:
            gen_df = pd.DataFrame(generation_details)
            gen_df.to_csv(os.path.join(self.save_directory, 'generation_details.csv'), index=False)
            print("Generation details saved to generation_details.csv", flush=True)

        return

    def initialize_population(self) -> None:
        """
        Function to initialize the population of pipelines for self.population.
        Size of self.population must be self.pop_size.
        """
        # quick check to make sure hub is initialized
        assert self.hub is not None, "Hub must be initialized before initializing population."
        assert len(self.population) == 0, "Population must be empty before initializing."

        # container for all pipeline set of branches
        pop_branch_sets = []
        # container to hold newly found branches to avoid duplicate work
        unseen_branches = set()
        # get list of chormosomes with available snps
        chomosome_list = self.hub.get_keys_with_snps()

        # create initial set of branch sets to integrate within pipelines
        while len(pop_branch_sets) < self.pop_size:
            # current set of branches
            branches = set()
            # generate sampling list for chromosomes
            sampling_list = self.get_sampling(cnt=self.branch_max, chrom_num=uint16_t(len(chomosome_list)))
            # shuffle chromosome list
            self.rng.shuffle(chomosome_list)
            # go through each chromosome and sample the required number of snps
            for chrom, cnt in enumerate(sampling_list):
                branches.update(self.hub.get_k_snps_from_chrom(self.rng, chomosome_list[chrom], cnt))
            assert len(branches) == self.branch_max, "Number of branches in initial pipeline does not match branch_max."

            # update unseen branches with all new branches
            # bc all branches are new at this point, we can just add them directly
            unseen_branches.update(branches)
            # add the current branch set to the population list
            pop_branch_sets.append(branches)

        # evaluate all unseen branches and update the hub
        self.evaluate_unseen_branches(unseen_branches, gen_seen=int16_t(0))

        # remove inactive branches from each branch set for the initial population
        assert len(pop_branch_sets) == self.pop_size, "Population branch sets size does not match population size."
        assert len(self.population) == 0, "Population must be empty before initializing."
        for b_set in pop_branch_sets:
            # extract only active branches
            b_set_active = self.hub.remove_inactive_branches(b_set)

            assert 0 <= len(b_set_active) <= len(b_set), "Active branch set size must be less than or equal to original branch set size."
            if len(b_set_active) == 0:
                continue

            # generate random pipeline and add to population
            self.population.append(self.reproduction.generate_random_pipeline(self.rng, b_set_active, self.seed))
        assert 1 <= len(self.population) <= self.pop_size, "Population size contained no valid pipelines after branch set evaluation."

        # evaluate the initial population
        print('Evaluating initial population pipelines...', flush=True)
        self.population, _ = self.evaluation(self.population, gen_info=int16_t(0))
        assert 1 <= len(self.population) <= self.pop_size, "Population size contained no valid pipelines after pipeline evaluation."
        return

    def evaluation(self, pipelines: List[Pipeline], gen_info: int16_t) -> Tuple[List[Pipeline], Dict]:
        """
        Function to evaluate entire pipelines.
        All of this should be done in asyncronous parallel jobs for maximum efficiency.

        Parameters:
        pipelines: List[Pipeline]
            List of pipelines to evaluate.
        gen_info: int16_t
            Generation number for logging purposes.

        Returns:
        Tuple[List[Pipeline], Dict]: 
            - List of evaluated pipelines (pipelines updated with evaluation results)
            - Dictionary containing evaluation statistics (fs_only_count, ld_fs_count, sequential_selector_count, fs_time)
        """
        # quick checks
        assert len(pipelines) > 0, "No pipelines to evaluate."

        # keep a count of number of pipelines that are calling only fs vs ld+fs
        fs_only_count = 0
        ld_fs_count = 0
        sequential_selector_count = 0

        # create ray jobs for each pipeline evaluation depending on if ld is needed or not
        ray_jobs = []
        pipeline_evaluation_details = {}
        for i, pipeline in enumerate(pipelines):
            # Count SequentialFeatureSelector usage
            if pipeline.get_selector_node().name == 'SequentialFeatureSelector':
                sequential_selector_count += 1
            pipeline_evaluation_details[i] = {snp_t('r2'): float32_t(0.0), snp_t('feature_cnt'): None, snp_t('features'): None,
                                              snp_t('ld_used'): False, snp_t('error'): False, snp_t('count'): uint16_t(0)}

            # Check if LD pruning should be applied:
            # 1. ld_flag must be True
            # 2. Pipeline must contain SNPs from the same chromosome
            if self.ld_flag and self.snps_on_the_same_chromosome(pipeline.branch_set):
                ray_jobs.append(ray_utils.ray_eval_pipeline_ld_fs.remote(snp_names=[snp for snp in pipeline.get_branch_set()],
                                                                         x_train_ori=[self.hub.get_ori_ray_id(snp) for snp in pipeline.get_branch_set()],
                                                                         x_train_enc=[self.hub.get_enc_ray_id(snp) for snp in pipeline.get_branch_set()],
                                                                         y_train=self.all_y_ray_id,
                                                                         train_idx=self.train_idx_ray,
                                                                         selector_node=pipeline.get_selector_node(),
                                                                         ld_node=pipeline.get_ld_node(),
                                                                         pop_id=uint16_t(i),
                                                                         snp_r2_set=self.hub.generate_r2_dict(pipeline.get_branch_set())))
                pipeline_evaluation_details[i][snp_t('ld_used')] = True
                ld_fs_count += 1
            # else, no need for ld pruner (either ld_flag is False or SNPs are not on same chromosome)
            else:
                ray_jobs.append(ray_utils.ray_eval_pipeline_fs.remote(snp_names=[snp for snp in pipeline.get_branch_set()],
                                                                     x_train_enc=[self.hub.get_enc_ray_id(snp) for snp in pipeline.get_branch_set()],
                                                                     y_train=self.all_y_ray_id,
                                                                     train_idx=self.train_idx_ray,
                                                                     selector_node=pipeline.get_selector_node(),
                                                                     pop_id=uint16_t(i)))
                fs_only_count += 1
        print(f"Total pipelines calling only FS: {fs_only_count}, Total pipelines calling LD + FS: {ld_fs_count}", flush=True)   
        # keep track of LD prunned snps
        pruned_snps = set()
        # will hold the snp details after LD for each pipeline
        snp_details_per_snp = {}
        # process results as they come in
        start_time = time.time()
        while len(ray_jobs) > 0:
            finished, ray_jobs = ray.wait(ray_jobs)
            error, feature_cnt, pop_id, features, ld_details = ray.get(finished[0])
            assert feature_cnt == len(features), "Feature count does not match number of features returned."

            # update pipeline evaluation details
            if error < float32_t(0.0):
                pipeline_evaluation_details[pop_id][snp_t('error')] = True
            pipeline_evaluation_details[pop_id][snp_t('feature_cnt')] = feature_cnt
            pipeline_evaluation_details[pop_id][snp_t('features')] = features

            # add a check that if snp_details_after_ld is an empty dict, we skip the for loop
            if ld_details is None or len(ld_details) == 0:
                continue

            # update the pruned snps based on the snp details after LD
            for snp, details in ld_details.items():
                if details['pruned'] == True:
                    pruned_snps.add(snp)
                    snp_details_per_snp[snp] = details
        # timing print and save
        fs_time = (time.time() - start_time) / 60  # in minutes
        print(f"Feature selection (ld->fs | fs) took {fs_time} mins", flush=True)

        # send pipelines with no error to be evaluated for r2 across k-folds (only pipelines with error == False)
        ray_jobs = []
        for pipeline_id in pipeline_evaluation_details:
            if pipeline_evaluation_details[pipeline_id][snp_t('error')]:
                continue

            # create a ray job for each of the folds
            for _, fold_data in self.train_fold_dict_ray.items():
                ray_jobs.append(ray_utils.ray_eval_pipeline_r2.remote(X = [self.hub.get_enc_ray_id(snp) for snp in pipeline_evaluation_details[pipeline_id][snp_t('features')]],
                                                                      y = self.all_y_ray_id,
                                                                      train_idx = fold_data['train_idx'],
                                                                      valid_idx = fold_data['val_idx'],
                                                                      pop_id = uint16_t(pipeline_id)))
        # process results as they come in
        start_time = time.time()
        while len(ray_jobs) > 0:
            finished, ray_jobs = ray.wait(ray_jobs)
            r2, pop_id, error = ray.get(finished[0])
            if error < float32_t(0.0):
                pipeline_evaluation_details[pop_id][snp_t('error')] = True

            # update r2 and count
            pipeline_evaluation_details[pop_id][snp_t('r2')] += r2
            pipeline_evaluation_details[pop_id][snp_t('count')] += uint16_t(1)
        # timing print
        print(f"R2 evaluation took {(time.time() - start_time) / 60} mins", flush=True)

        # update hubs with prunned snps info
        self.hub.process_pruned_snps(pruned_snps, snp_details_per_snp, gen_info)

        # will hold the evaluated pipelines that passed evaluation
        evaluated_pipelines : List[Pipeline] = []

        # update pipelines with evaluation results
        for pipeline_id in pipeline_evaluation_details:
            # skip pipelines with error, negative r2, or all snps are inactive
            if pipeline_evaluation_details[pipeline_id][snp_t('error')] or \
                pipeline_evaluation_details[pipeline_id][snp_t('r2')] < float32_t(0.0) or \
                self.hub.at_least_one_active_snp(pipeline_evaluation_details[pipeline_id][snp_t('features')]) == False:
                continue

            assert pipeline_evaluation_details[pipeline_id][snp_t('count')] == uint16_t(self.k), "Pipeline evaluation must have k-fold evaluations."
            pipelines[pipeline_id].set_traits([ pipeline_evaluation_details[pipeline_id][snp_t('r2')] / float32_t(self.k),
                                                pipeline_evaluation_details[pipeline_id][snp_t('feature_cnt')],
                                                set(pipeline_evaluation_details[pipeline_id][snp_t('features')]) ])
            # add to evaluated pipelines
            evaluated_pipelines.append(pipelines[pipeline_id])

        # Create statistics dictionary
        eval_stats = {
            'fs_only_count': fs_only_count,
            'ld_fs_count': ld_fs_count,
            'sequential_selector_count': sequential_selector_count,
            'fs_ld_time_all_pipelines': fs_time
        }

        return evaluated_pipelines, eval_stats

    def process_offspring(self, pipelines: List[Pipeline], gen_info: int16_t) -> List[Pipeline]:
        # quick checks
        assert len(pipelines) > 0, "No pipelines to process."
        assert len(pipelines) <= self.pop_size * 2, "Number of pipelines exceeds maximum offspring size."
        assert gen_info >= 0, "Generation info must be non-negative."


        print('Processing offspring pipelines: evaluating unseen branches and removing inactive branches...', flush=True)
        # collect all snps from each pipeline and send to hub to find unseen snps
        all_snps = set()
        for pipeline in pipelines:
            all_snps.update(pipeline.get_branch_set())
        unseen_snps = self.hub.get_unseen_snps(all_snps)

        # evaluate all unseen snps if we have any to evaluate
        if len(unseen_snps) > 0:
            self.evaluate_unseen_branches(unseen_snps, gen_info)

        # offspring pipelines with no good snps
        updated_pipelines = []

        for pipeline in pipelines:
            good_snps = self.hub.remove_inactive_branches(pipeline.get_branch_set())
            if len(good_snps) == 0:
                # skip this iteration if there are no good snps
                continue

            updated_pipelines.append(Pipeline(
                branch_set=good_snps,
                selector_node=pipeline.get_selector_node(),
                ld_node=pipeline.get_ld_node()
            ))
        return updated_pipelines

    # function to check is a branch set has snps on the same chromosome: will return True if so
    def snps_on_the_same_chromosome(self, branch_set: Set[snp_t]) -> bool:
        """
        Function to check if a branch set has SNPs on the same chromosome.

        Parameters:
        branch_set: Set[snp_t]
            A set of SNPs to check.

        Returns:
        bool: True if any SNPs are on the same chromosome, False otherwise.
        """
        assert len(branch_set) > 0, "Branch set must not be empty."
        assert len(branch_set) <= self.branch_max, "Branch set size exceeds maximum allowed branches."
        assert all(isinstance(snp, snp_t) for snp in branch_set), "All elements in branch set must be of type snp_t."
        assert all('.' in snp for snp in branch_set), "All SNPs must be in the format 'chrom.pos'."

        chromosomes = set()
        for snp in branch_set:
            chrom, _ = snp_chrm_pos(snp)
            if chrom in chromosomes:
                return True
            chromosomes.add(chrom)
        return False

    def evaluate_unseen_branches(self, unseen_branches: Set, gen_seen: int16_t) -> None:
        """
        Function to evaluate all unseen branches and add their best R2 and Encoder type to the Hub.
        All of this should be done in asyncronous parallel jobs.
        Only SNPs with r2 > self.branch_explainability_threshold will have their encoded version stored in the hub.

        Parameters:
        unseen_branches: Set
            Unseen branches in a set to evaluate.
        """
        # quick checks
        assert len(unseen_branches) > 0, "No unseen branches to evaluate."
        assert gen_seen >= 0, "Generation seen must be non-negative."

        # container for ray object ids
        ray_jobs = []
        for snp in unseen_branches:
            assert isinstance(snp, snp_t), "SNP must be of type snp_t."
            assert '.' in snp, "SNP must be a string with chromosome and position separated by a dot."

            # Launch one Ray job per SNP per fold that evaluates ALL 9 encodings
            for _, fold_data in self.train_fold_dict_ray.items():
                ray_jobs.append(ray_utils.ray_snp_eval_all_encodings.remote(
                    X = self.hub.get_ori_ray_id(snp), 
                    y = self.all_y_ray_id, 
                    train_idx = fold_data['train_idx'],
                    valid_idx = fold_data['val_idx'], 
                    snp = snp
                ))
        assert len(ray_jobs) == len(unseen_branches) * self.k  # k folds for each unseen snp (all encodings in one call)

        # container to hold snp performance (accumulated r2, count, error flag, encoder type, encoded_x ray id(depending on r2 / count >= threshold))
        snp_perf = {}
        for snp_name in unseen_branches:
            # initialize with best encoder as None and encoded_x as None
            snp_perf[snp_name] = {snp_t('b_encoder'): None, snp_t('encoded_x'): None}
            for encoder in self.encoder_types:
                # extend snp_perf with r2 and count for each of the encoders
                snp_perf[snp_name][encoder] = {snp_t('r2'): float32_t(0.0), snp_t('cnt'): float32_t(0.0)}
        assert len(snp_perf) == len(unseen_branches), "SNP performance dictionary size does not match unseen branches."

        # process results as they come in
        start_time = time.time()
        while len(ray_jobs) > 0:
            # collect results
            finished, ray_jobs = ray.wait(ray_jobs)
            encoding_results = ray.get(finished)[0]  # Returns dict of {encoding_name: (r2, snp, enc, error)}
            
            # Process results for all encodings from this single job
            for enc_name, (r2, snp_name, lo, error) in encoding_results.items():
                assert error >= 0.0, f"Error flag must be non-negative for {enc_name}. Error during SNP evaluation cannot occur."
                # add them up
                snp_perf[snp_name][lo][snp_t('r2')] += r2
                snp_perf[snp_name][lo][snp_t('cnt')] += float32_t(1.0)
        # timing print
        print(f"Evaluating {len(unseen_branches)} unseen branches took {(time.time() - start_time) / 60} mins", flush=True)

        # obtain encoded snp for each unseen branch with a best positve r2
        ray_jobs = []
        for snp_name in unseen_branches:
            # find best encoder for the snp
            best_r2 = float32_t(-100000.0)
            best_encoder = None

            # go through each encoder and find the best average r2
            for encoder in self.encoder_types:
                # make sure we have at least k-folds of results
                assert int(snp_perf[snp_name][encoder][snp_t('cnt')]) == float32_t(self.k), "SNP performance count does not match k-folds."

                # only care about largest aggregated r2 up to this point
                if snp_perf[snp_name][encoder][snp_t('r2')] > best_r2:
                    best_r2 = snp_perf[snp_name][encoder][snp_t('r2')]
                    best_encoder = encoder

            # save best encoder for the snp
            snp_perf[snp_t(snp_name)][snp_t('b_encoder')] = snp_t(best_encoder)

            # if we have an average r2 greater or equal than the threshold, get the encoded snp ray id
            if best_r2 / float32_t(self.k) >= self.branch_explainability_threshold and best_encoder != snp_t('additive'):
                ray_jobs.append(ray_utils.ray_snp_encoder.remote(X = self.hub.get_ori_ray_id(snp_name),
                                                                 y = self.all_y_ray_id,
                                                                 train_idx = self.train_idx_ray,
                                                                 enc = best_encoder,
                                                                 snp = snp_name))
        # process encoded snp results
        start_time = time.time()
        count = 0
        while len(ray_jobs) > 0:
            # collect results and store encoded snp
            finished, ray_jobs = ray.wait(ray_jobs)
            encoded_snp, snp_name = ray.get(finished)[0]
            snp_perf[snp_name][snp_t('encoded_x')] = encoded_snp
            count += 1
        # timing print
        print(f"Encoding {count} unseen branches took {(time.time() - start_time) / 60} mins", flush=True)

        # update the hub with best r2 and encoded snp ray id (if r2 >= threshold)
        for snp_name in unseen_branches:
            enc_id = None
            # set enc_id to those snps with r2 / k >= threshold
            if snp_perf[snp_name][snp_perf[snp_name][snp_t('b_encoder')]][snp_t('r2')] / float32_t(self.k) >= self.branch_explainability_threshold:
                enc_id = ray.put(snp_perf[snp_name][snp_t('encoded_x')])

            self.hub.update_snp_hub_r2_enc(snp=snp_name,
                                          r2=snp_perf[snp_name][snp_perf[snp_name][snp_t('b_encoder')]][snp_t('r2')] / float32_t(self.k),
                                          enc=snp_perf[snp_name][snp_t('b_encoder')],
                                          enc_x=enc_id,
                                          gen_seen=gen_seen,
                                          snp_explainability_threshold=self.branch_explainability_threshold)

    def get_sampling(self, cnt:uint16_t, chrom_num:uint16_t) -> npt.NDArray[uint16_t]:
        """
        Function to get the sampling list that evenly splits the number of snps to sample from each chromosome.
        Note that the chrom_num index is mapped to the chromosome keys provided by the hub.

        Parameters:
        cnt: uint16_t
            total number of snps to sample based on each pipeline's randomized init
        chrom_num: uint16_t
            total number of chromosomes based on snp hub
        """
        assert cnt > 0
        assert chrom_num > 0

        # how many SNPs should each chromosome get
        sample_num = cnt//chrom_num
        # how many extra SNPs are needed to complete the count
        remainder = cnt%chrom_num
        # create sampling list with the base number of SNPs per chromosome
        sampling_list = np.full(shape=chrom_num,fill_value=sample_num)

        # distribute the remainder SNPs randomly across chromosomes
        if remainder > 0:
            start_idx = self.rng.integers(low=0, high=chrom_num)
            for i in range(remainder):
                # Use modulo to wrap around and avoid index errors
                sampling_list[(start_idx+i) % chrom_num] += 1
        return sampling_list

    def post_analysis_with_good_snps(self):
        """
        Function to perform analysis on all the Pareto front pipelines using the validation set using only the good snps seen during evolution.
        """
        # get the Pareto front pipelines
        _, rank = nsga.non_dominated_sorting(obj_scores=self.get_pipeline_scores(self.population, weights=(float32_t(1.0), int32_t(-1))))
        pareto_front_pipelines = []
        for i, r in enumerate(rank):
            if r == 0: # rank 0 is the Pareto front
                pareto_front_pipelines.append(self.population[i])
        print(f"Number of pipelines in Pareto front for post analysis: {len(pareto_front_pipelines)}", flush=True)

        # # sort the Pareto front by feature count
        # pareto_front_pipelines = sorted(pareto_front_pipelines, key=lambda x: x.get_trait_feature_cnt())

        # # print all the pipelines in the Pareto front
        # for pid, pipeline in enumerate(pareto_front_pipelines):
        #     print(f"Pipeline ID: {pid}", flush=True)
        #     print(f"Train R2: {pipeline.get_trait_r2()}", flush=True)
        #     print(f"Feature Count: {pipeline.get_trait_feature_cnt()}", flush=True)
        #     print(f"Feature Set: {pipeline.get_trait_feature_names()}", flush=True)
        #     print(f"Selector: {pipeline.get_selector_node().name} with params {pipeline.get_selector_node().get_params()}", flush=True)
        #     print('-'*30, flush=True)

        # collect all the parallel jobs for evaluating the pipelines on the validation set
        ray_jobs = []
        pareto_validation_r2 = {}
        # collect all snps that made it to the regressor for Pareto pipelines to evaluate on validation set (the pruned ones are not considered)
        for pipeline_id, pipeline in enumerate(pareto_front_pipelines):
            # filter only active SNPs
            features_final = [snp for snp in pipeline.get_trait_feature_names() if self.hub.get_active_flag(snp) == True]

            # store details for each pipeline
            pareto_validation_r2[pipeline_id] = {'validation_r2': float32_t(-1.0),
                                                 'test_r2': float32_t(-1.0),
                                                 'train_r2': pipeline.get_trait_r2(),
                                                 'feature_cnt': pipeline.get_trait_feature_cnt(),
                                                 'selector': pipeline.get_selector_node().name,
                                                 'selector_params': pipeline.get_selector_node().get_params(),
                                                 'feature_set': set(features_final),
                                                 'pipeline': pipeline  # Store pipeline object to access root node
                                                 }
            # create ray job for evaluating the pipeline on the validation set
            ray_jobs.append(ray_utils.ray_eval_pipeline_r2.remote(X=[self.hub.get_enc_ray_id(snp) for snp in features_final],
                                                                   y=self.all_y_ray_id,
                                                                   train_idx=self.train_idx_ray,
                                                                   valid_idx=self.val_idx_ray,
                                                                   pop_id=uint16_t(pipeline_id)))
            # process results as they come in
            while len(ray_jobs) > 0:
                finished, ray_jobs = ray.wait(ray_jobs)
                r2, pop_id, error = ray.get(finished)[0]
                if error < float32_t(0.0):
                    print(f"Error during post analysis evaluation of pipeline {pop_id}", flush=True)
                    continue
                pareto_validation_r2[pop_id]['validation_r2'] = float32_t(r2)
        print("Post analysis on validation set completed.", flush=True)

        # sort pareto_validation_r2 by key (pipeline_id)
        pareto_validation_r2 = dict(sorted(pareto_validation_r2.items()))

        ################# FIND THE UTOPIA MODEL #################

        # find model based on Utopia point (maximize r2 and minimize feature count)
        utopia_pipeline_id = {}
        max_r2 = max([data["validation_r2"] for data in pareto_validation_r2.values()])
        min_r2 = min([data["validation_r2"] for data in pareto_validation_r2.values()])
        max_comp = max([data["feature_cnt"] for data in pareto_validation_r2.values()])
        min_comp = min([data["feature_cnt"] for data in pareto_validation_r2.values()])

        for pid, data in pareto_validation_r2.items():
            # save utopia distance score
            r2 = (1 - ((data["validation_r2"] - min_r2) / (max_r2 - min_r2)))**2
            comp = (1 - (1 - (data["feature_cnt"] - min_comp) / (max_comp - min_comp)))**2
            utopia_pipeline_id[pid] = np.sqrt(r2 + comp)

        # find the set of snps with the smallest utopia distance
        min_distance = float32_t(10000000.0)
        utopia_point_pipeline_id = None
        for pid, distance in utopia_pipeline_id.items():
            if min_distance > distance:
                min_distance = distance
                utopia_point_pipeline_id = pid

        # print the details of the utopia point pipeline
        print(f"Utopia Point Pipeline ID: {utopia_point_pipeline_id}", flush=True)
        print(f"Utopia Point Pipeline Train R2: {pareto_validation_r2[utopia_point_pipeline_id]['train_r2']}", flush=True)
        print(f"Utopia Point Pipeline Validation R2: {pareto_validation_r2[utopia_point_pipeline_id]['validation_r2']}", flush=True)
        print(f"Utopia Point Pipeline Feature Count: {pareto_validation_r2[utopia_point_pipeline_id]['feature_cnt']}", flush=True)
        print(f"Utopia Point Pipeline Feature Set: {pareto_validation_r2[utopia_point_pipeline_id]['feature_set']}", flush=True)

        # send snps of pipeline with utopia point to final test
        self.final_pipeline_test(pareto_validation_r2[utopia_point_pipeline_id],
                                 'utopia_pipeline_test_results.csv',
                                 pareto_validation_r2[utopia_point_pipeline_id]['train_r2'],
                                 pareto_validation_r2[utopia_point_pipeline_id]['validation_r2'],
                                 pareto_validation_r2[utopia_point_pipeline_id]['feature_cnt'])
        return

    # function to take in a set of snp names and perform final test on the test dataset
    def final_pipeline_test(self, pipeline_data: Dict, file_name: str, train_r2: float32_t, validation_r2: float32_t, size: int16_t) -> None:
        """
        Function to perform the final test on the test dataset.
        Combines training and validation datasets to fit a linear regression model.
        Then, evaluates the model on the test dataset and computes PFI using ray_pfi.
        Saves results to a CSV file.
        """
        snp_names = list(pipeline_data['feature_set'])  # Get features from pipeline_data
        print(f"Performing final test on {len(snp_names)} SNPs...", flush=True)

        # Combine training and validation indices for final model training
        combined_train_idx = np.concatenate([ray.get(self.train_idx_ray), ray.get(self.val_idx_ray)])
        combined_idx_ray_id = ray.put(combined_train_idx)
        test_idx_ray_id = ray.put(self.test_idx)

        # Re-encode SNPs based on their best encoding type
        # Use combined train+validation data for encoding (to learn encoding from larger dataset)
        ray_jobs = []
        for snp in snp_names:
            ray_jobs.append(ray_utils.ray_snp_encoder.remote(
                X=self.hub.get_ori_ray_id(snp),
                y=self.all_y_ray_id,
                train_idx=combined_train_idx,  # Use combined indices for encoding
                enc=self.hub.get_encoding(snp),
                snp=snp
            ))

        print(f"Encoding {len(snp_names)} SNPs for final test...", flush=True)

        # Process encoding results
        transformed_snp_ray_ids = {}
        while len(ray_jobs) > 0:
            finished, ray_jobs = ray.wait(ray_jobs)
            encoded_x, snp_name = ray.get(finished)[0]
            # Put encoded array into Ray object store
            transformed_snp_ray_ids[snp_name] = ray.put(encoded_x)

        # Create column names for PFI (include encoding type)
        column_names_with_encoding = [f'chr{snp}_{self.hub.get_encoding(snp)}' for snp in snp_names]

        # Calculate train + validation R² using ray remote function
        print("Calculating train + validation R²...", flush=True)
        train_valid_r2_job = ray_utils.ray_eval_pipeline_r2.remote(
            X=[transformed_snp_ray_ids[snp] for snp in snp_names],
            y=self.all_y_ray_id,
            train_idx=combined_idx_ray_id,
            valid_idx=combined_idx_ray_id,
            pop_id=uint16_t(0)
        )
        train_val_r2, _, error = ray.get(train_valid_r2_job)
        if error < float32_t(0.0):
            print(f"Error during train + validation R² calculation", flush=True)
            train_val_r2 = float32_t(-1.0)
        print(f'Train + Validation R² Score: {train_val_r2}', flush=True)

        # Calculate test R² using ray remote function
        print("Calculating test R²...", flush=True)
        test_r2_job = ray_utils.ray_eval_pipeline_r2.remote(
            X=[transformed_snp_ray_ids[snp] for snp in snp_names],
            y=self.all_y_ray_id,
            train_idx=combined_idx_ray_id,
            valid_idx=test_idx_ray_id,
            pop_id=uint16_t(0)
        )

        test_r2, _, error = ray.get(test_r2_job)

        if error < float32_t(0.0):
            print(f"Error during test R² calculation", flush=True)
            test_r2 = float32_t(-1.0)

        pipeline_data['test_r2'] = test_r2
        print(f'Test R² Score: {test_r2}', flush=True)

        # Calculate PFI on test set using ray_pfi
        print("Calculating permutation feature importance on test set...", flush=True)

        # Create an OLS regressor for PFI calculation
        ols_regressor = OLSRegressor()

        # Call ray_pfi
        pfi_job = ray_utils.ray_pfi.remote(
            X=[transformed_snp_ray_ids[snp] for snp in snp_names],
            y=self.all_y_ray_id,
            train_idx=combined_train_idx,
            valid_idx=self.test_idx,
            new_column_names=column_names_with_encoding,
            root_node=ols_regressor,
            random_state=self.rng.integers(0, 100000),
            pop_id=uint16_t(0)
        )

        pfi_results, _ = ray.get(pfi_job)

        print(f"PFI calculated for {len(pfi_results)} features", flush=True)

        # Create DataFrame with PFI results
        pfi_df = pd.DataFrame(list(pfi_results.items()), columns=['SNP', 'Importance'])
        pfi_df = pfi_df.sort_values(by='Importance', ascending=False)
        pfi_df['Cross-validated Train R2'] = train_val_r2
        pfi_df['Validation R2'] = validation_r2
        pfi_df['Train+Valid R2'] = train_val_r2
        pfi_df['Test R2'] = test_r2
        pfi_df['Model Size'] = size

        # Save to CSV
        output_path = os.path.join(self.save_directory, file_name)
        pfi_df.to_csv(output_path, index=False)
        print(f"Results saved to {file_name}", flush=True)

    # function to save and plot Pareto front at the end of evolution
    def save_and_plot_pareto_front(self):
        """
        Function to save and plot the Pareto front at the end of evolution.
        Saves the Pareto front pipelines to a CSV file and generates a plot.
        """
        # get the front 0 pipelines
        _, rank = nsga.non_dominated_sorting(obj_scores=self.get_pipeline_scores(self.population, weights=(float32_t(1.0), int32_t(-1))))
        pareto_front_pipelines = []
        for i, r in enumerate(rank):
            if r == 0: # rank 0 is the Pareto front
                pareto_front_pipelines.append(self.population[i])
        print(f"Number of pipelines in Pareto front at the end of evolution:{len(pareto_front_pipelines)}", flush=True)
        # sort the Pareto front by feature count
        pareto_front_pipelines = sorted(pareto_front_pipelines, key=lambda x: x.get_trait_feature_cnt())
        # save the Pareto pipeline details to a csv file
        pareto_data = []
        for pid, pipeline in enumerate(pareto_front_pipelines):
            pareto_data.append({
                'Pipeline ID': pid + 1,  # Start from 1 instead of 0
                'Cross-validated R2': pipeline.get_trait_r2(),
                'Feature Count': pipeline.get_trait_feature_cnt(),
                'Feature Set': ';'.join(pipeline.get_trait_feature_names()),
                'Selector': pipeline.get_selector_node().name,
                'Selector Params': pipeline.get_selector_node().get_params()
            })
        pareto_df = pd.DataFrame(pareto_data)
        pareto_df.to_csv(os.path.join(self.save_directory, 'pareto_front_pipelines.csv'), index=False)
        print("Pareto front pipelines saved to pareto_front_pipelines.csv", flush=True)
        # plot the Pareto front
        plt.figure(figsize=(10, 6))
        plt.title('Pareto Front: R2 vs Feature Count')
        plt.xlabel('Feature Count')
        plt.ylabel('Cross-validated R2')
        plt.grid(True)

        # plot pareto front pipelines as red dots
        pareto_r2 = [pipeline.get_trait_r2() for pipeline in pareto_front_pipelines]
        pareto_feat_cnt = [pipeline.get_trait_feature_cnt() for pipeline in pareto_front_pipelines]
        plt.scatter(pareto_feat_cnt, pareto_r2, color='red', label='Pareto Front')

        # Annotate the points with pipeline numbers (indexes in pareto front)
        for i, (feature_count, r2_score) in enumerate(zip(pareto_feat_cnt, pareto_r2)):
            plt.annotate(
                str(i + 1),  # Text label (pipeline ID starting from 1)
                (feature_count, r2_score),  # The point where the annotation should be
                textcoords="offset points",  # Use offset for better readability
                xytext=(5, 5),  # Offset position (x, y)
                ha='center',  # Horizontal alignment
                fontsize=9,
                color='blue'
            )

        plt.legend()
        plt.savefig(os.path.join(self.save_directory, 'pareto_front_plot.png'))
        print("Pareto front plot saved to pareto_front_plot.png", flush=True)
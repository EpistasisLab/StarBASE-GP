
# import base EA class and types
from ..Base.evovler import EA
from ..Base.types import (float32_t, int16_t, prob_t, int32_t, snp_t, uint16_t, uint32_t)
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

@typechecked
class K1_Evolver(EA):
    def __init__(self,
                 seed: int,
                 pop_size: uint32_t,
                 branch_max: uint16_t,
                 branch_min: uint16_t,
                 cores: int,
                 mut_prob: prob_t = prob_t(.5), # probability of mutation
                 cross_prob: prob_t = prob_t(.5), # probability of crossover
                 mut_selector_p: prob_t = prob_t(.5), # probability of mutating the feature selector
                 mut_ld_p: prob_t = prob_t(.5), # probability of mutating the ld pruner
                 mut_root_p: prob_t = prob_t(.5), # probability of mutating the root node (regressor or classifier)
                 mut_ran_p: prob_t = prob_t(1.0), # probability of random mutation
                 mut_smt_p: prob_t = prob_t(0.0), # probability of smart mutation
                 m_in_win_p: prob_t = prob_t(.33), # probability for smart in window mutation
                 m_out_win_p: prob_t = prob_t(.33), # probability for smart out window mutation
                 m_out_chr_p: prob_t = prob_t(.33), # probability for smart out of chromosome mutation
                 save_directory: str = "",
                 window_distance: int32_t = int32_t(1000000),
                 branch_explainability_threshold: float32_t = float32_t(0.0),
                 ld_flag: bool = True,
                 encoding_flag: bool = True,
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
        self.encoding_flag = encoding_flag

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
        self.hub = K1_Hub(snp_list=self.snp_labels, snps_ray_ids=feature_ray_ids, window_distance=self.window_distance)
        return

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
            gens (uint16_t): Number of generations to evolve the population.
        """

        # quick check
        assert gens >= 0, "Number of generations must be non-negative."

        # print initial hub stats
        print('Initial Hub details:')
        self.hub.seen_snps_proportion()
        print('', flush=True)
        # list to store the generation details - front zero size, still consider snp set size, number of snps pruned
        generation_details = []
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
                _, rank = nsga.non_dominated_sorting(obj_scores=self.get_pipeline_scores(pipelines=self.population, weights=(float32_t(1.0), int32_t(-1))))
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

            # start generation timer
            gen_start_time = time.time()
            
            # Timing for parent selection and offspring generation
            selection_start = time.time()

            # get order of mutation/crossover to do with the extra offspring
            var_order = None
            parent_cnt = None
            parent_ids = None
            if len(self.population) == 1:
                var_order, parent_cnt = [snp_t('m')] * uint32_t(2*self.pop_size), uint32_t(2*self.pop_size)
                parent_ids = [uint16_t(0)] * parent_cnt
            else:
                var_order, parent_cnt = self.reproduction.variation_order(self.rng, uint32_t(2*self.pop_size))
                parent_ids = self.parent_selection(parent_cnt)

            # generate offspring
            offspring = self.reproduction.produce_offspring(rng = self.rng,
                                                           hub = self.hub,
                                                           offspring_cnt = uint32_t(2*self.pop_size),
                                                           parent_ids = parent_ids,
                                                           population = self.population,
                                                           order = var_order)
            selection_time = (time.time() - selection_start) / 60
            print(f"Parent selection + offspring generation: {selection_time:.2f} mins", flush=True)
            
            # make sure we have the correct number of offspring solutions
            assert 0 < len(offspring) <= 2 * self.pop_size

            # Timing for processing offspring
            process_start = time.time()
            # process offspring: evaluation interactions and remove bad interactions
            offspring = self.process_offspring(offspring, int16_t(g))
            process_time = (time.time() - process_start) / 60
            print(f"Process offspring (unseen branches eval): {process_time:.2f} mins", flush=True)

            # Timing for offspring evaluation
            eval_start = time.time()
            # evaluate the offspring
            print('Evaluating offspring pipelines...', flush=True)
            offspring, eval_stats = self.evaluation(offspring, int16_t(g))
            eval_time = (time.time() - eval_start) / 60
            print(f"Offspring evaluation (LD+FS+R2): {eval_time:.2f} mins", flush=True)

            # must be less than or equal because of potential negative r2 offspring pipelines
            assert (0 < len(offspring) + len(self.population) <= 3 * self.pop_size)

            # Timing for survival selection
            survival_start = time.time()
            # survival selection
            self.population = self.survival_selection(offspring)
            survival_time = (time.time() - survival_start) / 60
            print(f"Survival selection: {survival_time:.2f} mins", flush=True)

            # make sure we have the correct number of pipelines
            assert len(self.population) <= self.pop_size

            # Calculate total generation time
            gen_time = (time.time() - gen_start_time) / 60  # in minutes
            print(f"TOTAL generation time: {gen_time:.2f} minutes", flush=True)
            
            # Print breakdown percentages
            print(f"Time breakdown - Selection: {(selection_time/gen_time)*100:.1f}%, Process: {(process_time/gen_time)*100:.1f}%, Eval: {(eval_time/gen_time)*100:.1f}%, Survival: {(survival_time/gen_time)*100:.1f}%", flush=True)

            # Add evaluation stats and timing to generation details
            gen_stats.update({
                'total_time_mins': gen_time,
                'selection_time_mins': selection_time,
                'process_time_mins': process_time,
                'eval_time_mins': eval_time,
                'survival_time_mins': survival_time,
                'fs_only_count': eval_stats['fs_only_count'],
                'pipelines_evaluated': eval_stats['pipelines_evaluated']
            })
            generation_details.append(gen_stats)

            print('-'*50, flush=True)

        # prints for the end of a run and the final population
        print('Final run/population details')
        print('Final population size:', len(self.population), flush=True)

        # collect all the population ids that belong to front 0
        self.front_zero_ids = nsga.front_zero(obj_scores=self.get_pipeline_scores(pipelines=self.population, weights=(float32_t(1.0), int32_t(-1))))

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

        # break up unseen_branches into chunks of 2000 to avoid ray overload and then run evaluate_unseen_branches on each chunk
        unseen_branches_list = list(unseen_branches)
        for i in range(0, len(unseen_branches_list), 2000):
            print(f"Evaluating unseen branches chunk {i // 2000 + 1} / {(len(unseen_branches_list) - 1) // 2000 + 1}", flush=True)
            chunk = set(unseen_branches_list[i:i+2000])
            self.evaluate_unseen_branches(chunk, gen_seen=int16_t(0))

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
        Function to evaluate pipelines.
        All of this should be done in asyncronous parallel jobs for maximum efficiency.

        Parameters:
        pipelines: List[Pipeline]
            List of pipelines to evaluate.
        gen_info: int16_t
            Generation number for logging purposes.

        Returns:
        Tuple[List[Pipeline], Dict]:
            - List of evaluated pipelines (pipelines updated with evaluation results)
            - Dictionary containing evaluation statistics (fs_only_count, pipelines_evaluated)
        """

        # quick checks
        assert len(pipelines) > 0, "No pipelines to evaluate."

        # keep a count of number of pipelines that are calling only fs vs ld+fs
        fs_only_count = 0

        # create ray jobs for each pipeline evaluation depending on if ld is needed or not
        ray_jobs = []
        pipeline_evaluation_details = {}
        for i, pipeline in enumerate(pipelines):
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
                                                                         pop_id=uint32_t(i),
                                                                         snp_r2_set=self.hub.generate_r2_dict(pipeline.get_branch_set())))
                pipeline_evaluation_details[i][snp_t('ld_used')] = True
            # else, no need for ld pruner (either ld_flag is False or SNPs are not on same chromosome)
            else:
                ray_jobs.append(ray_utils.ray_eval_pipeline_fs.remote(snp_names=[snp for snp in pipeline.get_branch_set()],
                                                                     x_train_enc=[self.hub.get_enc_ray_id(snp) for snp in pipeline.get_branch_set()],
                                                                     y_train=self.all_y_ray_id,
                                                                     train_idx=self.train_idx_ray,
                                                                     selector_node=pipeline.get_selector_node(),
                                                                     pop_id=uint32_t(i)))
                fs_only_count += 1
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

            # Track if this was an LD job (has ld_details)
            if ld_details is not None and len(ld_details) > 0:
                # update the pruned snps based on the snp details after LD
                for snp, details in ld_details.items():
                    if details['pruned'] == True:
                        pruned_snps.add(snp)
                        snp_details_per_snp[snp] = details

        # if self.ld_flag is False, pruned_snps should be empty
        assert (len(pruned_snps) == 0) if self.ld_flag == False else True, "Pruned SNPs should be empty when LD flag is False."
        # timing print
        print(f"Feature selection (ld->fs | fs) took {(time.time() - start_time) / 60} mins", flush=True)

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
                                                                      pop_id = uint32_t(pipeline_id)))
        # process results as they come in
        start_time = time.time()
        while len(ray_jobs) > 0:
            finished, ray_jobs = ray.wait(ray_jobs)
            r2, pop_id, error = ray.get(finished[0])
            if error < float32_t(0.0):
                pipeline_evaluation_details[pop_id][snp_t('error')] = True

            # update r2 and count
            pipeline_evaluation_details[pop_id][snp_t('r2')] += r2
            pipeline_evaluation_details[pop_id][snp_t('count')] += uint32_t(1)
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

        return evaluated_pipelines, {'fs_only_count': fs_only_count, 'pipelines_evaluated': len(evaluated_pipelines)}

    def process_offspring(self, pipelines: List[Pipeline], gen_info: int16_t) -> List[Pipeline]:
        """
        Function to process the offspring pipelines after they have been generated.
        This includes branch set updates, pipeline evaluation, removal of bad pipelines, etc.

        Args:
            pipelines (List[Pipeline]): List of offspring pipelines to process.
            gen_info (int16_t): Generation information for logging purposes.

        Returns:
            List[Pipeline]: List of processed pipelines.
        """
        process_start = time.time()

        # quick checks
        assert len(pipelines) > 0, "No pipelines to process."
        assert len(pipelines) <= self.pop_size * 2, "Number of pipelines exceeds maximum offspring size."
        assert gen_info >= 0, "Generation info must be non-negative."

        print('Processing offspring pipelines: evaluating unseen branches and removing inactive branches...', flush=True)
        
        # Step 1: Collect all SNPs and identify unseen ones
        collect_start = time.time()
        all_snps = set()
        for pipeline in pipelines:
            all_snps.update(pipeline.get_branch_set())
        unseen_snps = self.hub.get_unseen_snps(all_snps)
        collect_time = time.time() - collect_start
        print(f"  Process - Collect SNPs & identify unseen: {collect_time:.4f}s ({len(all_snps)} total, {len(unseen_snps)} unseen)", flush=True)

        # Step 2: Evaluate all unseen snps if we have any to evaluate
        eval_unseen_time = 0.0
        if len(unseen_snps) > 0:
            eval_unseen_start = time.time()
            # break up unseen_branches into chunks of 2000 to avoid ray overload and then run evaluate_unseen_branches on each chunk
            unseen_branches_list = list(unseen_snps)
            for i in range(0, len(unseen_branches_list), 2000):
                print(f"Evaluating unseen branches chunk {i // 2000 + 1} / {(len(unseen_branches_list) - 1) // 2000 + 1}", flush=True)
                chunk = set(unseen_branches_list[i:i+2000])
                self.evaluate_unseen_branches(chunk, gen_seen=int16_t(gen_info))
            eval_unseen_time = time.time() - eval_unseen_start
            print(f"  Process - Evaluate unseen branches: {eval_unseen_time:.4f}s", flush=True)

        # Step 3: Filter pipelines and remove inactive branches
        filter_start = time.time()
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
        filter_time = time.time() - filter_start
        print(f"  Process - Filter inactive branches: {filter_time:.4f}s ({len(pipelines)} -> {len(updated_pipelines)} pipelines)", flush=True)
        
        total_process_time = time.time() - process_start
        print(f"  Process - TOTAL: {total_process_time:.4f}s", flush=True)
        
        return updated_pipelines

    def snps_on_the_same_chromosome(self, branch_set: Set[snp_t]) -> bool:
        """
        Function to check if a branch set has SNPs on the same chromosome.

        Parameters:
            branch_set (Set[snp_t]): Set of SNPs to check.

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

    def evaluate_unseen_branches(self, unseen_branches: Set[snp_t], gen_seen: int16_t) -> None:
        """
        Function to evaluate all unseen branches and add their best R2 and Encoder type to the Hub.
        All of this should be done in asyncronous parallel jobs.
        We update the Hub with the results as they come in.

        Parameters:
            unseen_branches (Set[snp_t]): Set of unseen branches (SNPs) to evaluate.

            gen_seen (int16_t): Generation number when these branches were first seen.
        """
        eval_unseen_total_start = time.time()

        # quick checks
        assert len(unseen_branches) > 0, "No unseen branches to evaluate."
        assert gen_seen >= 0, "Generation seen must be non-negative."

        # Step 1: Launch Ray jobs for SNP evaluation
        launch_jobs_start = time.time()
        ray_jobs = []
        for snp in unseen_branches:
            assert isinstance(snp, snp_t), "SNP must be of type snp_t."
            assert '.' in snp, "SNP must be a string with chromosome and position separated by a dot."

            # Check encoding_flag to determine which encodings to evaluate
            if self.encoding_flag:
                # Launch one Ray job per SNP per fold that evaluates ALL 9 encodings
                for _, fold_data in self.train_fold_dict_ray.items():
                    ray_jobs.append(ray_utils.ray_snp_eval_all_encodings.remote(
                        X = self.hub.get_ori_ray_id(snp),
                        y = self.all_y_ray_id,
                        train_idx = fold_data['train_idx'],
                        valid_idx = fold_data['val_idx'],
                        snp = snp
                    ))
            else:
                # Only evaluate additive encoding
                for _, fold_data in self.train_fold_dict_ray.items():
                    ray_jobs.append(ray_utils.ray_snp_eval_add.remote(
                        X = self.hub.get_ori_ray_id(snp),
                        y = self.all_y_ray_id,
                        train_idx = fold_data['train_idx'],
                        valid_idx = fold_data['val_idx'],
                        snp = snp,
                        lo = snp_t('additive')
                    ))
        assert len(ray_jobs) == len(unseen_branches) * self.k  # k folds for each unseen snp
        launch_jobs_time = time.time() - launch_jobs_start
        print(f"    Unseen - Launch R² eval jobs: {launch_jobs_time:.4f}s ({len(ray_jobs)} jobs)", flush=True)

        # Step 2: Initialize performance tracking
        init_perf_start = time.time()
        snp_perf = {}
        for snp_name in unseen_branches:
            # initialize with best encoder as None, encoded_x as None, and pager_lut_sum as zeros array
            snp_perf[snp_name] = {snp_t('b_encoder'): None, snp_t('encoded_x'): None, snp_t('pager_lut_sum'): np.zeros(3, dtype=float32_t), snp_t('pager_lut_cnt'): 0}
            if self.encoding_flag:
                # Include all encoder types
                for encoder in self.encoder_types:
                    # extend snp_perf with r2 and count for each of the encoders
                    snp_perf[snp_name][encoder] = {snp_t('r2'): float32_t(0.0), snp_t('cnt'): float32_t(0.0)}
            else:
                # Only additive encoding
                snp_perf[snp_name][snp_t('additive')] = {snp_t('r2'): float32_t(0.0), snp_t('cnt'): float32_t(0.0)}
        assert len(snp_perf) == len(unseen_branches), "SNP performance dictionary size does not match unseen branches."
        init_perf_time = time.time() - init_perf_start
        print(f"    Unseen - Initialize perf tracking: {init_perf_time:.4f}s", flush=True)

        # Step 3: Process R² evaluation results
        r2_eval_start = time.time()
        while len(ray_jobs) > 0:
            # collect results
            finished, ray_jobs = ray.wait(ray_jobs)

            if self.encoding_flag:
                # ray_snp_eval_all_encodings returns dict of {encoding_name: (r2, snp, enc, error, pager_lut)}
                encoding_results = ray.get(finished)[0]
                # Process results for all encodings from this single job
                for enc_name, (r2, snp_name, lo, error, pager_lut) in encoding_results.items():
                    assert error >= 0.0, f"Error flag must be non-negative for {enc_name}. Error during SNP evaluation cannot occur."
                    # add them up
                    snp_perf[snp_name][lo][snp_t('r2')] += r2
                    snp_perf[snp_name][lo][snp_t('cnt')] += float32_t(1.0)
                    # Accumulate PAGER LUT values across all folds for averaging
                    if enc_name == 'pager' and pager_lut is not None:
                        snp_perf[snp_name][snp_t('pager_lut_sum')] += pager_lut
                        snp_perf[snp_name][snp_t('pager_lut_cnt')] += 1
            else:
                # ray_snp_eval_add returns (r2, snp, enc, error)
                r2, snp_name, lo, error = ray.get(finished)[0]
                assert error >= 0.0, f"Error flag must be non-negative for additive. Error during SNP evaluation cannot occur."
                # add them up
                snp_perf[snp_name][lo][snp_t('r2')] += r2
                snp_perf[snp_name][lo][snp_t('cnt')] += float32_t(1.0)
        r2_eval_time = time.time() - r2_eval_start
        print(f"    Unseen - Process R² results: {r2_eval_time:.4f}s", flush=True)

        # Step 4: Find best encoder and launch encoding jobs
        best_encoder_start = time.time()
        ray_jobs = []
        for snp_name in unseen_branches:
            # find best encoder for the snp
            best_r2 = float32_t(-10000000000000.0)
            best_encoder = None

            # go through each encoder and find the best average r2
            if self.encoding_flag:
                # Check all encoder types
                for encoder in self.encoder_types:
                    # make sure we have at least k-folds of results
                    assert float32_t(snp_perf[snp_name][encoder][snp_t('cnt')]) == float32_t(self.k), "SNP performance count does not match k-folds."

                    # only care about largest aggregated r2 up to this point
                    if snp_perf[snp_name][encoder][snp_t('r2')] > best_r2:
                        best_r2 = snp_perf[snp_name][encoder][snp_t('r2')]
                        best_encoder = encoder
            else:
                # Only additive encoding
                encoder = snp_t('additive')
                # make sure we have at least k-folds of results
                assert float32_t(snp_perf[snp_name][encoder][snp_t('cnt')]) == float32_t(self.k), "SNP performance count does not match k-folds."
                best_r2 = snp_perf[snp_name][encoder][snp_t('r2')]
                best_encoder = encoder

            # save best encoder for the snp
            snp_perf[snp_t(snp_name)][snp_t('b_encoder')] = snp_t(best_encoder)

            # if we have an average r2 greater or equal than the threshold, get the encoded snp ray id
            # Note: For additive encoding, no need to encode since data is already in additive format
            if best_r2 / float32_t(self.k) >= self.branch_explainability_threshold and best_encoder != snp_t('additive'):
                ray_jobs.append(ray_utils.ray_snp_encoder.remote(X = self.hub.get_ori_ray_id(snp_name),
                                                                 y = self.all_y_ray_id,
                                                                 train_idx = self.train_idx_ray,
                                                                 enc = best_encoder,
                                                                 snp = snp_name))
            elif float32_t(0.0) > self.branch_explainability_threshold and best_encoder != snp_t('additive'):
                ray_jobs.append(ray_utils.ray_snp_encoder.remote(X = self.hub.get_ori_ray_id(snp_name),
                                                                 y = self.all_y_ray_id,
                                                                 train_idx = self.train_idx_ray,
                                                                 enc = best_encoder,
                                                                 snp = snp_name))
        best_encoder_time = time.time() - best_encoder_start
        print(f"    Unseen - Find best encoder & launch encoding: {best_encoder_time:.4f}s ({len(ray_jobs)} encoding jobs)", flush=True)

        # Step 5: Process encoding results
        encoding_start = time.time()
        count = 0
        while len(ray_jobs) > 0:
            # collect results and store encoded snp
            finished, ray_jobs = ray.wait(ray_jobs)
            encoded_snp, snp_name = ray.get(finished)[0]
            snp_perf[snp_name][snp_t('encoded_x')] = encoded_snp
            count += 1
        encoding_time = time.time() - encoding_start
        print(f"    Unseen - Process encoding results: {encoding_time:.4f}s ({count} encodings)", flush=True)

        # Step 6: Update hub with results
        hub_update_start = time.time()
        for snp_name in unseen_branches:
            enc_id = None
            # set enc_id to those snps with r2 / k >= threshold
            if snp_perf[snp_name][snp_perf[snp_name][snp_t('b_encoder')]][snp_t('r2')] / float32_t(self.k) >= self.branch_explainability_threshold:
                enc_id = ray.put(snp_perf[snp_name][snp_t('encoded_x')])
            # or set enc_id to those snps if threshold < 0.0
            elif float32_t(0.0) > self.branch_explainability_threshold and snp_perf[snp_name][snp_t('b_encoder')] != snp_t('additive'):
                enc_id = ray.put(snp_perf[snp_name][snp_t('encoded_x')])

            # Get averaged PAGER LUT if encoding is pager
            pager_lut = None
            if snp_perf[snp_name][snp_t('b_encoder')] == snp_t('pager'):
                pager_lut_cnt = snp_perf[snp_name][snp_t('pager_lut_cnt')]
                if pager_lut_cnt > 0:
                    # Average PAGER LUT values across all k-folds
                    pager_lut = snp_perf[snp_name][snp_t('pager_lut_sum')] / float32_t(pager_lut_cnt)

            self.hub.update_snp_hub_r2_enc(snp=snp_name,
                                          r2=snp_perf[snp_name][snp_perf[snp_name][snp_t('b_encoder')]][snp_t('r2')] / float32_t(self.k),
                                          enc=snp_perf[snp_name][snp_t('b_encoder')],
                                          enc_x=enc_id,
                                          gen_seen=gen_seen,
                                          snp_explainability_threshold=self.branch_explainability_threshold,
                                          pager_lut=pager_lut)
        hub_update_time = time.time() - hub_update_start
        print(f"    Unseen - Update hub: {hub_update_time:.4f}s", flush=True)

        total_unseen_time = time.time() - eval_unseen_total_start
        print(f"    Unseen - TOTAL evaluate_unseen_branches: {total_unseen_time:.4f}s", flush=True)

    def get_sampling(self, cnt:uint16_t, chrom_num:uint16_t) -> npt.NDArray[uint16_t]:
        """
        Function to get the sampling list that evenly splits the number of snps to sample from each chromosome.
        Note that the chrom_num index is mapped to the chromosome keys provided by the hub.

        Parameters:
            cnt (uint16_t): Total number of snps to sample.
            chrom_num (uint16_t): Number of chromosomes to sample from.

        Returns:
            npt.NDArray[uint16_t]: Array of size chrom_num with the number of snps to sample from each chromosome.
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
        pareto_front_pipelines = []
        for i in self.front_zero_ids:
            pareto_front_pipelines.append(self.population[i])
        print(f"Number of pipelines in Pareto front for post analysis: {len(pareto_front_pipelines)}", flush=True)

        # Sort the Pareto front by feature count (MUST match save_and_plot_pareto_front ordering)
        pareto_front_pipelines = sorted(pareto_front_pipelines, key=lambda x: x.get_trait_feature_cnt())

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
                                                                   pop_id=uint32_t(pipeline_id)))
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

        ################# SAVE PARETO FRONT PIPELINES TO CSV #################
        # Create pareto_front_pipelines.csv with validation R2
        pareto_data = []
        for pid, data in pareto_validation_r2.items():
            pareto_data.append({
                'Pipeline ID': pid + 1,  # Start from 1 instead of 0
                'Cross-validated Train R2': data['train_r2'],
                'Validation R2': data['validation_r2'],
                'Feature Count': data['feature_cnt'],
                'Selector': data['selector'],
                'Selector Params': data['selector_params'],
                'Feature Set': ';'.join(sorted(data['feature_set']))  # Feature Set at the end
            })
        pareto_df = pd.DataFrame(pareto_data)
        pareto_df.to_csv(os.path.join(self.save_directory, 'pareto_front_pipelines.csv'), index=False)
        print("Pareto front pipelines saved to pareto_front_pipelines.csv", flush=True)

        ################# PLOT VALIDATION R2 VS FEATURE COUNT #################
        # Extract data for plotting
        feature_counts = [data['feature_cnt'] for data in pareto_validation_r2.values()]
        validation_r2s = [data['validation_r2'] for data in pareto_validation_r2.values()]
        pipeline_ids = list(pareto_validation_r2.keys())

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot all pipelines
        colors = ['red' if pid == utopia_point_pipeline_id else 'blue' for pid in pipeline_ids]
        plt.scatter(feature_counts, validation_r2s, c=colors, s=100, alpha=0.6, edgecolors='black', linewidth=1.5)

        # Annotate each point with pipeline ID (shifted by +1 to start from 1)
        for i, pid in enumerate(pipeline_ids):
            plt.annotate(str(pid + 1), (feature_counts[i], validation_r2s[i]),
                        textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9)

        # Set x-axis to integer increments (automatically scaled based on data range)
        min_features = min(feature_counts)
        max_features = max(feature_counts)
        feature_range = max_features - min_features

        # Determine appropriate step size based on range
        if feature_range <= 10:
            step = 1
        elif feature_range <= 20:
            step = 2
        elif feature_range <= 50:
            step = 5
        elif feature_range <= 100:
            step = 10
        else:
            step = 20

        # Create tick positions starting from 0 or nearest multiple
        x_min = int(min_features // step) * step
        x_max = int(max_features // step + 1) * step
        x_ticks = np.arange(x_min, x_max + step, step)
        plt.xticks(x_ticks)

        plt.xlabel('Feature Count', fontsize=12)
        plt.ylabel('Validation R²', fontsize=12)
        plt.title('Pareto Front: Validation R² vs Feature Count', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', edgecolor='black', label='Pareto Front Pipeline'),
            Patch(facecolor='red', edgecolor='black', label='Utopia Point Pipeline')
        ]
        plt.legend(handles=legend_elements, loc='best')

        # Save the plot
        plt.tight_layout()
        plt.savefig(self.save_directory + 'pareto_validation_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Pareto validation plot saved to {self.save_directory}pareto_validation_plot.png", flush=True)

        # send snps of pipeline with utopia point to final test
        self.final_pipeline_test(pareto_validation_r2[utopia_point_pipeline_id],
                                 'utopia_pipeline_test_results.csv',
                                 pareto_validation_r2[utopia_point_pipeline_id]['train_r2'],
                                 pareto_validation_r2[utopia_point_pipeline_id]['validation_r2'],
                                 pareto_validation_r2[utopia_point_pipeline_id]['feature_cnt'])
        return

    def final_pipeline_test(self, pipeline_data: Dict, file_name: str, train_r2: float32_t, validation_r2: float32_t, size: int16_t) -> None:
        """
        Function to perform the final assessment on the test dataset.
        Combines training and validation datasets to fit a linear regression model.
        Then, evaluates the model on the test dataset and computes PFI using ray_pfi.
        Saves results to a CSV file.

        Args:
            pipeline_data (Dict): Dictionary containing pipeline information including 'feature_set'.
            file_name (str): Name of the output CSV file to save results.
            train_r2 (float32_t): Cross-validated training R² score.
            validation_r2 (float32_t): Validation R² score.
            size (int16_t): Number of features in the pipeline.

        """
        snp_names = list(pipeline_data['feature_set'])  # Get features from pipeline_data
        print(f"Performing final assessment on {len(snp_names)} SNPs...", flush=True)

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
            pop_id=uint32_t(0)
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
            pop_id=uint32_t(0)
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
            pop_id=uint32_t(0)
        )

        pfi_results, _ = ray.get(pfi_job)

        print(f"PFI calculated for {len(pfi_results)} features", flush=True)

        # Create DataFrame with PFI results
        pfi_df = pd.DataFrame(list(pfi_results.items()), columns=['SNP', 'Importance'])
        pfi_df = pfi_df.sort_values(by='Importance', ascending=False)
        pfi_df['Cross-validated Train R2'] = train_r2
        pfi_df['Validation R2'] = validation_r2
        pfi_df['Train+Valid R2'] = train_val_r2
        pfi_df['Test R2'] = test_r2
        pfi_df['Model Size'] = size

        # Save to CSV
        output_path = os.path.join(self.save_directory, file_name)
        pfi_df.to_csv(output_path, index=False)
        print(f"Results saved to {file_name}", flush=True)

    def save_and_plot_pareto_front(self):
        """
        Function to save and plot the Pareto front at the end of evolution.
        Saves the Pareto front pipelines to a CSV file and generates a plot.
        """

        # get the front 0 pipelines
        pareto_front_pipelines = []
        for i in self.front_zero_ids:
            pareto_front_pipelines.append(self.population[i])
        print(f"Number of pipelines in Pareto front at the end of evolution:{len(pareto_front_pipelines)}", flush=True)
        # sort the Pareto front by feature count
        pareto_front_pipelines = sorted(pareto_front_pipelines, key=lambda x: x.get_trait_feature_cnt())

        # plot the Pareto front
        plt.figure(figsize=(10, 6))
        plt.title('Pareto Front: Cross-validated R² vs Feature Count', fontsize=14)
        plt.xlabel('Feature Count', fontsize=12)
        plt.ylabel('Cross-validated R²', fontsize=12)
        plt.grid(True, alpha=0.3)

        # plot pareto front pipelines as blue dots
        pareto_r2 = [pipeline.get_trait_r2() for pipeline in pareto_front_pipelines]
        pareto_feat_cnt = [pipeline.get_trait_feature_cnt() for pipeline in pareto_front_pipelines]
        plt.scatter(pareto_feat_cnt, pareto_r2, color='blue', s=100, alpha=0.6, edgecolors='black', linewidth=1.5, label='Pareto Front')

        # Annotate the points with pipeline numbers (indexes in pareto front)
        for i, (feature_count, r2_score) in enumerate(zip(pareto_feat_cnt, pareto_r2)):
            plt.annotate(
                str(i + 1),  # Text label (pipeline ID starting from 1)
                (feature_count, r2_score),  # The point where the annotation should be
                textcoords="offset points",  # Use offset for better readability
                xytext=(0, 5),  # Offset position above the point
                ha='center',  # Horizontal alignment
                fontsize=9
            )

        # Set x-axis to integer increments (automatically scaled based on data range)
        min_features = min(pareto_feat_cnt)
        max_features = max(pareto_feat_cnt)
        feature_range = max_features - min_features

        # Determine appropriate step size based on range
        if feature_range <= 10:
            step = 1
        elif feature_range <= 20:
            step = 2
        elif feature_range <= 50:
            step = 5
        elif feature_range <= 100:
            step = 10
        else:
            step = 20

        # Create tick positions starting from 0 or nearest multiple
        x_min = int(min_features // step) * step
        x_max = int(max_features // step + 1) * step
        x_ticks = np.arange(x_min, x_max + step, step)
        plt.xticks(x_ticks)

        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_directory, 'pareto_front_plot.png'), dpi=300, bbox_inches='tight')
        print("Pareto front plot saved to pareto_front_plot.png", flush=True)
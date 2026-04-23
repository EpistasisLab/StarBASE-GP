
# import base EA class and types
from ..Base.evovler import EA
from ..Base.types import (float32_t, int16_t, prob_t, int32_t, snp_t, uint16_t, uint32_t, interaction_t)
from ..Base.pipeline import Pipeline
from ..Base.utils import snp_chrm_pos
from ..Base import nsga_tool as nsga
from ..Base.selectors import OLSRegressor

# import K1 specific classes
from .epi_hub import K2_Hub
from . import ray_utils
from .reproduction import K2_Reproduction

# import other necessary libraries
import numpy as np
from typeguard import typechecked
from typing import List, Dict, Set, Tuple
import pandas as pd
import os
import ray
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import time

@typechecked
class K2_Evolver(EA):
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
                 mut_ran_p: prob_t = prob_t(0.3), # probability of random mutation
                 mut_neighbor_p: prob_t = prob_t(.5),
                 m_keep_left: prob_t = prob_t(.33),
                 m_keep_right: prob_t = prob_t(.33),
                 m_climb_both: prob_t = prob_t(.33),
                 mut_ioc_p: prob_t = prob_t(.2), # probability of interaction out of chromosome mutation
                 m_in_win_p: prob_t = prob_t(0.0), # probability for smart in window mutation
                 m_out_win_p: prob_t = prob_t(.5), # probability for smart out window mutation
                 m_out_chr_p: prob_t = prob_t(.5), # probability for smart out of chromosome mutation
                 save_directory: str = "./", # directory to save results in
                 window_distance: int32_t = int32_t(1000000),
                 branch_explainability_threshold: float32_t = float32_t(0.0),
                 phantom_epistasis_threshold: float32_t = float32_t(0.0004),
                 ld_flag: bool = True,
                 encoding_flag: bool = True,
                 regression: bool = True,
                 starting_snps_csv_path: str | None = None, # optional path to csv containing starting snps for the initial population (with column name 'snp')
                 branch_batch_eval_size: int32_t = int32_t(400),
                 pipeline_batch_eval_size: int32_t = int32_t(1000),
                 l1_wt: float = 0.0 # L1 weight for elastic net regularization (0.0 = Ridge, 1.0 = Lasso)
                 ) -> None:
        """
        K2 Evolver class that extends the EA base class.
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
                         mut_ran_p=mut_ran_p,
                         m_in_win_p=m_in_win_p,
                         m_out_win_p=m_out_win_p,
                         m_out_chr_p=m_out_chr_p,
                         save_directory=save_directory,
                         window_distance=window_distance,
                         branch_explainability_threshold=branch_explainability_threshold,
                         ld_flag=ld_flag,
                         branch_batch_eval_size=branch_batch_eval_size,
                         pipeline_batch_eval_size=pipeline_batch_eval_size)
        self.regression = regression
        self.encoding_flag = encoding_flag
        self.phantom_epistasis_threshold = phantom_epistasis_threshold
        self.starting_snps_csv_path = starting_snps_csv_path
        self.l1_wt = l1_wt

        # initialize reproduction class
        self.reproduction = K2_Reproduction(branch_max=self.branch_max,
                                           branch_min=self.branch_min,
                                           mut_prob=self.mut_prob,
                                           cross_prob=self.cross_prob,
                                           mut_selector_p=self.mut_selector_p,
                                           mut_ld_p=self.mut_ld_p,
                                           mut_ran_p=self.mut_ran_p,
                                           m_in_win_p=self.m_in_win_p,
                                           m_out_win_p=self.m_out_win_p,
                                           m_out_chr_p=self.m_out_chr_p,
                                           window_distance=self.window_distance,
                                           mut_neighbor_p=mut_neighbor_p,
                                           m_keep_left=m_keep_left,
                                           m_keep_right=m_keep_right,
                                           m_climb_both=m_climb_both,
                                           mut_ioc_p=mut_ioc_p)

        return

    def initialize_hubs(self) -> None:
        """
        Initialize the hubs needed for the run.
        These specific hub classes must be implemented in the derived class folders.
        """
        print("[Timing] Initializing hubs...", flush=True)
        hub_init_start = time.time()

        # dictionary of feature names and ray_ids for each specific column put into ray
        ray_put_start = time.time()
        feature_ray_ids = {}
        for feature in self.snp_labels:
            feature_ray_ids[feature] = ray.put(self.all_x[feature].to_numpy(dtype=float32_t))
        self.all_y_ray_id = ray.put(self.all_y)
        ray_put_time = time.time() - ray_put_start

        # initialize the hubs
        hub_create_start = time.time()
        self.hub = K2_Hub(snp_list=self.snp_labels, snps_ray_ids=feature_ray_ids)
        hub_create_time = time.time() - hub_create_start

        total_hub_time = time.time() - hub_init_start

        pct_ray = (ray_put_time / total_hub_time * 100) if total_hub_time > 0 else 0
        pct_hub = (hub_create_time / total_hub_time * 100) if total_hub_time > 0 else 0

        print(f"\n[Timing] Hub initialization: {total_hub_time:.2f}s", flush=True)
        print(f"  - Ray put ops:   {ray_put_time:6.2f}s ({pct_ray:5.1f}%)", flush=True)
        print(f"  - Hub creation:  {hub_create_time:6.2f}s ({pct_hub:5.1f}%)\n", flush=True)
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
        self.hub.seen_interactions_count()
        print('', flush=True)
        # list to store the generation details - front zero size, still consider interaction set size, number of snps pruned
        generation_details = []
        # create the initial population
        print('Initializing population...', flush=True)
        start_time = time.time()
        self.initialize_population()
        print(f"Population initialized in {(time.time() - start_time) / 60 / 60} hours", flush=True)
        print('Entering evolutionary proccess.\n', flush=True)

        # run the algorithm for the specified number of generations
        for g in range(gens):
            print('='*80, flush=True)
            print(f'[Timing] Generation {g} starting...', flush=True)
            gen_start_time = time.time()

            # how many pipelines are in the population
            print('Population size:', len(self.population), flush=True)
            assert(0 < len(self.population) <= self.pop_size)

            # get the size of the front 0 after each generation
            pareto_start = time.time()
            count = 0
            if len(self.population) >= 2:
                _, rank = nsga.non_dominated_sorting(obj_scores=self.get_pipeline_scores(pipelines=self.population, weights=(float32_t(1.0), int32_t(-1))))
                count = 0
                for r in rank:
                    if r == 0:
                        count += 1
            else:
                count = 1
            pareto_time = time.time() - pareto_start
            print(f'Size of Pareto Front: {count} (computed in {pareto_time:.4f}s)', flush=True)
            self.hub.seen_interactions_count()  # count the number of seen interactions after each generation

            # Initialize generation stats dictionary (will be updated after evaluation)
            gen_stats = {
                'generation': g,
                'front_zero_size': count,
                'interactions_seen_so_far': self.hub.get_epi_db_size()
            }

            # get order of mutation/crossover to do with the extra offspring
            selection_start = time.time()
            var_order = None
            parent_cnt = None
            parent_ids = None
            if len(self.population) == 1:
                var_order, parent_cnt = [snp_t('m')] * uint32_t(2*self.pop_size), uint32_t(2*self.pop_size)
                parent_ids = [uint32_t(0) for _ in range(2*self.pop_size)]
            else:
                var_order, parent_cnt = self.reproduction.variation_order(self.rng, uint32_t(2*self.pop_size))
                parent_ids = self.parent_selection(parent_cnt)
            selection_time = time.time() - selection_start
            print(f"[Timing] Parent selection: {selection_time:.4f}s", flush=True)

            # generate offspring
            repro_start = time.time()
            offspring = self.reproduction.produce_offspring(rng = self.rng,
                                                           hub = self.hub,
                                                           offspring_cnt = uint32_t(2*self.pop_size),
                                                           parent_ids = parent_ids,
                                                           population = self.population,
                                                           order = var_order)
            repro_time = time.time() - repro_start
            print(f"[Timing] Offspring generation (crossover/mutation): {repro_time:.4f}s", flush=True)
            # make sure we have the correct number of offspring solutions
            assert 0 < len(offspring) <= 2 * self.pop_size

            # process offspring: evaluation interactions and remove bad interactions
            process_start = time.time()
            offspring = self.process_offspring(offspring, int16_t(g))
            process_time = time.time() - process_start
            print(f"[Timing] Process offspring (unseen SNPs + filtering): {process_time:.4f}s", flush=True)

            # evaluate the offspring
            print('Evaluating offspring pipelines...', flush=True)
            eval_start = time.time()

            offspring, eval_stats = self.evaluation(offspring, gen_info=int16_t(g))

            eval_time = time.time() - eval_start
            print(f"[Timing] Offspring evaluation (LD+FS+R2): {eval_time:.4f}s", flush=True)

            # must be less than or equal because of potential negative r2 offspring pipelines
            assert (0 < len(offspring) + len(self.population) <= 3 * self.pop_size)

            # survival selection
            survival_start = time.time()
            self.population = self.survival_selection(offspring)
            survival_time = time.time() - survival_start
            print(f"[Timing] Survival selection: {survival_time:.4f}s", flush=True)

            # make sure we have the correct number of pipelines
            assert len(self.population) <= self.pop_size

            # Calculate total generation time and percentages
            gen_time = time.time() - gen_start_time
            gen_time_mins = gen_time / 60

            pct_selection = (selection_time / gen_time * 100) if gen_time > 0 else 0
            pct_repro = (repro_time / gen_time * 100) if gen_time > 0 else 0
            pct_process = (process_time / gen_time * 100) if gen_time > 0 else 0
            pct_eval = (eval_time / gen_time * 100) if gen_time > 0 else 0
            pct_survival = (survival_time / gen_time * 100) if gen_time > 0 else 0

            print(f"\n[Timing] Generation {g} completed: {gen_time:.2f}s ({gen_time_mins:.2f} mins)", flush=True)
            print(f"  - Selection:    {selection_time:6.2f}s ({pct_selection:5.1f}%)", flush=True)
            print(f"  - Reproduction: {repro_time:6.2f}s ({pct_repro:5.1f}%)", flush=True)
            print(f"  - Processing:   {process_time:6.2f}s ({pct_process:5.1f}%)", flush=True)
            print(f"  - Evaluation:   {eval_time:6.2f}s ({pct_eval:5.1f}%)", flush=True)
            print(f"  - Survival:     {survival_time:6.2f}s ({pct_survival:5.1f}%)", flush=True)

            # Add evaluation stats and timing to generation details
            gen_stats.update({
                'total_time_mins': gen_time_mins,
                'selection_time_s': selection_time,
                'reproduction_time_s': repro_time,
                'processing_time_s': process_time,
                'evaluation_time_s': eval_time,
                'survival_time_s': survival_time,
                'fs_only_count': eval_stats['fs_only_count'],
                'pipelines_evaluated': eval_stats['pipelines_evaluated']
            })
            generation_details.append(gen_stats)

            print('='*80, flush=True)

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

        print("[Timing] Initializing population...", flush=True)
        pop_init_start = time.time()

        # quick check to make sure hub is initialized
        assert self.hub is not None, "Hub must be initialized before initializing population."
        assert len(self.population) == 0, "Population must be empty before initializing."

        # container for all pipeline set of branches
        pop_branch_sets = []
        # container to hold newly found branches to avoid duplicate work
        unseen_branches = set()

        # create initial set of branch sets to integrate within pipelines
        sampling_start = time.time()

        if self.starting_snps_csv_path == None:
            print('No starting SNPs CSV provided, initializing population with random branches.', flush=True)
            while len(pop_branch_sets) < self.pop_size:
                # current set of branches - set of tuples where each tuple is (snp1, snp2) for an interaction branch
                branches = set()

                while len(branches) < self.branch_max:
                    # sample a random interaction from the hub
                    interaction = self.hub.get_ran_interaction(self.rng)
                    # add the interaction to the branch set
                    branches.add(interaction)

                assert len(branches) == self.branch_max, "Number of branches in initial pipeline does not match branch_max."

                # update unseen branches with all new branches
                # bc all branches are new at this point, we can just add them directly
                unseen_branches.update(branches)
                # add the current branch set to the population list
                pop_branch_sets.append(branches)
        else:
            print('Starting SNPs CSV provided, initializing population with branches from CSV.', flush=True)

            # assert to make sure the csv file exists
            assert os.path.exists(self.starting_snps_csv_path), f"Starting SNPs CSV file not found at {self.starting_snps_csv_path}"

            # read in the csv file and extract the 'SNP_Base' and 'weight' columns as lists
            starting_snps_df = pd.read_csv(self.starting_snps_csv_path)
            assert 'SNP_Base' in starting_snps_df.columns, "Starting SNPs CSV must contain 'SNP_Base' column."
            assert 'weight' in starting_snps_df.columns, "Starting SNPs CSV must contain 'weight' column."

            snp_bases = starting_snps_df['SNP_Base'].tolist()
            weights = starting_snps_df['weight'].tolist()

            # normalize the weights to sum to 1
            total_weight = sum(weights)
            weights = [weight / total_weight for weight in weights]

            while len(pop_branch_sets) < self.pop_size:
                # current set of branches - set of tuples where each tuple is (snp1, snp2) for an interaction branch
                branches = set()

                while len(branches) < self.branch_max:
                    # sample a two random interactions using numpy without replacement based on the weights provided in the csv file
                    snp1, snp2 = self.rng.choice(snp_bases, size=2, replace=False, p=weights)
                    snp1 = snp_t(snp1[3:])
                    snp2 = snp_t(snp2[3:])
                    # create interaction
                    interaction = (snp1, snp2) if snp1 < snp2 else (snp2, snp1)
                    # add the interaction to the branch set
                    branches.add(interaction)

                assert len(branches) == self.branch_max, "Number of branches in initial pipeline does not match branch_max."

                # update unseen branches with all new branches
                unseen_branches.update(branches)
                # add the current branch set to the population list
                pop_branch_sets.append(branches)

        sampling_time = time.time() - sampling_start

        # break up unseen_branches into chunks of self.branch_batch_eval_size to avoid ray overload and then run evaluate_unseen_branches on each chunk
        eval_unseen_start = time.time()
        unseen_branches_list = list(unseen_branches)
        for i in range(0, len(unseen_branches_list), self.branch_batch_eval_size):
            print(f"Evaluating unseen branches chunk {i // self.branch_batch_eval_size + 1} / {(len(unseen_branches_list) - 1) // self.branch_batch_eval_size + 1}", flush=True)
            chunk = set(unseen_branches_list[i:i+self.branch_batch_eval_size])
            self.evaluate_unseen_branches(chunk, gen_seen=int16_t(0))
        eval_unseen_time = time.time() - eval_unseen_start

        # remove inactive branches from each branch set for the initial population
        filter_start = time.time()
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
        filter_time = time.time() - filter_start
        assert 1 <= len(self.population) <= self.pop_size, "Population size contained no valid pipelines after branch set evaluation."

        # evaluate the initial population
        print('Evaluating initial population pipelines...', flush=True)
        eval_pop_start = time.time()
        self.population, _ = self.evaluation(self.population, gen_info=int16_t(0))
        eval_pop_time = time.time() - eval_pop_start
        assert 1 <= len(self.population) <= self.pop_size, "Population size contained no valid pipelines after pipeline evaluation."

        total_pop_init = time.time() - pop_init_start

        pct_sampling = (sampling_time / total_pop_init * 100) if total_pop_init > 0 else 0
        pct_eval_unseen = (eval_unseen_time / total_pop_init * 100) if total_pop_init > 0 else 0
        pct_filter = (filter_time / total_pop_init * 100) if total_pop_init > 0 else 0
        pct_eval_pop = (eval_pop_time / total_pop_init * 100) if total_pop_init > 0 else 0

        print(f"\n[Timing] Population initialization: {total_pop_init:.2f}s ({total_pop_init/60:.2f} mins)", flush=True)
        print(f"  - Sampling:       {sampling_time:6.2f}s ({pct_sampling:5.1f}%)", flush=True)
        print(f"  - Eval unseen:    {eval_unseen_time:6.2f}s ({pct_eval_unseen:5.1f}%)", flush=True)
        print(f"  - Filtering:      {filter_time:6.2f}s ({pct_filter:5.1f}%)", flush=True)
        print(f"  - Eval population:{eval_pop_time:6.2f}s ({pct_eval_pop:5.1f}%)\n", flush=True)
        return

    def evaluation(self, pipelines: List[Pipeline], gen_info: int16_t) -> Tuple[List[Pipeline], Dict]:
        """
        Function to evaluate pipelines.
        All of this should be done in asyncronous parallel jobs for maximum efficiency.
        Processes pipelines in batches of 1000 to avoid ray overload.

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

        print(f"[Timing] Starting evaluation of {len(pipelines)} pipelines...", flush=True)
        eval_method_start = time.time()

        # keep a count of number of pipelines that are calling only fs vs ld+fs
        fs_only_count = 0

        # Global data structures to accumulate results across batches
        pipeline_evaluation_details = {}
        pruned_interactions = set()
        interactions_details_per_interaction = {}

        # Batch size for processing
        num_batches = (len(pipelines) - 1) // self.pipeline_batch_eval_size + 1

        # Timing accumulators
        total_job_creation_time = 0.0
        total_ld_fs_time = 0.0
        total_r2_job_time = 0.0
        total_r2_eval_time = 0.0

        # Process pipelines in batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.pipeline_batch_eval_size
            end_idx = min(start_idx + self.pipeline_batch_eval_size, len(pipelines))
            batch = pipelines[start_idx:end_idx]

            print(f"  Processing batch {batch_idx + 1}/{num_batches} (pipelines {start_idx} to {end_idx - 1})...", flush=True)

            # create ray jobs for each pipeline evaluation depending on if ld is needed or not
            job_creation_start = time.time()
            ray_jobs = []
            for i, pipeline in enumerate(batch):
                global_id = start_idx + i
                pipeline_evaluation_details[global_id] = {snp_t('r2'): float32_t(0.0), snp_t('feature_cnt'): None, snp_t('features'): None,
                                                          snp_t('ld_used'): False, snp_t('error'): False, snp_t('count'): uint16_t(0)}
                # Check if LD pruning should be applied:
                # 1. ld_flag must be True
                # 2. Pipeline must contain SNPs from the same chromosome
                if self.ld_flag and self.interactions_on_same_hyperchromosome(pipeline.get_branch_set()):
                    ray_jobs.append(ray_utils.ray_eval_pipeline_ld_fs.remote(component_map=self.hub.build_component_map(pipeline.get_branch_set()),
                                                                             y_train=self.all_y_ray_id,
                                                                             train_idx=self.train_idx_ray,
                                                                             selector_node=pipeline.get_selector_node(),
                                                                             ld_node=pipeline.get_ld_node(),
                                                                             pop_id=uint32_t(global_id),
                                                                             interaction_r2_set=self.hub.generate_r2_set(pipeline.get_branch_set())))
                    pipeline_evaluation_details[global_id][snp_t('ld_used')] = True

                # else, no need for ld pruner (either ld_flag is False or interactions are not on same hyperchromosome)
                else:
                    ray_jobs.append(ray_utils.ray_eval_pipeline_fs.remote(component_map=self.hub.build_component_map(pipeline.get_branch_set()),
                                                                         y_train=self.all_y_ray_id,
                                                                         train_idx=self.train_idx_ray,
                                                                         selector_node=pipeline.get_selector_node(),
                                                                         pop_id=uint32_t(global_id)))
                    fs_only_count += 1
            job_creation_time = time.time() - job_creation_start
            total_job_creation_time += job_creation_time

            # process LD/FS results as they come in
            ld_fs_start = time.time()
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
                    # update the pruned interactions based on the interaction details after LD
                    for interaction, details in ld_details.items():
                        if details['pruned'] == True:
                            # Convert interaction tuple to use snp_t types (Ray returns regular strings)
                            interaction_typed = (snp_t(interaction[0]), snp_t(interaction[1]))
                            pruned_interactions.add(interaction_typed)
                            interactions_details_per_interaction[interaction_typed] = details

            ld_fs_time = time.time() - ld_fs_start
            total_ld_fs_time += ld_fs_time

            # send pipelines with no error to be evaluated for r2 across k-folds (only pipelines with error == False)
            r2_job_start = time.time()
            ray_jobs = []
            for i in range(start_idx, end_idx):
                if pipeline_evaluation_details[i][snp_t('error')]:
                    continue

                # create a ray job for each of the folds
                for _, fold_data in self.train_fold_dict_ray.items():
                    feature = set()
                    for f in pipeline_evaluation_details[i][snp_t('features')]:
                        feature.add((snp_t(f[0]), snp_t(f[1])))

                    ray_jobs.append(ray_utils.ray_eval_pipeline_r2.remote(component_map=self.hub.build_component_map(feature),
                                                                              y = self.all_y_ray_id,
                                                                              train_idx = fold_data['train_idx'],
                                                                              valid_idx = fold_data['val_idx'],
                                                                              pop_id = uint32_t(i),
                                                                              l1_wt = self.l1_wt))
            r2_job_time = time.time() - r2_job_start
            total_r2_job_time += r2_job_time

            # process R2 results as they come in
            r2_eval_start = time.time()
            while len(ray_jobs) > 0:
                finished, ray_jobs = ray.wait(ray_jobs)
                r2, pop_id, error, _, _, _, _, _, _ = ray.get(finished[0])
                if error < float32_t(0.0):
                    pipeline_evaluation_details[pop_id][snp_t('error')] = True

                # update r2 and count
                pipeline_evaluation_details[pop_id][snp_t('r2')] += r2
                pipeline_evaluation_details[pop_id][snp_t('count')] += uint32_t(1)
            r2_eval_time = time.time() - r2_eval_start
            total_r2_eval_time += r2_eval_time

        # if self.ld_flag is False, pruned_interactions should be empty
        assert (len(pruned_interactions) == 0) if self.ld_flag == False else True, "Pruned interactions should be empty when LD flag is False."
        print(f"  - Total LD/FS processing: {total_ld_fs_time:.4f}s ({total_ld_fs_time/60:.2f} mins), pruned {len(pruned_interactions)} interactions", flush=True)

        # update hubs with prunned interactions info
        hub_update_start = time.time()
        self.hub.process_pruned_interactions(pruned_interactions, interactions_details_per_interaction, gen_info)
        hub_update_time = time.time() - hub_update_start
        print(f"  - Hub pruned interaction updates: {hub_update_time:.4f}s", flush=True)

        # print all the pipeline evaluation details for this generation
        for pipeline_id in pipeline_evaluation_details:
            print(f"Pipeline {pipeline_id} evaluation details: R2={pipeline_evaluation_details[pipeline_id][snp_t('r2')]/float32_t(self.k):.4f}, \
                Feature Count={pipeline_evaluation_details[pipeline_id][snp_t('feature_cnt')]}, \
                    LD Used={pipeline_evaluation_details[pipeline_id][snp_t('ld_used')]}, \
                        Interactions {pipeline_evaluation_details[pipeline_id][snp_t('features')]}", flush=True)

        # will hold the evaluated pipelines that passed evaluation
        evaluated_pipelines : List[Pipeline] = []

        # update pipelines with evaluation results
        for pipeline_id in pipeline_evaluation_details:
            # print(f"Final evaluation for Pipeline {pipeline_id}: R2={pipeline_evaluation_details[pipeline_id][snp_t('r2')]:.4f}, \
            #     Feature Count={pipeline_evaluation_details[pipeline_id][snp_t('feature_cnt')]}, \
            #         LD Used={pipeline_evaluation_details[pipeline_id][snp_t('ld_used')]}, \
            #             Interactions {pipeline_evaluation_details[pipeline_id][snp_t('features')]}", flush=True)

            # skip pipelines with error, negative r2, or all snps are inactive
            if pipeline_evaluation_details[pipeline_id][snp_t('error')] or \
                pipeline_evaluation_details[pipeline_id][snp_t('r2')] <= float32_t(0.0):
                continue

            # clean up the interaction features list to be tuple of np.str_
            raw_features = pipeline_evaluation_details[pipeline_id][snp_t('features')]
            typed_features = []
            for feature in raw_features:
                if isinstance(feature, tuple):
                    assert len(feature) == 2, "Interaction feature tuple must have length 2."
                    typed_features.append(tuple(snp_t(f) for f in feature))
                elif isinstance(feature, np.str_):
                    typed_features.append(snp_t(str(feature)))
                else:
                    raise ValueError(f"Unexpected feature type: {type(feature)} for feature {feature}")

            # check if at least one interaction is active
            if not self.hub.at_least_one_active_interaction(typed_features):
                continue

            assert pipeline_evaluation_details[pipeline_id][snp_t('count')] == uint16_t(self.k), "Pipeline evaluation must have k-fold evaluations."
            pipelines[pipeline_id].set_traits([ pipeline_evaluation_details[pipeline_id][snp_t('r2')] / float32_t(self.k),
                                                pipeline_evaluation_details[pipeline_id][snp_t('feature_cnt')],
                                                set(typed_features) ])
            # add to evaluated pipelines
            evaluated_pipelines.append(pipelines[pipeline_id])

        total_eval_time = time.time() - eval_method_start

        # Calculate percentages
        pct_job_create = (total_job_creation_time / total_eval_time * 100) if total_eval_time > 0 else 0
        pct_ld_fs = (total_ld_fs_time / total_eval_time * 100) if total_eval_time > 0 else 0
        pct_r2_job = (total_r2_job_time / total_eval_time * 100) if total_eval_time > 0 else 0
        pct_r2_eval = (total_r2_eval_time / total_eval_time * 100) if total_eval_time > 0 else 0
        pct_hub_update = (hub_update_time / total_eval_time * 100) if total_eval_time > 0 else 0

        print(f"\n[Timing] Evaluation ({len(pipelines)} pipelines): {total_eval_time:.2f}s ({total_eval_time/60:.2f} mins)", flush=True)
        print(f"  - Job creation:  {total_job_creation_time:6.2f}s ({pct_job_create:5.1f}%)", flush=True)
        print(f"  - LD/FS process: {total_ld_fs_time:6.2f}s ({pct_ld_fs:5.1f}%)", flush=True)
        print(f"  - R2 jobs:       {total_r2_job_time:6.2f}s ({pct_r2_job:5.1f}%)", flush=True)
        print(f"  - R2 evaluation: {total_r2_eval_time:6.2f}s ({pct_r2_eval:5.1f}%)", flush=True)
        print(f"  - Hub updates:   {hub_update_time:6.2f}s ({pct_hub_update:5.1f}%)\n", flush=True)

        return evaluated_pipelines, {'fs_only_count': fs_only_count, 'pipelines_evaluated': len(evaluated_pipelines)}

    # function to check if a branch set have interactions in the same hyperchromosome (that SNP 1 and SNP3 are on the same chromosome and SNP2 and SNP4 are on the same chromosome) - if so, we can apply LD pruning, if not, we skip LD pruning and just evaluate with FS
    def interactions_on_same_hyperchromosome(self, branch_set: Set[interaction_t]) -> bool:
        """
        Function to check if a branch set has interactions that share the same hyperchromosome.
        This is determined by checking if at least 2 interactions involve the same pair of chromosomes.
        LD pruning is useful when multiple interactions share the same hyperchromosome, as their
        component SNPs may be in linkage disequilibrium.

        Args:
            branch_set (Set[interaction_t]): Set of interactions (tuples of SNP pairs) to check.
        Returns:
            bool: True if at least 2 interactions share the same hyperchromosome, False otherwise.
        """
        hyperchromosome_count = {} # count of interactions per hyperchromosome
        for interaction in branch_set:
            snp1, snp2 = interaction # unpack the interaction tuple (snp1, snp2)
            snp1_chrom, _ = snp_chrm_pos(snp1)
            snp2_chrom, _ = snp_chrm_pos(snp2)
            hc = (snp1_chrom, snp2_chrom) if snp1_chrom <= snp2_chrom else (snp2_chrom, snp1_chrom) # create a hyperchromosome tuple with ordered chromosome names
            hyperchromosome_count[hc] = hyperchromosome_count.get(hc, 0) + 1

        # Return True if any hyperchromosome has 2 or more interactions
        # (those interactions could be in LD and should be pruned)
        return any(count >= 2 for count in hyperchromosome_count.values())

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

        # quick checks
        assert len(pipelines) > 0, "No pipelines to process."
        assert len(pipelines) <= self.pop_size * 2, "Number of pipelines exceeds maximum offspring size."
        assert gen_info >= 0, "Generation info must be non-negative."

        print('Processing offspring pipelines: evaluating unseen branches and removing inactive branches...', flush=True)
        # collect all interactions from each pipeline and send to hub to find unseen interactions
        all_interactions = set()
        for pipeline in pipelines:
            all_interactions.update(pipeline.get_branch_set())
        unseen_interactions = self.hub.get_unseen_interactions(all_interactions)

        # evaluate all unseen interactions if we have any to evaluate
        if len(unseen_interactions) > 0:
            # break up unseen_branches into chunks of self.branch_batch_eval_size to avoid ray overload and then run evaluate_unseen_branches on each chunk
            unseen_branches_list = list(unseen_interactions)
            for i in range(0, len(unseen_branches_list), self.branch_batch_eval_size):
                print(f"Evaluating unseen branches chunk {i // self.branch_batch_eval_size + 1} / {(len(unseen_branches_list) - 1) // self.branch_batch_eval_size + 1}", flush=True)
                chunk = set(unseen_branches_list[i:i+self.branch_batch_eval_size])
                self.evaluate_unseen_branches(chunk, gen_seen=int16_t(gen_info))

        # offspring pipelines with no good interactions
        updated_pipelines = []

        for pipeline in pipelines:
            good_interactions = self.hub.remove_inactive_branches(pipeline.get_branch_set())
            if len(good_interactions) == 0:
                # skip this iteration if there are no good interactions
                continue

            updated_pipelines.append(Pipeline(
                branch_set=good_interactions,
                selector_node=pipeline.get_selector_node(),
                ld_node=pipeline.get_ld_node()
            ))
        return updated_pipelines

    # modified for epistasis - optimized with two-stage evaluation
    def evaluate_unseen_branches(self, unseen_branches: Set[interaction_t], gen_seen: int16_t) -> None:
        """
        Function to evaluate all unseen branches and add their best R2 and Encoder type to the Hub.
        Uses two-stage approach:
        1. Pre-screen interactions (MLG + Pearson correlation) - once per interaction
        2. Evaluate encodings on CV folds - only for interactions that pass pre-screening

        This minimizes Ray overhead by avoiding redundant checks across CV folds.

        Parameters:
            unseen_branches (Set[interaction_t]): Set of unseen branch tuples (SNP1, SNP2).
            gen_seen (int16_t): Generation number when these branches were first seen.
        """

        # quick checks
        assert len(unseen_branches) > 0, "No unseen branches to evaluate."
        assert gen_seen >= 0, "Generation seen must be non-negative."

        print(f"[Timing] Evaluating {len(unseen_branches)} unseen branches...", flush=True)
        unseen_eval_start = time.time()

        # STEP 1: Pre-screen all interactions (MLG + Pearson correlation)
        print("  [Step 1] Pre-screening interactions (MLG + Pearson correlation)...", flush=True)
        prescreen_start = time.time()
        prescreen_jobs = []

        for snp_pair in unseen_branches:
            snp_1, snp_2 = snp_pair
            assert isinstance(snp_1, snp_t), "SNP must be of type snp_t."
            assert isinstance(snp_2, snp_t), "SNP must be of type snp_t."

            X1 = self.hub.get_snp_ori_ray_id(snp_1) # get the original univariate encoding for snp1 from the hub as a Ray object reference
            X2 = self.hub.get_snp_ori_ray_id(snp_2) # get the original univariate encoding for snp2 from the hub as a Ray object reference

            job = ray_utils.ray_prescreen_interaction.remote(
                X1, X2, self.train_idx_ray, snp_1, snp_2
            )
            prescreen_jobs.append((job, snp_pair))

        # Process pre-screening results
        passed_interactions = {}  # snp_pair -> correlation_r2
        failed_interactions = {}  # snp_pair -> (failure_code, correlation_r2)

        while len(prescreen_jobs) > 0:
            done, _ = ray.wait([job[0] for job in prescreen_jobs], num_returns=1) # returns list of ready job references and list of remaining job references
            job_idx = [job[0] for job in prescreen_jobs].index(done[0]) # get the index of the finished job in the prescreen_jobs list
            snp_pair = prescreen_jobs[job_idx][1] # get the corresponding snp_pair for the finished job

            pass_flag, failure_code, correlation_r2 = ray.get(done[0])

            if pass_flag:
                passed_interactions[snp_pair] = correlation_r2
            else:
                failed_interactions[snp_pair] = (failure_code, correlation_r2)

            prescreen_jobs = [prescreen_jobs[i] for i in range(len(prescreen_jobs)) if i != job_idx] # remove the finished job from the list

        prescreen_time = time.time() - prescreen_start
        print(f"    Pre-screening complete: {len(passed_interactions)} passed, {len(failed_interactions)} failed ({prescreen_time:.2f}s)", flush=True)

        # STEP 2: Evaluate encodings for passed interactions
        print("  [Step 2] Evaluating encodings for passed interactions...", flush=True)
        encoding_eval_start = time.time()

        # Initialize results structure
        inter_perf = {} # snp_pair -> dict with keys: cartesian_r2_folds, xor_r2_folds, mdr_r2_folds, mdr_mappings, correlation_r2, failure_code, avg_r2, best_enc
        for snp_pair in unseen_branches:
            inter_perf[snp_pair] = {
                'cartesian_r2_folds': [],
                'xor_r2_folds': [],
                'mdr_r2_folds': [],
                'mdr_mappings': [],
                'correlation_r2': failed_interactions.get(snp_pair, (None, float32_t(-1.0)))[1] if snp_pair in failed_interactions else passed_interactions.get(snp_pair, float32_t(-1.0)),
                'failure_code': failed_interactions.get(snp_pair, (None, None))[0] if snp_pair in failed_interactions else None,
                'avg_r2': float32_t(-1.0),
                'best_enc': None,
                'error': False
            }

        # Create encoding evaluation jobs only for passed interactions
        encoding_jobs = []
        if self.encoding_flag:
            # Evaluate all encodings (cartesian, xor, mdr)
            for snp_pair in passed_interactions:
                snp_1, snp_2 = snp_pair
                X1 = self.hub.get_snp_ori_ray_id(snp_1)
                X2 = self.hub.get_snp_ori_ray_id(snp_2)

                for _, fold_data in self.train_fold_dict_ray.items():
                    job = ray_utils.ray_evaluate_interaction_encodings.remote(
                        X1, X2, self.all_y_ray_id,
                        fold_data['train_idx'], fold_data['val_idx'],
                        snp_1, snp_2
                    )
                    encoding_jobs.append((job, snp_pair))
        else:
            # Evaluate only cartesian encoding (ablation study)
            for snp_pair in passed_interactions:
                snp_1, snp_2 = snp_pair
                X1 = self.hub.get_snp_ori_ray_id(snp_1)
                X2 = self.hub.get_snp_ori_ray_id(snp_2)

                for _, fold_data in self.train_fold_dict_ray.items():
                    job = ray_utils.ray_evaluate_interaction_cartesian.remote(
                        X1, X2, self.all_y_ray_id,
                        fold_data['train_idx'], fold_data['val_idx'],
                        snp_1, snp_2
                    )
                    encoding_jobs.append((job, snp_pair))

        print(f"    Created {len(encoding_jobs)} encoding evaluation jobs ({len(passed_interactions)} interactions × {self.k} folds)", flush=True)

        # Process encoding evaluation results
        while len(encoding_jobs) > 0:
            done, _ = ray.wait([job[0] for job in encoding_jobs], num_returns=1) # returns list of ready job references and list of remaining job references
            job_idx = [job[0] for job in encoding_jobs].index(done[0]) # get the index of the finished job in the encoding_jobs list
            snp_pair = encoding_jobs[job_idx][1] # get the corresponding snp_pair for the finished job

            if self.encoding_flag:
                results, mdr_mapping, error = ray.get(done[0])
                inter_perf[snp_pair]['cartesian_r2_folds'].append(results['cartesian'])
                inter_perf[snp_pair]['xor_r2_folds'].append(results['xor'])
                inter_perf[snp_pair]['mdr_r2_folds'].append(results['mdr'])

                if error['cartesian'] or error['xor'] or error['mdr'] or error['base_model']:
                    inter_perf[snp_pair]['error'] = True

                if mdr_mapping is not None:
                    inter_perf[snp_pair]['mdr_mappings'].append(mdr_mapping) # mdr_mapping is a dict with keys (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0) and values are the corresponding case/control ratios for that genotype combination
            else:
                cartesian_r2, error = ray.get(done[0])
                inter_perf[snp_pair]['cartesian_r2_folds'].append(cartesian_r2)

                if error:
                    inter_perf[snp_pair]['error'] = True

            encoding_jobs = [encoding_jobs[i] for i in range(len(encoding_jobs)) if i != job_idx] # remove the finished job from the list

        encoding_eval_time = time.time() - encoding_eval_start
        print(f"    Encoding evaluation complete ({encoding_eval_time:.2f}s, {encoding_eval_time/60:.2f} mins)", flush=True)

        # STEP 3: Aggregate results and select best encoding
        print("  [Step 3] Aggregating results and selecting best encodings...", flush=True)
        aggregate_start = time.time()

        for snp_pair in passed_interactions:
            if self.encoding_flag:
                # assert that we have k R2 results for each encoding
                assert len(inter_perf[snp_pair]['cartesian_r2_folds']) == self.k, f"Expected {self.k} Cartesian R2 results for {snp_pair}, got {len(inter_perf[snp_pair]['cartesian_r2_folds'])}."
                assert len(inter_perf[snp_pair]['xor_r2_folds']) == self.k, f"Expected {self.k} XOR R2 results for {snp_pair}, got {len(inter_perf[snp_pair]['xor_r2_folds'])}."
                assert len(inter_perf[snp_pair]['mdr_r2_folds']) == self.k, f"Expected {self.k} MDR R2 results for {snp_pair}, got {len(inter_perf[snp_pair]['mdr_r2_folds'])}."

                # Average R2 across folds for each encoding
                avg_cartesian = np.mean(inter_perf[snp_pair]['cartesian_r2_folds'])
                avg_xor = np.mean([r2 for r2 in inter_perf[snp_pair]['xor_r2_folds']])
                avg_mdr = np.mean([r2 for r2 in inter_perf[snp_pair]['mdr_r2_folds']])

                # Select best encoding
                best_r2 = max(avg_cartesian, avg_xor, avg_mdr)
                if best_r2 == avg_cartesian:
                    best_enc = snp_t('cartesian')
                elif best_r2 == avg_xor:
                    best_enc = snp_t('xor')
                else:
                    best_enc = snp_t('mdr')

                inter_perf[snp_pair]['avg_r2'] = float32_t(best_r2)
                inter_perf[snp_pair]['best_enc'] = best_enc
            else:
                # Only cartesian encoding
                # assert that we have k R2 results for cartesian encoding
                assert len(inter_perf[snp_pair]['cartesian_r2_folds']) == self.k, f"Expected {self.k} Cartesian R2 results for {snp_pair}, got {len(inter_perf[snp_pair]['cartesian_r2_folds'])}."
                avg_cartesian = np.mean([r2 for r2 in inter_perf[snp_pair]['cartesian_r2_folds']])
                inter_perf[snp_pair]['avg_r2'] = float32_t(avg_cartesian)
                inter_perf[snp_pair]['best_enc'] = snp_t('cartesian')

            # Check threshold
            if inter_perf[snp_pair]['avg_r2'] <= self.phantom_epistasis_threshold:
                inter_perf[snp_pair]['failure_code'] = float32_t(-3.0)  # Phantom epistasis

        aggregate_time = time.time() - aggregate_start
        print(f"    Aggregation complete ({aggregate_time:.2f}s)", flush=True)

        # STEP 4: Create encoding jobs for interactions above threshold - to store the encoded interactions as ray objects in the hub for future use in pipeline evaluations
        print("  [Step 4] Creating encoding jobs for interactions above threshold...", flush=True)
        encoding_job_start = time.time()

        encoding_jobs = []
        interactions_to_encode = []

        for snp_pair in passed_interactions:
            if inter_perf[snp_pair]['avg_r2'] >= self.branch_explainability_threshold:
                interactions_to_encode.append(snp_pair)
                snp_1, snp_2 = snp_pair
                X1 = self.hub.get_snp_ori_ray_id(snp_1)
                X2 = self.hub.get_snp_ori_ray_id(snp_2)
                best_enc = inter_perf[snp_pair]['best_enc']

                # Create the interaction name with best encoding
                interaction_name = f"{snp_1}_{best_enc}_{snp_2}"

                # Use ray_interaction_encoder to encode the full dataset
                job = ray_utils.ray_interaction_encoder.remote(
                    X1, X2, self.all_y_ray_id,
                    self.train_idx_ray, best_enc, snp_t(interaction_name)
                )
                encoding_jobs.append((job, snp_pair))

        encoding_job_time = time.time() - encoding_job_start
        print(f"    Created {len(encoding_jobs)} encoding jobs for {len(interactions_to_encode)} interactions ({encoding_job_time:.2f}s)", flush=True)

        # Process encoding jobs
        encoding_exec_start = time.time()
        while len(encoding_jobs) > 0:
            done, _ = ray.wait([job[0] for job in encoding_jobs], num_returns=1) # returns list of ready job references and list of remaining job references
            job_idx = [job[0] for job in encoding_jobs].index(done[0]) # get the index of the finished job in the encoding_jobs list
            snp_pair = encoding_jobs[job_idx][1] # get the corresponding snp_pair for the finished job

            encoded_data, _ = ray.get(done[0])
            inter_perf[snp_pair]['encoded_data'] = encoded_data

            encoding_jobs = [encoding_jobs[i] for i in range(len(encoding_jobs)) if i != job_idx]

        encoding_exec_time = time.time() - encoding_exec_start
        print(f"    Encoding execution complete: {len(interactions_to_encode)} interactions encoded ({encoding_exec_time:.2f}s)", flush=True)

        # STEP 5: Update hub with results
        print("  [Step 5] Updating hub with interaction results...", flush=True)
        hub_update_start = time.time()

        for snp_pair in unseen_branches:
            snp_1, snp_2 = snp_pair

            # Put encoded data in Ray store if it exists and above threshold
            encoded_ray_id = None
            if 'encoded_data' in inter_perf[snp_pair] and inter_perf[snp_pair]['avg_r2'] >= self.branch_explainability_threshold:
                encoded_ray_id = ray.put(inter_perf[snp_pair]['encoded_data'])

            # Get MDR mapping for MDR encoding - average across all folds
            mdr_mapping = None
            if inter_perf[snp_pair]['best_enc'] == snp_t('mdr') and len(inter_perf[snp_pair]['mdr_mappings']) > 0:
                # Average MDR mappings across all k-folds
                # MDR mapping is a dict: {(genotype1, genotype2): value}
                all_mappings = inter_perf[snp_pair]['mdr_mappings']
                averaged_mapping = {} # key is a genotype combination tuple (genotype1, genotype2) and value is the average case/control ratio across folds for that genotype combination

                # Get all unique keys across all folds
                all_keys = set()
                for mapping in all_mappings:
                    all_keys.update(mapping.keys())

                # Average the values for each genotype combination
                for key in all_keys:
                    values = [mapping.get(key) for mapping in all_mappings if key in mapping] # only average over folds where this key exists
                    averaged_mapping[key] = np.mean(values)

                mdr_mapping = averaged_mapping

            # Update hub with interaction performance
            # The hub expects the snp_pair tuple and will handle it as an interaction
            # Use pager_lut parameter to pass MDR mapping (reusing existing parameter)
            self.hub.add_interaction_to_hub(
                interaction=snp_pair,
                r2=inter_perf[snp_pair]['avg_r2'],
                enc_rid=encoded_ray_id,
                enc_x=inter_perf[snp_pair]['best_enc'],
                gen_seen=gen_seen,
                mdr_mapping=mdr_mapping
            )

        # update hub with interactions that failed for phantom epistasis (error code -3)
        for snp_pair in inter_perf:
            if inter_perf[snp_pair]['failure_code'] == float32_t(-3.0):
                self.hub.flip_active_flag_pe(snp_pair, gen_seen)

        # update hub wit interactions that filed during preprocessing (failed_interactions)
        for snp_pair in failed_interactions:
            self.hub.flip_active_flag_pre(snp_pair, gen_seen)

        hub_update_time = time.time() - hub_update_start
        print(f"    Hub updates complete ({hub_update_time:.2f}s)", flush=True)

        # Summary of unseen branch evaluation - time breakdown and results
        total_time = time.time() - unseen_eval_start
        print(f"\n[Timing] Total unseen branch evaluation: {total_time:.2f}s ({total_time/60:.2f} mins)", flush=True)
        print(f"  Step 1 (Pre-screen):  {prescreen_time:6.2f}s ({prescreen_time/total_time*100:5.1f}%)", flush=True)
        print(f"  Step 2 (Eval encode): {encoding_eval_time:6.2f}s ({encoding_eval_time/total_time*100:5.1f}%)", flush=True)
        print(f"  Step 3 (Aggregate):   {aggregate_time:6.2f}s ({aggregate_time/total_time*100:5.1f}%)", flush=True)
        print(f"  Step 4 (Encode jobs): {encoding_job_time + encoding_exec_time:6.2f}s ({(encoding_job_time + encoding_exec_time)/total_time*100:5.1f}%)", flush=True)
        print(f"  Step 5 (Hub update):  {hub_update_time:6.2f}s ({hub_update_time/total_time*100:5.1f}%)", flush=True)
        print(f"  Results: {len(passed_interactions)} passed preprocessing, {len(failed_interactions)} failed preprocessing, {len(interactions_to_encode)} above phantom epistasis threshold\n", flush=True)

    def post_analysis_with_good_snps(self):
        """
        Function to perform analysis on all the Pareto front pipelines using the validation set using only the good snps seen during evolution.
        """

        # get the Pareto front pipelines
        pareto_front_pipelines = []
        for i in self.front_zero_ids:
            pareto_front_pipelines.append(self.population[i])
        print(f"Number of pipelines in Pareto front for post analysis: {len(pareto_front_pipelines)}", flush=True)
        print ("The IDs of the pipelines in the Pareto front are:", flush=True)
        print([i for i in self.front_zero_ids], flush=True)
        assert len(pareto_front_pipelines) > 0, "No pipelines in Pareto front for post analysis."
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
            ray_jobs.append(ray_utils.ray_eval_pipeline_r2.remote(component_map=self.hub.build_component_map(pipeline.get_branch_set()),
                                                                      y = self.all_y_ray_id,
                                                                      train_idx = self.train_idx_ray,  # Use all training data for final evaluation
                                                                      valid_idx = self.val_idx_ray,
                                                                      pop_id = uint32_t(pipeline_id),
                                                                      l1_wt = self.l1_wt))


            # process results as they come in
            while len(ray_jobs) > 0:
                finished, ray_jobs = ray.wait(ray_jobs)
                r2, pop_id, error, alpha_base, alpha_joint, base_r2_train, base_r2_valid, joint_r2_train, joint_r2_valid = ray.get(finished)[0]
                if error < float32_t(0.0):
                    print(f"Error during post analysis evaluation of pipeline {pop_id}", flush=True)
                    continue

                # Store all R² values
                pareto_validation_r2[pop_id]['validation_r2'] = float32_t(r2)
                pareto_validation_r2[pop_id]['base_r2_train'] = float32_t(base_r2_train)
                pareto_validation_r2[pop_id]['base_r2_valid'] = float32_t(base_r2_valid)
                pareto_validation_r2[pop_id]['joint_r2_train'] = float32_t(joint_r2_train)
                pareto_validation_r2[pop_id]['joint_r2_valid'] = float32_t(joint_r2_valid)
                pareto_validation_r2[pop_id]['alpha_base'] = alpha_base
                pareto_validation_r2[pop_id]['alpha_joint'] = alpha_joint

                # Compute deltas
                delta_base = base_r2_valid - base_r2_train
                delta_joint = joint_r2_valid - joint_r2_train

                print(f"\nPipeline {pop_id} Validation Results:", flush=True)
                print(f"  Model 0 (Base) - Train R²: {base_r2_train:.6f}, Valid R²: {base_r2_valid:.6f}, Delta: {delta_base:.6f}", flush=True)
                print(f"  Model 1 (Joint) - Train R²: {joint_r2_train:.6f}, Valid R²: {joint_r2_valid:.6f}, Delta: {delta_joint:.6f}", flush=True)
                print(f"  Epistasis R²: {r2:.6f}", flush=True)
                print(f"  Alpha - Base: {alpha_base:.4f}, Joint: {alpha_joint:.4f}, L1_wt: {self.l1_wt:.4f}", flush=True)

        print("\nPost analysis on validation set completed.", flush=True)

        # sort pareto_validation_r2 by key (pipeline_id)
        pareto_validation_r2 = dict(sorted(pareto_validation_r2.items()))

        ################# FIND THE UTOPIA MODEL #################
        assert len(pareto_validation_r2) > 0, "No utopia pipeline found."

        utopia_point_pipeline_id = None
        if len(pareto_validation_r2) > 1:
            # find model based on Utopia point (maximize r2 and minimize feature count)
            utopia_pipeline_id = {}
            max_r2 = max([data["validation_r2"] for data in pareto_validation_r2.values()])
            min_r2 = min([data["validation_r2"] for data in pareto_validation_r2.values()])
            max_comp = max([data["feature_cnt"] for data in pareto_validation_r2.values()])
            min_comp = min([data["feature_cnt"] for data in pareto_validation_r2.values()])

            # Edge case: all pipelines have the same r2, pick the one with smallest feature count
            if max_r2 == min_r2:
                min_feature_cnt = float('inf')
                for pid, data in pareto_validation_r2.items():
                    if data["feature_cnt"] < min_feature_cnt:
                        min_feature_cnt = data["feature_cnt"]
                        utopia_point_pipeline_id = pid
            else:
                for pid, data in pareto_validation_r2.items():
                    # save utopia distance score
                    r2 = (1 - ((data["validation_r2"] - min_r2) / (max_r2 - min_r2)))**2
                    comp = ((data["feature_cnt"] - min_comp) / (max_comp - min_comp))**2
                    utopia_pipeline_id[pid] = np.sqrt(r2 + comp)

                # find the set of snps with the smallest utopia distance
                min_distance = float32_t(10000000.0)
                utopia_point_pipeline_id = None
                for pid, distance in utopia_pipeline_id.items():
                    if min_distance > distance:
                        min_distance = distance
                        utopia_point_pipeline_id = pid
        else:
            # only one pipeline in pareto front, it is the utopia point
            utopia_point_pipeline_id =  0
        assert utopia_point_pipeline_id is not None, "Utopia point pipeline ID should not be None."

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
            # Convert interaction tuples to string format: chr1.123:chr2.456
            feature_set_str = ";".join(sorted([f"chr{interaction[0]}:chr{interaction[1]}" for interaction in data['feature_set']]))

            pareto_data.append({
                'Pipeline ID': pid + 1,  # Start from 1 instead of 0
                'Cross-validated Train R2': data['train_r2'],
                'Validation R2': data['validation_r2'],
                'Feature Count': data['feature_cnt'],
                'Selector': data['selector'],
                'Selector Params': data['selector_params'],
                'Feature Set': feature_set_str  # Interaction pairs as semicolon-separated string
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
            print(f"Encoding SNP {snp} for final test...", flush=True)
            snp1, snp2 = snp
            ray_jobs.append(ray_utils.ray_interaction_encoder.remote(
                self.hub.get_snp_ori_ray_id(snp1),
                self.hub.get_snp_ori_ray_id(snp2),
                self.all_y_ray_id,
                combined_idx_ray_id,
                self.hub.get_encoding(snp),
                snp
            ))

        print(f"Encoding {len(snp_names)} SNPs for final test...", flush=True)

        # Process encoding results
        transformed_snp_ray_ids = {}
        while len(ray_jobs) > 0:
            finished, ray_jobs = ray.wait(ray_jobs)
            encoded_x, snp_name = ray.get(finished)[0] # 
            # Put encoded array into Ray object store
            transformed_snp_ray_ids[snp_name] = ray.put(encoded_x)
        
        # update hub with the new encodings based on full train+validation data
        for snp in snp_names:
            self.hub.update_enc_ray_id(snp, transformed_snp_ray_ids[snp])

        # Create column names for PFI (include encoding type)
        column_names_with_encoding = [f'chr{snp}_{self.hub.get_encoding(snp)}' for snp in snp_names]

        # Calculate train + validation R² using ray remote function - this will be the R² of the final model trained on combined train+validation data and evaluated on the same combined train+validation data (to check for overfitting)
        print("Calculating train + validation R²...", flush=True)
        train_valid_r2_job = ray_utils.ray_eval_pipeline_r2.remote(
            component_map=self.hub.build_component_map(pipeline_data['pipeline'].get_branch_set()),
            y=self.all_y_ray_id,
            train_idx=combined_idx_ray_id,
            valid_idx=combined_idx_ray_id,  # Use combined train+validation indices for evaluation
            pop_id=uint32_t(0),
            l1_wt=self.l1_wt
        )
        train_val_r2, _, error, alpha_base_tv, alpha_joint_tv, base_r2_tv_train, base_r2_tv_valid, joint_r2_tv_train, joint_r2_tv_valid = ray.get(train_valid_r2_job)
        if error < float32_t(0.0):
            print(f"Error during train + validation R² calculation", flush=True)
            train_val_r2 = float32_t(-1.0)
            base_r2_tv_train = float32_t(-1.0)
            base_r2_tv_valid = float32_t(-1.0)
            joint_r2_tv_train = float32_t(-1.0)
            joint_r2_tv_valid = float32_t(-1.0)

        print(f'\nTrain + Validation Set Results:', flush=True)
        print(f'  Model 0 (Base) - Train R²: {base_r2_tv_train:.6f}, Valid R²: {base_r2_tv_valid:.6f}, Delta: {base_r2_tv_valid - base_r2_tv_train:.6f}', flush=True)
        print(f'  Model 1 (Joint) - Train R²: {joint_r2_tv_train:.6f}, Valid R²: {joint_r2_tv_valid:.6f}, Delta: {joint_r2_tv_valid - joint_r2_tv_train:.6f}', flush=True)
        print(f'  Epistasis R²: {train_val_r2:.6f}', flush=True)
        print(f'  Alpha - Base: {alpha_base_tv:.4f}, Joint: {alpha_joint_tv:.4f}, L1_wt: {self.l1_wt:.4f}', flush=True)

        # Calculate test R² using ray remote function
        print("Calculating test R²...", flush=True)
        test_r2_job = ray_utils.ray_eval_pipeline_r2.remote(
            component_map=self.hub.build_component_map(pipeline_data['pipeline'].get_branch_set()),
            y=self.all_y_ray_id,
            train_idx=combined_idx_ray_id,
            valid_idx=test_idx_ray_id,
            pop_id=uint32_t(0),
            l1_wt=self.l1_wt
        )

        test_r2, _, error, alpha_base_test, alpha_joint_test, base_r2_test_train, base_r2_test_test, joint_r2_test_train, joint_r2_test_test = ray.get(test_r2_job)

        if error < float32_t(0.0):
            print(f"Error during test R² calculation", flush=True)
            test_r2 = float32_t(-1.0)
            base_r2_test_train = float32_t(-1.0)
            base_r2_test_test = float32_t(-1.0)
            joint_r2_test_train = float32_t(-1.0)
            joint_r2_test_test = float32_t(-1.0)

        pipeline_data['test_r2'] = test_r2
        print(f'\nTest Set Results:', flush=True)
        print(f'  Model 0 (Base) - Train R²: {base_r2_test_train:.6f}, Test R²: {base_r2_test_test:.6f}, Delta: {base_r2_test_test - base_r2_test_train:.6f}', flush=True)
        print(f'  Model 1 (Joint) - Train R²: {joint_r2_test_train:.6f}, Test R²: {joint_r2_test_test:.6f}, Delta: {joint_r2_test_test - joint_r2_test_train:.6f}', flush=True)
        print(f'  Epistasis R²: {test_r2:.6f}', flush=True)
        print(f'  Alpha - Base: {alpha_base_test:.4f}, Joint: {alpha_joint_test:.4f}, L1_wt: {self.l1_wt:.4f}', flush=True)

        # Calculate PFI on test set using ray_pfi
        print("Calculating permutation feature importance on test set...", flush=True)

        # Create an OLS regressor for PFI calculation
        ols_regressor = OLSRegressor()

        # Call ray_pfi
        pfi_job = ray_utils.ray_pfi.remote(
            # component_map=self.hub.build_component_map(pipeline_data['pipeline'].get_branch_set()),
            X=[transformed_snp_ray_ids[snp] for snp in snp_names],
            y=self.all_y_ray_id,
            train_idx=combined_idx_ray_id,
            valid_idx=test_idx_ray_id,
            new_column_names=column_names_with_encoding,
            root_node=ols_regressor,
            random_state=self.rng.integers(0, 100000),
            pop_id=uint32_t(0)
        )

        pfi_results, _ = ray.get(pfi_job)

        print(f"PFI calculated for {len(pfi_results)} features", flush=True)

        # Generate correlation heatmaps for validation and test sets
        print("\nGenerating correlation heatmaps for utopia model features...", flush=True)

        # Get the encoded feature matrices for validation and test sets
        # For validation set
        X_valid_features = np.column_stack([ray.get(transformed_snp_ray_ids[snp])[ray.get(self.val_idx_ray)] for snp in snp_names])

        # For test set
        X_test_features = np.column_stack([ray.get(transformed_snp_ray_ids[snp])[self.test_idx] for snp in snp_names])

        # Create correlation matrices
        corr_valid = np.corrcoef(X_valid_features, rowvar=False)
        corr_test = np.corrcoef(X_test_features, rowvar=False)

        # Create feature labels (use encoding type in label)
        feature_labels = [f"{snp[0]}_{self.hub.get_encoding(snp)}_{snp[1]}" for snp in snp_names]

        # Plot validation set correlation heatmap
        plt.figure(figsize=(max(10, len(snp_names)), max(8, len(snp_names) * 0.8)))
        sns.heatmap(corr_valid,
                    xticklabels=feature_labels,
                    yticklabels=feature_labels,
                    cmap='coolwarm',
                    center=0,
                    vmin=-1,
                    vmax=1,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={'label': 'Correlation'})
        plt.title('Feature Correlation Heatmap - Validation Set', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        valid_heatmap_path = os.path.join(self.save_directory, 'utopia_correlation_heatmap_validation.png')
        plt.savefig(valid_heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Validation correlation heatmap saved to {valid_heatmap_path}", flush=True)

        # Plot test set correlation heatmap
        plt.figure(figsize=(max(10, len(snp_names)), max(8, len(snp_names) * 0.8)))
        sns.heatmap(corr_test,
                    xticklabels=feature_labels,
                    yticklabels=feature_labels,
                    cmap='coolwarm',
                    center=0,
                    vmin=-1,
                    vmax=1,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={'label': 'Correlation'})
        plt.title('Feature Correlation Heatmap - Test Set', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        test_heatmap_path = os.path.join(self.save_directory, 'utopia_correlation_heatmap_test.png')
        plt.savefig(test_heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Test correlation heatmap saved to {test_heatmap_path}", flush=True)

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
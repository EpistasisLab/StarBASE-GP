#####################################################################################################
#
# Hub Interface for K1 that will hold all SNP information and provide fast querying of SNPs
#
#####################################################################################################

from ..Base.hub import Hub
from ..Base.ordered_hub import Ordered_Hub
from ..Base.types import (snp_t, int32_t, rng_t, uint32_t, float32_t, int16_t, uint16_t)
from ..Base.utils import snp_chrm_pos

from typeguard import typechecked
from typing import List, Dict, Set
import numpy as np
import ray
import time

# SortedList is a sorted list implementation in Python for fast insertion and deletion
# https://grantjenks.com/docs/sortedcontainers/sortedlist.html
from sortedcontainers import SortedList

@typechecked
class K1_Hub(Hub):

    class Considered:
        """
        Container to maintain snps that have not been been flagged as inactive throughout the evolutionary process.
        This is used to ensure that we do not select snps that have been pruned by and get a speedup when selecting random snps.
        Hub in this class is broken down by chromosome and and sorted positions for fast querying of snps within a given distance.
        """

        def __init__(self, snps: List[snp_t]) -> None:
            """
            Create a dictionary with all snps broken down by chromosome and position.
            Then we save them in a dictionary with the chromosome as the key and the snps in a sorted list.
            The individual lists will be updated by removing snps that have been pruned.

            Args:
                snps (List[snp_t]): List of SNPs to be added to the considered hub.
            """

            # dictionary to store the left and right neighbors for each snp
            self.nearest_neighbor_dict: Dict[snp_t] = {}

            start_time = time.time()
            # create a dictionary to hold all snps
            self.hub = {}
            for snp in snps:
                # split snp up by chromosome and position
                chrom, pos = snp_chrm_pos(snp)

                # check if key exists
                if chrom not in self.hub:
                    self.hub[chrom] = [pos]
                else:
                    self.hub[chrom].append(pos)

            dict_build_time = time.time() - start_time
            print(f"  - Considered Hub dictionary building: {dict_build_time:.4f} seconds", flush=True)

            # sort all lists within dictionary
            sort_start = time.time()
            for chrom, pos_l in self.hub.items():
                # sort the list and store in SortedList and update the dictionary
                pos_l.sort()
                self.hub[chrom] = SortedList(pos_l)
            sort_time = time.time() - sort_start
            print(f"  - Considered Hub sorting and SortedList creation: {sort_time:.4f} seconds", flush=True)
            
            # Cache non-empty chromosomes for O(1) access during mutations
            self._non_empty_chroms = set(self.hub.keys())
            return
    
        def get_nearest_neighbor(self, chrom: int32_t, snp: snp_t, rng: rng_t) -> snp_t:
            """
            Get the nearest neighbor SNP position using the dictionary from the considered hub.
            """
            # safety checks
            assert chrom in self.hub, f"Chromosome {chrom} not found in hub"
            
            if snp not in self.nearest_neighbor_dict:
                # calculate nearest neighbor and store in dictionary
                neighbor = self.calculate_nearest_neighbor(chrom, snp_chrm_pos(snp)[1])
                self.nearest_neighbor_dict[snp] = neighbor

            # if no neighbors, get random snp from different chromosome
            if len(self.nearest_neighbor_dict[snp]) == 0:
                return self.get_ran_snp_out_chrm(chrom, rng)

            return rng.choice(self.nearest_neighbor_dict[snp])

        def calculate_nearest_neighbor(self, chrom: int32_t, position: int32_t) -> snp_t:
            """
            Get a random nearest neighbor SNP position in the given chromosome.
            Assumes position always exists in the list.

            Args:
                chrom (int32_t): chromosome number to get the nearest neighbor from.
                position (int32_t): position to find the nearest neighbor for (must exist in list).
                rng (rng_t): Random number generator.

            Returns:
                snp_t: Nearest neighbor SNP position in the given chromosome.
            """

            # safety checks
            assert chrom in self.hub, f"Chromosome {chrom} not found in hub"
            
            # get the sorted list for this chromosome
            pos_list = self.hub[chrom]
            n = len(pos_list)
            
            # position always exists, so bisect_left gives us its exact index
            idx = pos_list.bisect_left(position)
            
            # verify position exists at this index
            assert idx < n and pos_list[idx] == position, f"Position {position} not found in chromosome {chrom}"
            
            # determine valid neighbors
            left_idx = idx - 1
            right_idx = idx + 1

            neighbor_list = []
            
            # choose based on which neighbors exist
            if left_idx >= 0 and right_idx < n:
                # both neighbors exist - randomly choose one
                neighbor_list = [snp_t(f'{chrom}.{pos_list[left_idx]}'), snp_t(f'{chrom}.{pos_list[right_idx]}')]
            elif left_idx >= 0:
                # only left neighbor exists
                neighbor_list = [snp_t(f'{chrom}.{pos_list[left_idx]}')]
            elif right_idx < n:
                # only right neighbor exists
                neighbor_list = [snp_t(f'{chrom}.{pos_list[right_idx]}')]
            else:
                # only right neighbor exists
                neighbor_list = []
            
            # return the chosen neighbor
            return neighbor_list
        
        def clear_nearest_neighbor(self) -> None:
            """
            Clear the nearest neighbor cache dictionary.
            """
            self.nearest_neighbor_dict.clear()
            return

        def get_random_snp_in_chromosome(self, chrom: int32_t, position: int32_t, rng: rng_t) -> snp_t:
            """
            Get a random SNP position in the given chromosome.
            Make sure that the SNP returned is not the same as the input position.

            Args:
                chrom (int32_t): Chromosome number
                position (int32_t): Position to avoid
                rng (rng_t): Random number generator

            Returns:
                snp_t: A random SNP from the chromosome (different from input position if possible)
            """

            # make sure the chromosome exists
            assert chrom in self.hub
            
            pos_list = self.hub[chrom]
            n = len(pos_list)
            
            # if only one position, return it
            if n == 1:
                return snp_t(f'{chrom}.{pos_list[0]}')
            
            # replaced the earlier for loop implementation for efficiency
            # get random index, if it matches position, try adjacent index
            idx = rng.integers(0, n)
            pos = pos_list[idx]
            
            if pos != position:
                return snp_t(f'{chrom}.{pos}')
            
            # position matched, use next index (wrap around if needed)
            idx = (idx + 1) % n
            return snp_t(f'{chrom}.{pos_list[idx]}')

        def get_ran_snp_out_chrm(self, chrom: int32_t, rng: rng_t) -> snp_t:
            """
            Get a random SNP position from a different chromosome.

            Args:
                chrom (int32_t): Chromosome number to avoid.
                rng (rng_t): A numpy random number generator from the evolver

            Returns:
                snp_t: A random SNP from a different chromosome.
            """

            # Use cached non-empty chromosomes, excluding input chromosome
            available_chroms = self._non_empty_chroms - {chrom}
            assert len(available_chroms) > 0, "No SNPs available in the Considered Hub from different chromosomes."

            # get random chromosome from set (convert to list for rng.choice)
            c = rng.choice(list(available_chroms))
            i = rng.integers(0, len(self.hub[c]))

            # return a random position from the chromosome
            return snp_t(f"{c}.{self.hub[c][i]}")

        def remove_snp(self, snp: snp_t) -> None:
            """
            Remove a SNP from the considered hub.
            Only happens when a SNP has been flagged as inactive: due to pruning or SNP marginal R2 < threshold.

            Args:
                snp (snp_t): SNP to be removed from the considered hub.
            """

            # get chromosome and position
            chrom, pos = snp_chrm_pos(snp)

            # make sure the chromosome exists
            assert chrom in self.hub

            # remove snp from the list
            # will error if the position does not exist
            self.hub[chrom].remove(pos)
            
            # Update cache if chromosome is now empty
            if len(self.hub[chrom]) == 0:
                self._non_empty_chroms.discard(chrom)

            return

        def get_ran_snp(self, rng: rng_t, anchor: snp_t, mutation_tries: uint16_t) -> snp_t:
            """
            Given an anchor SNP, return a random SNP.
            First, find all chromosome keys that have at least one SNP position in them.
            Second, randomly pick a chromosome key and pick a random SNP position.
            Try this for mutation_tries number of attempts.

            Args:
                rng (rng_t): A numpy random number generator from the evolver
                anchor (snp_t): The anchor SNP to avoid returning
                mutation_tries (uint16_t): Number of attempts to find a random SNP that is not the anchor SNP.

            Returns:
                snp_t: A random SNP that is not the anchor SNP.
            """

            # get all chromosomes with at least one snp (use list comprehension)
            chrom = [k for k, v in self.hub.items() if len(v) > 0]
            assert len(chrom) > 0, "No SNPs available in the Considered Hub."

            # try to get a random snp that is not the anchor snp
            for _ in range(mutation_tries):
                # get random chromosome
                c = rng.choice(chrom)
                # get random position from the chromosome
                pos = rng.choice(self.hub[c])
                # construct the snp
                s = snp_t(f"{c}.{pos}")
                # make sure we do not return the anchor snp
                if s != anchor:
                    return s

            # if we exhaust all tries, return the anchor
            return anchor

        def get_total(self) -> uint32_t:
            """
            Get the total number of SNPs in the considered hub.

            Returns:
                uint32_t: Total number of SNPs in the considered hub.
            """

            sum = uint32_t(0)
            for _, pos_l in self.hub.items():
                sum += uint32_t(len(pos_l))
            return sum

        def get_keys_with_snps(self) -> List[int32_t]:
            """
            Get a list of all chromosome keys that have at least one SNP position in them.

            Returns:
                List[int32_t]: List of chromosome keys with at least one SNP position.
            """

            keys = []
            for k, v in self.hub.items():
                if len(v) > 0:
                    keys.append(k)
            return keys

    class DB:
        """
        Data base to hold all snp information.
        """
        def __init__(self):
            """
            self.hub: dictionary to hold all snp and values (assuming that all snps are already in the hub)

            Order of values in the list per SNP:
                (0) res (float32_t): best r2 result placeholder, default = -1
                (1) idx (int32_t): corresponding index of the snp position in the sorted list within Ordered_Hub
                (2) ori_rid (ray.ObjectID): ray id for original feature valeus of the corresponding snp in the hub value list, default = None
                (3) end_rid (None | ray.ObjectID): ray id for encoded feature values the corresponding snp in the hub value list, default = None
                (4) enc (snp_t): best encoder type placeholder, default = ''
                (5) seen (bool): has this snp been seen before, default = False
                (6) active (bool): is this snp still active (not prunned and r2 > snp_explainability_threshold), default = True
                (7) gen_seen (int16_t): generation seen, default = -1
                (8) gen_pruned (int16_t): generation prunned, default = -1
                (9) pruned_reason (snp_t): reason for pruning, default = ''
                (10) ld_threshold (float32_t): LD threshold, default = -1
                (11) ld_genomic_distance (int32_t): genomic distance, default = -1
                (12) anchor_snp (snp_t): anchor SNP used for LD pruning, default = ''
                (13) pager_0 (float32_t): PAGER LUT value for genotype 0, default = -1
                (14) pager_1 (float32_t): PAGER LUT value for genotype 0.5, default = -1
                (15) pager_2 (float32_t): PAGER LUT value for genotype 1, default = -1
                (16) left window index (int32_t): left window index for fast querying, default = -1
                (17) right window index (int32_t): right window index for fast querying, default = -1
            """

            # {snp: [res, idx, ori_rid, end_rid, enc, seen, active, gen_seen, gen_pruned, pruned_reason, ld_threshold, ld_genomic_distance, anchor_snp, pager_0, pager_1, pager_2, left_window_idx, right_window_idx]}
            self.hub = {}

        # will add snp, sum, bin, pos， idx, res, typ to the hub
        def add_to_hub(self,
                       snp: snp_t,
                       res: float32_t,
                       idx: int32_t,
                       ori_rid: ray.ObjectID,
                       end_rid: None,
                       enc: snp_t,
                       seen=False,
                       active=True,
                       gen_seen: int16_t=int16_t(-1),
                       gen_pruned:int16_t=int16_t(-1),
                       pruned_reason: snp_t = snp_t(''),
                       ld_threshold: float32_t = float32_t(-1.0),
                       ld_genomic_distance: int32_t = int32_t(-1),
                       anchor_snp: snp_t = snp_t(''),
                       pager_0: float32_t = float32_t(-1.0),
                       pager_1: float32_t = float32_t(-1.0),
                       pager_2: float32_t = float32_t(-1.0),
                       left_window_idx: int32_t = int32_t(-1),
                       right_window_idx: int32_t = int32_t(-1),
                       ) -> None:
            """
            Process Args and add to hub.

            Args:
                (k) snp (snp_t): chrm.pos string
                (0) res (float32_t): best r2 result placeholder, default = -1
                (1) idx (int32_t): corresponding index of the snp position in the sorted list within Ordered_Hub
                (2) ori_rid (ray.ObjectID): ray id for original feature valeus of the corresponding snp in the hub value list, default = None
                (3) end_rid (None | ray.ObjectID): ray id for encoded feature values the corresponding snp in the hub value list, default = None
                (4) enc (snp_t): best encoder type placeholder, default = ''
                (5) seen (bool): has this snp been seen before, default = False
                (6) active (bool): is this snp still active (not prunned and r2 > snp_explainability_threshold), default = True
                (7) gen_seen (int16_t): generation seen, default = -1
                (8) gen_pruned (int16_t): generation prunned, default = -1
                (9) pruned_reason (snp_t): reason for pruning, default = ''
                (10) ld_threshold (float32_t): LD threshold, default = -1
                (11) ld_genomic_distance (int32_t): genomic distance, default = -1
                (12) anchor_snp (snp_t): anchor SNP used for LD pruning, default = ''
                (13) pager_0 (float32_t): PAGER LUT value for genotype 0, default = -1
                (14) pager_1 (float32_t): PAGER LUT value for genotype 0.5, default = -1
                (15) pager_2 (float32_t): PAGER LUT value for genotype 1, default = -1
                (16) left_window_idx (int32_t): left window index for fast querying, default = -1
                (17) right_window_idx (int32_t): right window index for fast querying, default = -1
            """

            # add to hub
            self.hub[snp] = [res,idx,ori_rid,end_rid,enc,seen,active,gen_seen,gen_pruned,pruned_reason,ld_threshold,ld_genomic_distance,anchor_snp,pager_0,pager_1,pager_2,left_window_idx,right_window_idx]
            return

        def get_r2(self, snp: snp_t) -> float32_t:
            # check for snp existence
            assert snp in self.hub
            # return data
            return self.hub[snp][0]

        def get_idx(self, snp: snp_t) -> int32_t:
            # assert that snp is in hub
            assert snp in self.hub
            # return data
            return self.hub[snp][1]

        def get_ori_ray_id(self, snp: snp_t) -> ray.ObjectID:
            # assert that snp is in hub
            assert snp in self.hub
            # assert snp is active in hub
            assert self.get_active_flag(snp) == True
            # return data
            return self.hub[snp][2]

        def get_enc_ray_id(self, snp: snp_t) -> ray.ObjectID:
            # assert that snp is in hub
            assert snp in self.hub
            # assert snp is active in hub
            assert self.get_active_flag(snp)
            # assert snp has been seen in hub
            assert self.get_seen_flag(snp)
            # if snp encoding is additive, return original ray id
            if self.get_encoding(snp) == snp_t('additive'):
                return self.get_ori_ray_id(snp)
            # return data
            return self.hub[snp][3]

        def get_encoding(self, snp: snp_t) -> snp_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][4]

        def get_seen_flag(self, snp: snp_t) -> bool:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][5]

        def get_active_flag(self, snp: snp_t) -> bool:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][6]

        def get_gen_seen(self, snp: snp_t) -> int16_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][7]

        def get_gen_pruned(self, snp: snp_t) -> int16_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][8]

        def get_pruned_reason(self, snp: snp_t) -> snp_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][9]

        def get_ld_threshold(self, snp: snp_t) -> float32_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][10]

        def get_ld_genomic_distance(self, snp: snp_t) -> int32_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][11]

        def get_anchor_snp(self, snp: snp_t) -> snp_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][12]

        def get_pager_0(self, snp: snp_t) -> float32_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][13]

        def get_pager_1(self, snp: snp_t) -> float32_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][14]

        def get_pager_2(self, snp: snp_t) -> float32_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][15]

        def get_left_window_idx(self, snp: snp_t) -> int32_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][16]

        def get_right_window_idx(self, snp: snp_t) -> int32_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][17]

        def flip_activate_flag_r2(self, snp: snp_t) -> None:
            """
            Flip active flag via r2 thresholding.

            Args:
                snp (snp_t): SNP to flip the active flag for.
            """

            # check snp exists in the hub
            assert snp in self.hub
            # make sure we have not seen this snp before
            assert self.get_active_flag(snp) == True
            # flip the flag
            self.hub[snp][6] = False
            return

        def flip_activate_flag_ld(self, snp: snp_t, gen_pruned: int16_t) -> None:
            """
            Flip activation flag of SNP via LD pruning

            Args:
                snp (snp_t): SNP to flip the active flag for.
                gen_pruned (int16_t): Generation pruned.
            """

            # check snp exists in the hub
            assert snp in self.hub
            # make sure we have not seen this snp before
            assert self.get_active_flag(snp)
            # record the generation prunned
            self.hub[snp][8] = gen_pruned
            # delete the ray ids to save memory
            self.delete_ori_ray_id(snp)
            # only delete the encoding ray id if it's not additive
            if self.get_encoding(snp) != snp_t('additive'):
                self.delete_enc_ray_id(snp)
            # flip the active flag
            self.hub[snp][6] = False
            return

        def flip_seen_flag(self, snp: snp_t) -> None:
            """
            Flip seen flag of SNP once it has been evaluated

            Args:
                snp (snp_t): SNP to flip the seen flag for.
            """

            # check snp exists in the hub
            assert snp in self.hub
            # make sure we have not seen this snp before
            assert self.get_seen_flag(snp) == False
            # flip the flag
            self.hub[snp][5] = True
            return

        def delete_ori_ray_id(self, snp: snp_t) -> None:
            """
            Delete the ray ids for ori_rid to save memory

            Args:
                snp (snp_t): SNP to delete the original ray id for.
            """

            # assert that snp is in hub
            assert snp in self.hub
            assert self.hub[snp][2] is not None
            # delete the ray id
            self.hub[snp][2] = None
            return

        def delete_enc_ray_id(self, snp: snp_t) -> None:
            """
            Delete encoded array of SNP.

            Args:
                snp (snp_t): SNP to delete the encoded ray id for.
            """

            # assert that snp is in hub
            assert snp in self.hub
            assert self.hub[snp][3] is not None
            assert self.get_active_flag(snp) == True
            # delete the ray id
            self.hub[snp][3] = None
            return

        def update_r2_enc(self,
                              snp: snp_t,
                              r2: float32_t,
                              encoding:snp_t,
                              enc_x: ray.ObjectID | None,
                              gen_seen: int16_t,
                              snp_explainability_threshold: float32_t,
                              pager_lut: np.ndarray | None = None) -> None:
            """
            Update SNP hub with the r2 and encoding type & array once evaluated for its R2.

            Args:
                snp (snp_t): SNP to update.
                r2 (float32_t): R2 value to update.
                encoding (snp_t): Encoding type to update.
                enc_x (ray.ObjectID | None): Encoded ray Object ID to update.
                gen_seen (int16_t): Generation seen to update.
                snp_explainability_threshold (float32_t): SNP explainability threshold to determine if SNP is active.
                pager_lut (np.ndarray | None): PAGER LUT values to update if encoding is 'pager'.
            """

            # assert that snp is in hub
            assert snp in self.hub
            # make sure we have not seen this snp before
            assert self.get_seen_flag(snp) == False
            # make sure gen_seen is valid
            assert gen_seen >= int16_t(0)
            # check if r2 < 0 then enc_x should be None
            assert not (r2 < snp_explainability_threshold and enc_x is not None)

            # flip the seen flag
            self.flip_seen_flag(snp)
            # update the results
            self.hub[snp][0] = r2
            # update the encoder type
            self.hub[snp][4] = encoding
            # update the generation seen
            self.hub[snp][7] = gen_seen

            # Update PAGER LUT values if encoding is 'pager' and LUT is provided
            if encoding == snp_t('pager') and pager_lut is not None and len(pager_lut) == 3:
                self.hub[snp][13] = float32_t(pager_lut[0])  # pager_0
                self.hub[snp][14] = float32_t(pager_lut[1])  # pager_1
                self.hub[snp][15] = float32_t(pager_lut[2])  # pager_2

            # if r2 is negative, flip to inactive
            if snp_explainability_threshold >= float32_t(0.0) and r2 < snp_explainability_threshold:
                self.flip_activate_flag_r2(snp)
                self.delete_ori_ray_id(snp)
            else:
                # update the encoded ray id
                self.hub[snp][3] = enc_x
            return

        def add_ld_details(self, snp: snp_t, reason: snp_t, threshold: float32_t, genomic_distance: int32_t, anchor_snp: snp_t) -> None:
            """
            Add LD details for a pruned SNP.

            Args:
                snp (snp_t): SNP to add LD details for.
                reason (snp_t): What was the reason for pruning.
                threshold (float32_t): LD threshold used for pruning.
                genomic_distance (int32_t): Genomic distance used for pruning.
                anchor_snp (snp_t): Anchor SNP used for pruning.
            """

            # make sure the snp is in the hub
            assert snp in self.hub
            # update the details
            self.hub[snp][9] = reason
            self.hub[snp][10] = threshold
            self.hub[snp][11] = genomic_distance
            self.hub[snp][12] = anchor_snp
            return

    def __init__(self, snp_list: List[snp_t], snps_ray_ids:Dict[snp_t, ray.ObjectRef], window_distance:int32_t) -> None:
        """
        Create all required Hubs: Ordered, Considered, and Interfact specific tools

        Args:
            snp_list (List[snp_t]): List of SNPs we need to keep track of in the hub.
            snps_ray_ids (Dict[snp_t, ray.ObjectRef]): Dictionary mapping SNPs to their corresponding Ray Object IDs.
        """

        hub_init_start = time.time()

        # how many rolls do we try for mutations
        self.mutation_tries = uint16_t(20)

        # initialize non pruned hub
        print('Initializing Considered Hub')
        consider_start = time.time()
        self.consider = self.Considered(snp_list)
        consider_time = time.time() - consider_start
        print(f'Considered Hub Initialized in {consider_time:.4f} seconds\n')

        # order hub stuff
        print('Initializing Ordered Hub')
        order_start = time.time()
        self.order = Ordered_Hub()
        # get snps and their bin id
        snp_bin = self.order.generate_order(snp_list)
        order_time = time.time() - order_start
        print(f'Ordered Hub Initialized in {order_time:.4f} seconds\n')

        # snp db hub stuff
        print('Initializing SNP Database Hub')
        db_start = time.time()
        self.db = self.DB()
        db_create_time = time.time() - db_start
        print(f"  - DB object creation: {db_create_time:.4f} seconds", flush=True)

        # update snp_hub with snp_bin and snp header positions
        populate_start = time.time()
        for s in snp_bin:
            # make sure that the snp is in the correct format
            assert isinstance(s[0], snp_t)
            # make sure header position matches the csv header
            # assert s[0] == snps[h_pos]
            # make sure the snp bin poisition is correct
            assert snp_chrm_pos(s[0])[1] == self.order.order[snp_chrm_pos(s[0])[0]][s[1]]

            # break snp up into chrom and pos
            chrom, pos = snp_chrm_pos(s[0])
            # get left and right window indices for fast querying
            left_window_idx, right_window_idx = self.order.get_window_indices(chrom, pos, s[1], window_distance)

            # add snp to hub with all its data
            self.db.add_to_hub(snp=s[0],
                                res=float32_t(-1.0),
                                idx=int32_t(s[1]),
                                ori_rid=snps_ray_ids[s[0]],
                                end_rid=None,
                                enc=snp_t(''),
                                seen=False,
                                active=True,
                                gen_seen=int16_t(-1),
                                gen_pruned=int16_t(-1),
                                pruned_reason=snp_t(''),
                                ld_threshold=float32_t(-1.0),
                                ld_genomic_distance=int32_t(-1),
                                anchor_snp=snp_t(''),
                                left_window_idx=left_window_idx,
                                right_window_idx=right_window_idx,)
        populate_time = time.time() - populate_start
        print(f"  - DB population with {len(snp_bin)} SNPs: {populate_time:.4f} seconds", flush=True)
        print(f'SNP Hub Initialized in {time.time() - db_start:.4f} seconds\n')

        total_time = time.time() - hub_init_start
        print(f'=== Total K1_Hub Initialization Time: {total_time:.4f} seconds ===\n', flush=True)
        return

    def get_encoding(self, snp: snp_t) -> snp_t:
        # get best type of encoder for a given snp
        return self.db.get_encoding(snp)

    def get_r2(self, snp: snp_t) -> float32_t:
        # get r2 for a given snp from snp hub
        return self.db.get_r2(snp)

    # save the epi_hub and snp_hub to a file
    def save_hubs(self, save_dir: str) -> None:
        """
        Save the SNP hub to a CSV file.

        Header positions:
               res_pos = 0 # position for r2 recived from evaluation
               bin_pos = 1 # id for bin assigned to
               idx_pos = 2 # position for bin number in hub value list
               pos_pos = 3 # position for header position in hub value list
               enc_pos = 4 # position for the corresponding encoder types in hub value list
              seen_pos = 5 # position for the seen flag in hub value list
            pruned_pos = 6 # position for the pruned flag in hub value list
          gen_seen_pos = 7 # position for the seen flag in hub value list
        gen_pruned_pos = 8 # position for the pruned flag in hub value list
      pruned_reason_pos = 9 # position for the pruned reason in hub value list
       ld_threshold_pos = 10 # position for the LD threshold in hub value list
ld_genomic_distance_pos = 11 # position for the LD genomic distance in hub value list
         anchor_snp_pos = 12 # position for the anchor snp in hub value list
        """

        save_start = time.time()
        print(f"[Timing] Starting save_hubs to {save_dir}...", flush=True)

        # Save snp hub with headers
        collect_start = time.time()
        snp_data = []
        for k, v in self.db.hub.items():
            # k: snp (row[0])
            # v[0]: r2 (row[1])
            # v[1]: bin (row[2])
            # v[2]: idx (row[3])
            # v[3]: pos (row[4])
            # v[4]: enc (row[5])
            # v[5]: seen (row[6])
            # v[6]: prunned (row[7])
            # v[7]: gen_seen (row[8])
            # v[8]: gen_prunned (row[9])
            # v[9]: pruned_reason (row[10])
            # v[10]: ld_threshold (row[11])
            # v[11]: ld_genomic_distance (row[12])
            # v[12]: anchor_snp (row[13])
            # v[13]: pager_0 (row[14])
            # v[14]: pager_1 (row[15])
            # v[15]: pager_2 (row[16])
            snp_data.append([k, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]])

        collect_time = time.time() - collect_start
        print(f"  - Data collection: {collect_time:.4f}s for {len(snp_data)} SNPs", flush=True)

        # Sort snp_data by the second column (AVG_R2)
        sort_start = time.time()
        snp_data.sort(key=lambda x: x[1], reverse=True)  # reverse=True for descending order
        sort_time = time.time() - sort_start
        print(f"  - Data sorting: {sort_time:.4f}s", flush=True)

        # Write snp hub to file
        write_start = time.time()
        with open(save_dir+"snp_hub.csv", 'w') as f:
            # Write the headers for the snp_file (removed bin_idx column)
            f.write("snp,chr,bp,r2,bin_num,encoding,seen,active,gen_seen,gen_pruned,pruned_reason,ld_threshold,ld_genomic_distance,anchor_snp,pager_0,pager_1,pager_2\n")
            for row in snp_data:
                # split snp into chromosome and position
                chrom, pos = row[0].split('.')
                # Add 'chr' prefix to SNP name
                snp_with_chr = f"chr{row[0]}"
                # Get pager values from hub
                # row[13] = anchor_snp, row[14] = pager_0, row[15] = pager_1, row[16] = pager_2
                # Check if values are numeric (float) and not default -1
                try:
                    pager_0_val = float(row[14])
                    pager_0 = '' if pager_0_val < 0 else str(pager_0_val)
                except (ValueError, TypeError):
                    pager_0 = ''

                try:
                    pager_1_val = float(row[15])
                    pager_1 = '' if pager_1_val < 0 else str(pager_1_val)
                except (ValueError, TypeError):
                    pager_1 = ''

                try:
                    pager_2_val = float(row[16])
                    pager_2 = '' if pager_2_val < 0 else str(pager_2_val)
                except (ValueError, TypeError):
                    pager_2 = ''

                # Write all columns (removed bin_idx which was row[4]): row[13] is anchor_snp, then pager_0, pager_1, pager_2
                f.write(f"{snp_with_chr},{chrom},{pos},{row[1]},{row[2]},{row[5]},{row[6]},{row[7]},{row[8]},{row[9]},{row[10]},{row[11]},{row[12]},{row[13]},{pager_0},{pager_1},{pager_2}\n")

        write_snp_time = time.time() - write_start
        print(f"  - Writing snp_hub.csv: {write_snp_time:.4f}s", flush=True)

        # save csv with both seen and not prunned snps
        # Write consideration hub to file
        consider_write_start = time.time()
        with open(save_dir+"consideration_hub.csv", 'w') as f:
            # Write the headers for the snp_file
            f.write("snp,r2,encoding\n")
            for row in snp_data:
                if row[6] == True and row[7] == False:
                    # Add 'chr' prefix to SNP name
                    snp_with_chr = f"chr{row[0]}"
                    f.write(f"{snp_with_chr},{row[1]},{row[5]}\n")

        consider_write_time = time.time() - consider_write_start
        print(f"  - Writing consideration_hub.csv: {consider_write_time:.4f}s", flush=True)

        total_save_time = time.time() - save_start
        print(f"[Timing] save_hubs completed in {total_save_time:.4f}s\n", flush=True)
        return

    def update_snp_hub_r2_enc(self, snp:snp_t, r2:float32_t, enc: snp_t, enc_x: ray.ObjectID | None, gen_seen: int16_t, snp_explainability_threshold: float32_t, pager_lut: np.ndarray | None = None) -> None:
        """
        Update SNP hub with the r2 and encoding type & vector (if applicable).

        Args:
            snp (snp_t): SNP to be updated.
            r2 (float32_t): R2 value for the SNP.
            enc (snp_t): Encoding type for the SNP.
            enc_x (ray.ObjectID | None): Ray object ID for the encoded SNP values.
            gen_seen (int16_t): Generation when the SNP was seen.
            snp_explainability_threshold (float32_t): Threshold for SNP explainability.
            pager_lut (np.ndarray | None): PAGER LUT values if encoding is 'pager'.
        """

        # update Hub object: if r2 is negative, flip prunned flag
        self.db.update_r2_enc(snp, r2, enc, enc_x, gen_seen, snp_explainability_threshold, pager_lut)
        # update Consideration_Hub object: if r2 is less than threshold, remove snp from non prunned
        if r2 < snp_explainability_threshold and snp_explainability_threshold >= float32_t(0.0):
            self.consider.remove_snp(snp)
        return


    def get_ran_snp_in_window(self, anchor: snp_t, rng: rng_t) -> snp_t:
        """
        Get a random SNP from the same chromosome and within the specified window distance.

        Args:
            anchor (snp_t): Anchor SNP in "chromosome.position" format.
            rng (rng_t): Numpy random generator.
            in_window (List[int32_t]): List of positions within the window distance.

        Returns:
            snp_t: A randomly selected SNP from the same chromosome and within the specified window distance
        """

        # make sure there is a '-' inside the snp string
        assert '.' in anchor

        # break snp into chromosome and position
        chrom, pos = snp_chrm_pos(anchor)

        # roll a random snp from the list of snps
        return self.consider.get_nearest_neighbor(chrom, anchor, rng)

    def get_ran_snp_in_chrm(self, anchor: snp_t, rng: rng_t) -> snp_t:
        """
        Get a random SNP from the same chromosome but different bin.

        Args:
            anchor (snp_t): The anchor SNP in "chromosome.position" format.
            rng (rng_t): Numpy random generator.
            out_window (List[int32_t]): List of positions outside the window distance.

        Returns:
            snp_t: A randomly selected SNP from the same chromosome but outside the specified window distance.
        """

        # make sure there is a '.' inside the snp string
        assert '.' in anchor

        # break snp into chromosome and position
        chrom, pos = snp_chrm_pos(anchor)

        # return same snp
        return self.consider.get_random_snp_in_chromosome(chrom, pos, rng)


    def get_ran_snp_out_chrm(self, anchor: snp_t, rng: rng_t) -> snp_t:
        """
        Get a random SNP from outside the chromosome.

        Args:
            anchor (snp_t): The anchor SNP in "chromosome.position" format.
            rng (rng_t): Numpy random generator.

        Returns:
            snp_t: A randomly selected SNP from a different chromosome.
        """

        # make sure there is a '.' inside the snp string
        assert '.' in anchor

        # split up the snp into chromosome and position
        chrom, _ = snp_chrm_pos(anchor)

        # roll a random snp from the list of snps
        return self.consider.get_ran_snp_out_chrm(chrom, rng)

    def get_k_snps_from_chrom(self, rng:rng_t, chrom:int32_t, k:uint16_t) -> Set[snp_t]:
        """"
        Get k random SNPs from a specified chromosome.

        Args:
            rng (rng_t): Numpy random generator.
            chrom (int32_t): Chromosome number to sample SNPs from.
            k (uint16_t): Number of SNPs to sample.
        Returns:
            Set[snp_t]: A set of k randomly sampled SNPs from the specified chromosome.
        """

        k_snps = set()

        # make sure the chrom is not out of bound
        assert chrom in self.order.order

        # get random positions from chromosome's position lists
        while len(k_snps) < k: # to make sure there are no replicates
            # sample a random position from the chromosome's position list
            k_snps.add(snp_t(f"{chrom}.{rng.choice(self.order.order[chrom])}"))
        # return the snp set
        return k_snps

    def get_active_flag(self, snp: snp_t) -> bool:
        # is this snp active?
        return self.db.get_active_flag(snp)

    def process_pruned_snps(self, snps: Set[snp_t], snp_details_after_ld: Dict[snp_t, Dict], gen_pruned: int16_t) -> None:
        """
        Process pruned SNPs by updating their status in the SNP hub and removing them from the consideration hub.

        Args:
            snps (Set[snp_t]): Set of SNPs that have been pruned.
            snp_details_after_ld (Dict[snp_t, Dict]): Dictionary containing details for each pruned SNP, including reason, threshold, genomic distance, and anchor SNP.
            gen_pruned (int16_t): Generation number when the SNPs were pruned.
        """

        process_start = time.time()
        print(f"[Timing] Processing {len(snps)} pruned SNPs...", flush=True)

        # go through each snp and update the hub
        flip_total = 0.0
        add_details_total = 0.0
        remove_total = 0.0

        for snp in snps:
            # check to make sure we have not prunned this snp before
            assert self.db.get_gen_pruned(snp) == int16_t(-1)

            # flip snp to pruned
            flip_start = time.time()
            self.db.flip_activate_flag_ld(snp, gen_pruned)
            flip_total += time.time() - flip_start

            # add ld details to the snp hub
            add_start = time.time()
            self.db.add_ld_details(
                snp,
                snp_details_after_ld[snp]["reason"],
                snp_details_after_ld[snp]["threshold"],
                snp_details_after_ld[snp]["genomic_distance"],
                snp_details_after_ld[snp]["anchor_snp"]
            )
            add_details_total += time.time() - add_start

            # delete snp from non pruned
            remove_start = time.time()
            self.consider.remove_snp(snp)
            remove_total += time.time() - remove_start

        total_time = time.time() - process_start
        print(f"[Timing] process_pruned_snps completed: Flip flags={flip_total:.4f}s, Add LD details={add_details_total:.4f}s, Remove from consider={remove_total:.4f}s, Total={total_time:.4f}s", flush=True)

    def generate_r2_dict(self, snps: Set[snp_t]) -> List:
        """
        Generate a dictionary of SNPs and their corresponding r2 values.

        Args:
            snps (Set[snp_t]): Set of SNPs to generate r2 values for.

        Returns:
            List: A list of tuples containing SNPs and their r2 values.
        """

        # make sure
        assert len(snps) > 0

        return [(snp, self.get_r2(snp)) for snp in snps]

    def at_least_one_active_snp(self, snps: List[snp_t]) -> bool:
        """
        Function to check if at least one snp in the list is active.

        Parameters:
            snps (List[snp_t]): List of snps to check

        Returns:
            bool: True if at least one snp is active, False otherwise
        """

        # if one snp is active return true
        for snp in snps:
            if self.db.get_active_flag(snp):
                return True
        # return false if all snps are inactive
        return False

    def consideration_hub_size(self) -> uint32_t:
        # print size of consideration hub
        return self.consider.get_total()

    def get_random_considered_snp(self, rng: rng_t, anchor: snp_t) -> snp_t:
        # return a random snp from the consideration hub
        return self.consider.get_ran_snp(rng, anchor, self.mutation_tries)

    def get_random_snp_from_list(self, rng: rng_t, anchor: snp_t, snps: List[snp_t]) -> snp_t:
        """
        Given a set of SNPs to pick from, this function returns a random SNP.

        Args:
            rng (rng_t): Numpy random generator
            anchor (snp_t): The anchor SNP to avoid selecting
            snps (List[snp_t]): List of SNPs to choose from

        Returns:
            snp_t: A randomly selected SNP from the list, weighted by r2 values
        """

        # if no snps were collected, return a random snp from consideration hub
        if len(snps) == 0:
            return self.get_random_considered_snp(rng, anchor)

        # try to get a random snp that is not the same as the input snp
        choice = rng.choice(snps)
        assert choice != anchor, "SNP should not be the same as the anchor SNP"
        return choice

    def get_random_snp_weighted_by_r2(self, rng: rng_t, anchor: snp_t, snps: List[snp_t], r2_list: List[float32_t]) -> snp_t:
        """
        Given a set of SNPs to pick from, this function returns a random SNP weighted by its r2 value.

        Args:
            rng (rng_t): Numpy random generator
            anchor (snp_t): The anchor SNP to avoid selecting
            snps (List[snp_t]): List of SNPs to choose from
            r2_list (List[float32_t]): List of r2 values corresponding to the SNPs

        Returns:
            snp_t: A randomly selected SNP from the list, weighted by r2 values
        """

        # make sure the snps and r2_list are the same length
        assert len(snps) == len(r2_list)
        assert '.' in anchor

        # normalize r2_list to sum to 1
        r2_values = np.array(r2_list, dtype=float32_t)
        r2_values /= np.sum(r2_values)
        # get a random snp based on r2 scores as weights
        choice = rng.choice(snps, p=r2_values)

        assert choice != anchor, "SNP should not be the same as the anchor SNP"
        return choice

    def get_keys_with_snps(self) -> List[int32_t]:
        # get chromosomes with snps in consideration hub
        return self.consider.get_keys_with_snps()

    def get_ori_ray_id(self, snp: snp_t) -> ray.ObjectID:
        # get original feature from the snp hub (ray id)
        return self.db.get_ori_ray_id(snp)

    def get_enc_ray_id(self, snp: snp_t) -> ray.ObjectID:
        # get encoded feature from the snp hub (ray id)
        return self.db.get_enc_ray_id(snp)

    def seen_snps_proportion(self) -> None:
        """
        Function to print the proportion of unseen snps in the hub and the proportion of snps
        """

        unseen_count = 0
        for chrm in self.consider.hub:
            for pos in self.consider.hub[chrm]:
                # count if the snp have not been seen yet
                if self.db.get_seen_flag(snp_t(f"{chrm}.{pos}")) == False:
                    unseen_count += 1

        # print proportion of unseen snps
        print(f"{unseen_count/len(self.db.hub):.2%} of unseen SNPs: {unseen_count} of {len(self.db.hub)}", flush=True)
        print(f"{self.consider.get_total()/len(self.db.hub):.2%}% of SNPs still considered: {self.consider.get_total()} of {len(self.db.hub)}", flush=True)
        return

    def get_unseen_snps(self, snps: Set[snp_t]) -> Set[snp_t]:
        """
        Function to get all unseen snps from the pipelines.

        Parameters:
            snps (Set[snp_t]): Set of univariate snps

        Returns:
            Set[snp_t]: A set of unseen univariate snps
        """
        return {snp_name for snp_name in snps if not self.db.get_seen_flag(snp_name)}

    def remove_inactive_branches(self, branches: Set) -> Set:
        """
        Function to remove inactive branches from a given set of branches.

        Parameters:
            branches: Set of branches (univariate snps or interactions (K2, K3, ...))

        Returns:
            Set: An updated set of active branches
        """

        good_branches = set()
        for branch in branches:
            if self.db.get_active_flag(branch):
                good_branches.add(branch)
        return good_branches

    def get_in_window_positions(self, snp: snp_t, window_distance: int32_t) -> List[int32_t]:
        """
        Get the positions of SNPs within the window of a given SNP.

        Args:
            snp (snp_t): The SNP for which to get window positions.

        Returns:
            List[int32_t]: A list of positions within the window of the given SNP.
        """

        # make sure the snp is in the hub
        assert snp in self.db.hub
        # break snp into chromosome and position
        chrom, pos = snp_chrm_pos(snp)
        # get all positions in the window
        return self.consider.get_positions_in_window(chrom, window_distance, pos)

    def get_out_of_window_positions(self, snp: snp_t, window_distance: int32_t) -> List[int32_t]:
        """
        Get the positions of SNPs outside the window of a given SNP.

        Args:
            snp (snp_t): The SNP for which to get out-of-window positions.

        Returns:
            List[int32_t]: A list of positions outside the window of the given SNP.
        """

        # make sure the snp is in the hub
        assert snp in self.db.hub
        # break snp into chromosome and position
        chrom, pos = snp_chrm_pos(snp)
        # get all positions out of the window
        return self.consider.get_positions_out_of_window(chrom, window_distance, pos)
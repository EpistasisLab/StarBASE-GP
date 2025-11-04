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

# SortedList is a sorted list implementation in Python for fast insertion and deletion
# https://grantjenks.com/docs/sortedcontainers/sortedlist.html
from sortedcontainers import SortedList

@typechecked
class K1_Hub(Hub):

    class Considered:
        """
        Container to maintain snps that have not been been flagged as inactive throughout the evolutionary process.
        This is used to ensure that we do not select snps that have been pruned by and get a speedup when selecting random snps.
        Hub in this classe is broken down by chromosome and and sorted positions for fast querying of snps within a given distance.
        """

        def __init__(self, snps: List[snp_t]) -> None:
            """
            Create a dictionary with all snps broken down by chromosome and position.
            Then we save them in a dictionary with the chromosome as the key and the snps in a sorted list.
            The individual lists will be updated by removing snps that have been pruned.
            """

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

            # sort all lists within dictionary
            for chrom, pos_l in self.hub.items():
                # sort the list and store in SortedList and update the dictionary
                pos_l.sort()
                self.hub[chrom] = SortedList(pos_l)
            return

        def get_positions_in_chromosome(self, chrom: int32_t) -> List[int32_t]:
            """
            Get all snps in a given chromosome.
            """

            # make sure the chromosome exists
            assert chrom in self.hub
            # return all positions in the chromosome
            return list(self.hub[chrom])

        # collect all positions for a given chromosome that are not within a certain distance and given anchor position
        def get_positions_out_of_window(self, chrom: int32_t, distance: int32_t, anchor: int32_t) -> List[int32_t]:
            """
            Get all snps in a given chromosome that are outside the specified distance.
            """

            # make sure the chromosome exists
            assert chrom in self.hub
            assert anchor in self.hub[chrom]

            # get all positions in the chromosome
            pos_l = self.hub[chrom]

            # return all positions that are not within the distance
            return [pos for pos in pos_l if abs(pos - anchor) > distance]

        # remove snp from the hub when it has been flagged as inactive
        def remove_snp(self, snp: snp_t) -> None:
            # get chromosome and position
            chrom, pos = snp_chrm_pos(snp)

            # make sure the chromosome exists
            assert chrom in self.hub

            # remove snp from the list
            # will error if the position does not exist
            self.hub[chrom].remove(pos)

            return

        # get a random snp from the hub
        def get_ran_snp(self, rng: rng_t, anchor: snp_t, mutation_tries: uint16_t) -> snp_t:

            # get all chromosomes with at least one snp
            chrom = []
            for chrm in list(self.hub.keys()):
                if len(self.hub[chrm]) > 0:
                    chrom.append(chrm)
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

        # get total number of items in the hub dictionary
        def get_total(self) -> uint32_t:
            sum = uint32_t(0)
            for _, pos_l in self.hub.items():
                sum += uint32_t(len(pos_l))
            return sum

        # get list of keys with at least one snp
        def get_keys_with_snps(self) -> List[int32_t]:
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
            """

            # {snp: [res, idx, ori_rid, end_rid, enc, seen, active, gen_seen, gen_pruned, pruned_reason, ld_threshold, ld_genomic_distance, anchor_snp, pager_0, pager_1, pager_2]}
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
                       pager_2: float32_t = float32_t(-1.0)) -> None:
            """
            will take in a snp, sum, cnt, bin, and pos and add it to the hub

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
            """

            # add to hub
            self.hub[snp] = [res,idx,ori_rid,end_rid,enc,seen,active,gen_seen,gen_pruned,pruned_reason,ld_threshold,ld_genomic_distance,anchor_snp,pager_0,pager_1,pager_2]
            return

        # get snp result r^2
        def get_r2(self, snp: snp_t) -> float32_t:
            # check for snp existence
            assert snp in self.hub
            # return data
            return self.hub[snp][0]

        # get snp position idx in the ordered hub lists
        def get_idx(self, snp: snp_t) -> int32_t:
            # assert that snp is in hub
            assert snp in self.hub
            # return data
            return self.hub[snp][1]

        # get ray id for the corresponding snp in the hub value list
        def get_ori_ray_id(self, snp: snp_t) -> ray.ObjectID:
            # assert that snp is in hub
            assert snp in self.hub
            # assert snp is active in hub
            assert self.get_active_flag(snp) == True
            # return data
            return self.hub[snp][2]

        # get ray id for the corresponding snp in the hub value list
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

        # get snp encoder type
        def get_encoding(self, snp: snp_t) -> snp_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][4]

        # has this snp been seen before
        def get_seen_flag(self, snp: snp_t) -> bool:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][5]

        # is this snp still active (not prunned and r2 > snp_explainability_threshold)
        def get_active_flag(self, snp: snp_t) -> bool:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][6]

        # get generation seen
        def get_gen_seen(self, snp: snp_t) -> int16_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][7]

        # get generation pruned
        def get_gen_pruned(self, snp: snp_t) -> int16_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][8]

        # get the reason for pruning
        def get_pruned_reason(self, snp: snp_t) -> snp_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][9]

        # get the LD threshold
        def get_ld_threshold(self, snp: snp_t) -> float32_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][10]

        # get the genomic distance for LD
        def get_ld_genomic_distance(self, snp: snp_t) -> int32_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][11]

        # get the anchor snp that pruned this snp
        def get_anchor_snp(self, snp: snp_t) -> snp_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][12]

        # get pager_0 value (LUT for genotype 0)
        def get_pager_0(self, snp: snp_t) -> float32_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][13]

        # get pager_1 value (LUT for genotype 0.5)
        def get_pager_1(self, snp: snp_t) -> float32_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][14]

        # get pager_2 value (LUT for genotype 1)
        def get_pager_2(self, snp: snp_t) -> float32_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][15]

        # flip the active flag via r2
        def flip_activate_flag_r2(self, snp: snp_t) -> None:
            # check snp exists in the hub
            assert snp in self.hub
            # make sure we have not seen this snp before
            assert self.get_active_flag(snp) == True
            # flip the flag
            self.hub[snp][6] = False
            return

        # flip active flag via pruned snps
        def flip_activate_flag_ld(self, snp: snp_t, gen_pruned: int16_t) -> None:
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

        # flip the seen flag
        def flip_seen_flag(self, snp: snp_t) -> None:
            # check snp exists in the hub
            assert snp in self.hub
            # make sure we have not seen this snp before
            assert self.get_seen_flag(snp) == False
            # flip the flag
            self.hub[snp][5] = True
            return

        # delete the ray ids for ori_rid to save memory
        def delete_ori_ray_id(self, snp: snp_t) -> None:
            # assert that snp is in hub
            assert snp in self.hub
            assert self.hub[snp][2] is not None
            # delete the ray id
            self.hub[snp][2] = None
            return

        # delete the ray ids for enc_rid to save memory
        def delete_enc_ray_id(self, snp: snp_t) -> None:
            # assert that snp is in hub
            assert snp in self.hub
            assert self.hub[snp][3] is not None
            assert self.get_active_flag(snp) == True
            # delete the ray id
            self.hub[snp][3] = None
            return

        # update snp hub with the r2 and encoding type
        # assuming that this only gets called once per snp
        def update_r2_enc(self,
                              snp: snp_t,
                              r2: float32_t,
                              encoding:snp_t,
                              enc_x: ray.ObjectID | None,
                              gen_seen: int16_t,
                              snp_explainability_threshold: float32_t,
                              pager_lut: np.ndarray | None = None) -> None:
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
            if r2 < snp_explainability_threshold:
                self.flip_activate_flag_r2(snp)
                self.delete_ori_ray_id(snp)
            else:
                # update the encoded ray id
                self.hub[snp][3] = enc_x
            return

        # to update the details of the snp after LD pruning - add the anchor snp
        def add_ld_details(self, snp: snp_t, reason: snp_t, threshold: float32_t, genomic_distance: int32_t, anchor_snp: snp_t) -> None:
            # make sure the snp is in the hub
            assert snp in self.hub
            # update the details
            self.hub[snp][9] = reason
            self.hub[snp][10] = threshold
            self.hub[snp][11] = genomic_distance
            self.hub[snp][12] = anchor_snp
            return

    def __init__(self, snp_list: List[snp_t], snps_ray_ids:Dict[snp_t, ray.ObjectRef]) -> None:
        # how many rolls do we try for mutations
        self.mutation_tries = uint16_t(20)

        # initialize non pruned hub
        print('Initializing Considered Hub')
        self.consider = self.Considered(snp_list)
        print('Considered Hub Initialized')

        # order hub stuff
        print('Initializing Ordered Hub')
        self.order = Ordered_Hub()
        # get snps and their bin id
        snp_bin = self.order.generate_order(snp_list)
        print('Ordered Hub Initialized')

        # snp db hub stuff
        self.db = self.DB()
        # update snp_hub with snp_bin and snp header positions
        for s in snp_bin:
            # make sure that the snp is in the correct format
            assert isinstance(s[0], snp_t)
            # make sure header position matches the csv header
            # assert s[0] == snps[h_pos]
            # make sure the snp bin poisition is correct
            assert snp_chrm_pos(s[0])[1] == self.order.order[snp_chrm_pos(s[0])[0]][s[1]]

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
                                anchor_snp=snp_t(''))
        print('SNP Hub Initialized')
        return

    # get best type of encoder for a given snp
    def get_encoding(self, snp: snp_t) -> snp_t:
        return self.db.get_encoding(snp)

    # get r2 for a given snp from snp hub
    def get_r2(self, snp: snp_t) -> float32_t:
        return self.db.get_r2(snp)

    # save the epi_hub and snp_hub to a file
    def save_hubs(self, save_dir: str) -> None:
        """
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

        # Save snp hub with headers
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

        # Sort snp_data by the second column (AVG_R2)
        snp_data.sort(key=lambda x: x[1], reverse=True)  # reverse=True for descending order

        # Write snp hub to file
        with open(save_dir+"snp_hub.csv", 'w') as f:
            # Write the headers for the snp_file (removed bin_idx column)
            f.write("snp,chr,bp,r2,bin_num,encoding,seen,pruned,gen_seen,gen_pruned,pruned_reason,ld_threshold,ld_genomic_distance,anchor_snp,pager_0,pager_1,pager_2\n")
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

        # save csv with both seen and not prunned snps
        # Write consideration hub to file
        with open(save_dir+"consideration_hub.csv", 'w') as f:
            # Write the headers for the snp_file
            f.write("snp,r2,encoding\n")
            for row in snp_data:
                if row[6] == True and row[7] == False:
                    # Add 'chr' prefix to SNP name
                    snp_with_chr = f"chr{row[0]}"
                    f.write(f"{snp_with_chr},{row[1]},{row[5]}\n")
        return

    # update snp hub with best univariate r2 result and corresponding encoder type
    def update_snp_hub_r2_enc(self, snp:snp_t, r2:float32_t, enc: snp_t, enc_x: ray.ObjectID | None, gen_seen: int16_t, snp_explainability_threshold: float32_t, pager_lut: np.ndarray | None = None) -> None:
        # update Hub object: if r2 is negative, flip prunned flag
        self.db.update_r2_enc(snp, r2, enc, enc_x, gen_seen, snp_explainability_threshold, pager_lut)
        # update Consideration_Hub object: if r2 is less than threshold, remove snp from non prunned
        if r2 < snp_explainability_threshold:
            self.consider.remove_snp(snp)
        return

    # get a snp based on r2 performance from the same chromosome and within the same window distance
    def get_smt_snp_in_window(self, anchor: snp_t, rng: rng_t, window_distance: int32_t) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in anchor

        # break snp into chromosome and position
        chrom, position = snp_chrm_pos(anchor)
        # get window for the snp
        window = self.order.get_window(chrom, position, self.db.get_idx(anchor), window_distance)
        # reduce the window to only active snps
        valid_snps, r2_list = [], []

        for pos in window:
            s = snp_t(f"{chrom}.{pos}")

            # must be seen and active to use
            assert s != anchor, "SNP should not be the same as the anchor SNP"
            if self.db.get_seen_flag(s) and self.db.get_active_flag(s):
                valid_snps.append(s)
                r2_list.append(self.db.get_r2(s))

        # if no valid snps were found, attempt to get a random snp in window
        if len(valid_snps) == 0:
            return self.get_ran_snp_in_window(anchor, rng, window_distance)

        # get a random snp based on r2 scores as weights
        return self.get_random_snp_weighted_by_r2(rng, anchor, valid_snps, r2_list)

    # randomly sample a snp from the same chromosome and within the same window distance
    def get_ran_snp_in_window(self, anchor: snp_t, rng: rng_t, window_distance: int32_t) -> snp_t:
        # make sure there is a '-' inside the snp string
        assert '.' in anchor

        # break snp into chromosome and position
        chrom, position = snp_chrm_pos(anchor)
        # get window for the snp
        window = self.order.get_window(chrom, position, self.db.get_idx(anchor), window_distance)
        # collect all snps that have (not pruned and seen) or (r2 > 0.0 and seen)
        snps = []

        for p in window:
            # make snp
            s = snp_t(f"{chrom}.{p}")
            # if not seen, we can use it
            not_seen = self.db.get_seen_flag(s) == False
            # if seen, must be active to use it
            seen_and_active = self.db.get_seen_flag(s) and self.db.get_active_flag(s)

            assert s != anchor, "SNP should not be the same as the input SNP"
            if not_seen or seen_and_active:
                snps.append(s)

        # roll a random snp from the list of snps
        return self.get_random_snp_from_list(rng, anchor, snps)

    # geta snp from the same chromosome but different bin
    def get_smt_snp_in_chrm(self, anchor: snp_t, rng: rng_t, window_distance: int32_t) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in anchor

        # break snp into chromosome and position
        chrom, position = snp_chrm_pos(anchor)
        # get bin id for the snp
        snps_in_chrom = self.consider.get_positions_out_of_window(chrom, window_distance, position)
        # collect all snps that have not been pruned and have r2 > 0.0
        snps, r2 = [], []

        # loop through all non prunned snps and collect the ones with r2 > 0.0 and not pruned
        for pos in snps_in_chrom:
            s = snp_t(f"{chrom}.{pos}")

            assert s != anchor, "SNP should not be the same as the input SNP"
            if self.db.get_seen_flag(s) and self.db.get_active_flag(s):
                snps.append(s)
                r2.append(self.db.get_r2(s))

        # if no snps were returned, attempt to get a random snp out of chromosome
        if len(snps) == 0:
            return self.get_ran_snp_in_chrm(anchor, rng, window_distance)

        # get a random snp based on r2 scores as weights
        return self.get_random_snp_weighted_by_r2(rng, anchor, snps, r2)

    # get a random snp from the same chromosome but different bin
    def get_ran_snp_in_chrm(self, anchor: snp_t, rng: rng_t, window_distance: int32_t) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in anchor

        # break snp into chromosome and position
        chrom, position = snp_chrm_pos(anchor)
        # get bin id for the snp
        snps_in_chrom = self.consider.get_positions_out_of_window(chrom, window_distance, position)
        # collect all snps that have not been pruned and have r2 > 0.0
        snps = []

        # loop through all non prunned snps and collect the ones with r2 > 0.0 and not pruned
        for pos in snps_in_chrom:
            # make snps
            s = snp_t(f"{chrom}.{pos}")
            # if not seen, we can use it
            not_seen = self.db.get_seen_flag(s) == False
            # if seen, must be active to use it
            seen_and_active = self.db.get_seen_flag(s) == True and self.db.get_active_flag(s) == True

            assert s != anchor, "SNP should not be the same as the input SNP"
            if not_seen or seen_and_active:
                snps.append(s)

        # return same snp
        return self.get_random_snp_from_list(rng, anchor, snps)

    # get a snp from outside the chromosome with r2 > 0.0 based on r2 weight
    def get_smt_snp_out_chrm(self, anchor: snp_t, rng: rng_t) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in anchor

        # split up the snp into chromosome and position
        chrom, pos = snp_chrm_pos(anchor)
        # get keys for non pruned snps
        chrom_keys = self.consider.get_keys_with_snps()

        # remove the current chromosome from the list
        if chrom in chrom_keys:
            chrom_keys.remove(chrom)
        assert len(chrom_keys) > 0, "No other chromosomes available in Considered Hub."

        # randomly select a chromosome
        c_pic = rng.choice(chrom_keys)
        # collect all snps that have not been pruned and have r2 > 0.0
        snps, r2 = [], []

        # loop through all non pruned snps and collect the ones with r2 > 0.0 and not pruned
        for pos in self.consider.get_positions_in_chromosome(c_pic):
            s = snp_t(f"{c_pic}.{pos}")

            # must be seen and active to use it
            if self.db.get_active_flag(s) and self.db.get_seen_flag(s):
                snps.append(s)
                r2.append(self.db.get_r2(s))

        # if no snps were returned, attempt to get a random snp out of chromosome
        if len(snps) == 0:
            return self.get_ran_snp_out_chrm(anchor, rng)

        # get a random snp based on r2 scores as weights
        return self.get_random_snp_weighted_by_r2(rng, anchor, snps, r2)

    # get random snp from outside the chromosome
    def get_ran_snp_out_chrm(self, anchor: snp_t, rng: rng_t) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in anchor

        # split up the snp into chromosome and position
        chrom, pos = snp_chrm_pos(anchor)
        # get keys for non pruned snps
        chrom_keys = self.consider.get_keys_with_snps()

        # remove the current chromosome from the list
        if chrom in chrom_keys:
            chrom_keys.remove(chrom)
        assert len(chrom_keys) > 0, "No other chromosomes available in Considered Hub."

        # randomly select a chromosome
        c_pic = rng.choice(chrom_keys)
        # collect all snps that have not been pruned and have r2 > 0.0
        snps = []

        # loop through all non pruned snps and collect them
        for pos in self.consider.get_positions_in_chromosome(c_pic):
            # make snps
            s = snp_t(f"{c_pic}.{pos}")
            # not seen
            not_seen = self.db.get_seen_flag(s) == False
            # seen and active
            seen_r2_np = self.db.get_seen_flag(s) and self.db.get_active_flag(s)

            if not_seen or seen_r2_np:
                snps.append(s)

        # roll a random snp from the list of snps
        return self.get_random_snp_from_list(rng, anchor, snps)

    def get_k_snps_from_chrom(self, rng:rng_t, chrom:int32_t, k:uint16_t) -> Set[snp_t]:
        k_snps = set()

        # make sure the chrom is not out of bound
        assert chrom in self.order.order

        # get random positions from chromosome's position lists
        while len(k_snps) < k: # to make sure there are no replicates
            # sample a random position from the chromosome's position list
            k_snps.add(snp_t(f"{chrom}.{rng.choice(self.order.order[chrom])}"))
        # return the snp set
        return k_snps

    # is this snp active?
    def get_active_flag(self, snp: snp_t) -> bool:
        return self.db.get_active_flag(snp)

    # process the prunned snps
    def process_pruned_snps(self, snps: Set[snp_t], snp_details_after_ld: Dict[snp_t, Dict], gen_pruned: int16_t) -> None:
        # go through each snp and update the hub
        for snp in snps:
            # check to make sure we have not prunned this snp before
            assert self.db.get_gen_pruned(snp) == int16_t(-1)

            # flip snp to pruned
            self.db.flip_activate_flag_ld(snp, gen_pruned)
            # add ld details to the snp hub
            self.db.add_ld_details(
                snp,
                snp_details_after_ld[snp]["reason"],
                snp_details_after_ld[snp]["threshold"],
                snp_details_after_ld[snp]["genomic_distance"],
                snp_details_after_ld[snp]["anchor_snp"]
            )

            # delete snp from non pruned
            self.consider.remove_snp(snp)

    # function to take in a list of snps and generate a dictionary of snps and their corresponding r2 values
    def generate_r2_dict(self, snps: Set[snp_t]) -> List:
        # make sure
        assert len(snps) > 0

        return [(snp, self.get_r2(snp)) for snp in snps]

    # check if there is at least one active snp in the set
    def at_least_one_active_snp(self, snps: List[snp_t]) -> bool:
        # if one snp is active return true
        for snp in snps:
            if self.db.get_active_flag(snp):
                return True
        # return false if all snps are inactive
        return False

    # print size of non pruned hub
    def consideration_hub_size(self) -> uint32_t:
        return self.consider.get_total()

    # return a random snp from the consideration hub
    def get_random_considered_snp(self, rng: rng_t, anchor: snp_t) -> snp_t:
        return self.consider.get_ran_snp(rng, anchor, self.mutation_tries)

    # return random snp from a list of choices
    def get_random_snp_from_list(self, rng: rng_t, anchor: snp_t, snps: List[snp_t]) -> snp_t:
        # if no snps were collected, return a random snp from consideration hub
        if len(snps) == 0:
            return self.get_random_considered_snp(rng, anchor)

        # try to get a random snp that is not the same as the input snp
        choice = rng.choice(snps)
        assert choice != anchor, "SNP should not be the same as the anchor SNP"
        return choice

    # return a random snp weighted by r2 from a list of choices
    def get_random_snp_weighted_by_r2(self, rng: rng_t, anchor: snp_t, snps: List[snp_t], r2_list: List[float32_t]) -> snp_t:
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
        return self.consider.get_keys_with_snps()

    # get original feature from the snp hub (ray id)
    def get_ori_ray_id(self, snp: snp_t) -> ray.ObjectID:
        return self.db.get_ori_ray_id(snp)

    # get encoded feature from the snp hub (ray id)
    def get_enc_ray_id(self, snp: snp_t) -> ray.ObjectID:
        return self.db.get_enc_ray_id(snp)

    # count number of unseen snps in the hub from non_pruned object
    def seen_snps_proportion(self) -> None:
        unseen_count = 0
        for chrm in self.consider.hub:
            for pos in self.consider.hub[chrm]:
                # count if the snp have not been seen yet
                if self.db.get_seen_flag(snp_t(f"{chrm}.{pos}")) == False:
                    unseen_count += 1

        # print proportion of unseen snps
        print(f"% of unseen SNPs: {unseen_count/len(self.db.hub):.2%}", flush=True)
        print(f"% of SNPs still considered: {self.consider.get_total()/len(self.db.hub):.2%}", flush=True)
        return

    def get_unseen_snps(self, snps: Set[snp_t]) -> Set[snp_t]:
        """
        Function to get all unseen snps from the pipelines.

        Parameters:
        snps: Set of univariate snps

        Returns:
        Set: A set of unseen univariate snps
        """
        unseen_univariates = set()
        for snp_name in snps:
            if self.db.get_seen_flag(snp_name) == False:
                unseen_univariates.add(snp_name)
        return unseen_univariates

    def remove_inactive_branches(self, branches: Set) -> Set:
        """
        Function to remove inactive branches from a given set of branches.

        Parameters:
        branches: Set of branches (univariate snps or interactions (K2, K3, ...))
        hub: An interface to a branch hub to get branch specific information

        Returns:
        Set: An updated set of active branches
        """
        good_branches = set()
        for branch in branches:
            if self.db.get_active_flag(branch):
                good_branches.add(branch)
        return good_branches
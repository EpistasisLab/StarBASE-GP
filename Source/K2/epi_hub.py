#####################################################################################################
#
# Hub Interface for K1 that will hold all SNP information and provide fast querying of SNPs
#
#####################################################################################################

from ..Base.hub import Hub
from ..Base.ordered_hub import Ordered_Hub
from ..Base.types import (snp_t, int32_t, rng_t, uint32_t, float32_t, int16_t, uint16_t, interaction_t)
from ..Base.utils import snp_chrm_pos
from ..Base.considered import Considered

from typeguard import typechecked
from typing import List, Dict, Set
import numpy as np
import ray
import time

# todo
# - add pager lut to the epi hub and update the add_interaction_to_hub function to take in pager lut and store it in the epi hub

@typechecked
class K2_Hub(Hub):

    class EPI_DB:
        """
        Data base to hold all snp information.
        """
        def __init__(self):
            """
            self.hub: dictionary to hold all snp and values (assuming that all snps are already in the hub)

            Order of values in the list per SNP:
                (0) idx (int32_t): corresponding index of the snp position in the sorted list within Ordered_Hub
                (1) ori_rid (ray.ObjectID): ray id for original feature valeus of the corresponding snp in the hub value list, default = None
            """

            # {snp: [idx, ori_rid]}
            self.hub = {}

        def add_interaction_to_hub(self,
                       interaction: interaction_t,
                       r2: float32_t,
                       enc_rid: ray.ObjectID | None,
                       enc_x: snp_t | None,
                       gen_seen: int16_t,
                       active: bool,
                       pager_lut) -> None:
            """
            Process Args and add to hub.

            Args:
                (k) interaction (interaction_t): chrm.pos1:chrm.pos2 string
                (0) r2 (float32_t): r2 value of the interaction
                (1) enc_rid (ray.ObjectID): ray id for encoded feature values of the corresponding interaction in the hub value list
                (2) enc_x (snp_t): the encoding type for the interaction
                (3) gen_seen (int16_t): generation in which the interaction was seen
                (4) active (bool): active value of the interaction
                (5) pager_lut (Dict[snp_t, int32_t]): lookup table for pager values of each snp in the interaction #todo
            """

            # add to hub
            self.hub[interaction] = [r2, enc_rid, enc_x, gen_seen, active, pager_lut]
            return

        def get_r2(self, interaction: interaction_t) -> float32_t:
            # assert that interaction is in hub
            assert interaction in self.hub
            # return r2 value
            return self.hub[interaction][0]

        def get_enc_ray_id(self, interaction: interaction_t) -> ray.ObjectID:
            # assert that interaction is in hub
            assert interaction in self.hub
            # return ray id for encoded feature values
            return self.hub[interaction][1]

        def get_enc_x(self, interaction: interaction_t) -> snp_t | None:
            # assert that interaction is in hub
            assert interaction in self.hub
            # return encoding type for the interaction
            return self.hub[interaction][2]

        def get_gen_seen(self, interaction: interaction_t) -> int16_t:
            # assert that interaction is in hub
            assert interaction in self.hub
            # return generation seen
            return self.hub[interaction][3]

        def get_active(self, interaction: interaction_t) -> bool:
            # assert that interaction is in hub
            assert interaction in self.hub
            # return active value
            return self.hub[interaction][4]

        def get_pager_lut(self, interaction: interaction_t) -> Dict[snp_t, int32_t] | None:
            # assert that interaction is in hub
            assert interaction in self.hub
            # return pager lut
            return self.hub[interaction][5]

        def does_interaction_exist(self, interaction: interaction_t) -> bool:
            return interaction in self.hub

    class SNP_DB:
        """
        Data base to hold all snp information.
        """
        def __init__(self):
            """
            self.hub: dictionary to hold all snp and values (assuming that all snps are already in the hub)

            Order of values in the list per SNP:
                (0) idx (int32_t): corresponding index of the snp position in the sorted list within Ordered_Hub
                (1) ori_rid (ray.ObjectID): ray id for original feature valeus of the corresponding snp in the hub value list, default = None
            """

            # {snp: [idx, ori_rid]}
            self.hub = {}

        # will add snp, idx, ori_rid to the hub
        def add_to_hub(self,
                       snp: snp_t,
                       ori_rid: ray.ObjectID,
                       left_neighbor: snp_t | None,
                       right_neighbor: snp_t | None) -> None:
            """
            Process Args and add to hub.

            Args:
                (k) snp (snp_t): chrm.pos string
                (0) ori_rid (ray.ObjectID): ray id for original feature valeus of the corresponding snp in the hub value list
                (1) left_neighbor: (snp_t): snp name of the left neighbor in the same chromosome
                (2) right_neighbor: (snp_t): snp name of the right neighbor in the same chromosome
            """

            # add to hub
            self.hub[snp] = [ori_rid, left_neighbor, right_neighbor]
            return

        def get_ori_ray_id(self, snp: snp_t) -> ray.ObjectID:
            # assert that snp is in hub
            assert snp in self.hub
            # return data
            return self.hub[snp][0]

        def get_neighbors(self, snp: snp_t) -> List[snp_t]:
            # assert that snp is in hub
            assert snp in self.hub

            if self.hub[snp][1] is not None and self.hub[snp][2] is not None:
                return [self.hub[snp][1], self.hub[snp][2]]
            elif self.hub[snp][1] is not None:
                return [self.hub[snp][1]]
            elif self.hub[snp][2] is not None:
                return [self.hub[snp][2]]
            else:
                assert False, f"SNP should have at least one neighbor in the hub. Got {snp} with neighbors {self.hub[snp][1]} and {self.hub[snp][2]}"

    def __init__(self, snp_list: List[snp_t], snps_ray_ids:Dict[snp_t, ray.ObjectRef]) -> None:
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
        self.consider = Considered(snp_list)
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
        self.snp_db = self.SNP_DB()
        db_create_time = time.time() - db_start
        print(f"  - DB object creation: {db_create_time:.4f} seconds", flush=True)

        # epi db hub stuff
        print('Initializing Epistasis Database Hub')
        epi_db_start = time.time()
        self.epi_db = self.EPI_DB()
        epi_db_time = time.time() - epi_db_start
        print(f"  - EPI_DB object creation: {epi_db_time:.4f} seconds", flush=True)

        # update snp_hub with snp_bin and snp header positions
        populate_start = time.time()
        # iterate through all the chromosomes and positions in the order hub
        for chr, positions in self.order.order.items():
            for idx, pos in enumerate(positions):
                snp_name = snp_t(f"{chr}.{pos}")
                # get ray id for the snp
                ori_rid = snps_ray_ids[snp_name]
                # get left and right neighbors
                left_neighbor = snp_t(f"{chr}.{positions[idx-1]}") if idx > 0 else None
                right_neighbor = snp_t(f"{chr}.{positions[idx+1]}") if idx < len(positions) - 1 else None
                # add to hub
                self.snp_db.add_to_hub(snp_name, ori_rid, left_neighbor, right_neighbor)
        populate_time = time.time() - populate_start

        print(f"  - SNP_DB population with {len(snp_bin)} SNPs: {populate_time:.4f} seconds", flush=True)
        print(f'SNP Hub Initialized in {time.time() - db_start:.4f} seconds\n')

        total_time = time.time() - hub_init_start
        print(f'=== Total K2_Hub Initialization Time: {total_time:.4f} seconds ===\n', flush=True)
        return

    def add_interaction_to_hub(self,
                       interaction: interaction_t,
                       r2: float32_t,
                       enc_rid: ray.ObjectID | None,
                       enc_x: snp_t | None,
                       gen_seen: int16_t,
                       explainability_threshold: float32_t,
                       pager_lut) -> None:
        """
        Update SNP hub with the r2 and encoding type & vector (if applicable).

        Args:
            interaction (interaction_t): Interaction to be updated.
            r2 (float32_t): R2 value for the interaction.
            enc_rid (ray.ObjectID): Ray object ID for the encoded interaction values.
            enc_x (snp_t): The encoding type for the interaction.
            gen_seen (int16_t): Generation when the interaction was seen.
            explainability_threshold (float32_t): Threshold for SNP explainability.
            pager_lut (np.ndarray | None): PAGER LUT values if encoding is 'pager'.
        """

        # if r2 is below the explainability threshold and the threshold is non-negative, add hub with active flag
        self.epi_db.add_interaction_to_hub(interaction=interaction,
                                           r2=r2,
                                           enc_rid=enc_rid,
                                           enc_x=enc_x,
                                           gen_seen=gen_seen,
                                           active=False if r2 < explainability_threshold and explainability_threshold >= float32_t(0.0) else True,
                                           pager_lut=pager_lut)
        return

    def get_encoding(self, snp: snp_t) -> snp_t:
        # get best type of encoder for a given snp
        return self.db.get_encoding(snp)

    def get_r2(self, interaction: interaction_t) -> float32_t:
        # get r2 for a given interaction from epi hub
        return self.epi_db.get_r2(interaction)

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

    def get_ran_interaction(self, rng: rng_t) -> interaction_t:
        """
        Get a random interaction composed of completely random snps.

        Args:
            rng (rng_t): Numpy random generator.

        Returns:
            interaction_t: A tuple of two SNPs that interact with each other.
        """
        # get first random snp from considered hub
        snp1 = self.consider.get_ran_snp(rng, snp_t('nada'), self.mutation_tries)
        # get second random snp from considered hub based on the first snp
        snp2 = self.consider.get_ran_snp(rng, snp1, self.mutation_tries)

        # return the interaction as a tuple of the two snps with snp_x < snp_y for consistency
        return (snp1, snp2) if snp1 < snp2 else (snp2, snp1)

    def get_ran_snp_in_window(self, anchor: snp_t, rng: rng_t) -> snp_t:
        """
        Get a random SNP from the same chromosome and within the specified neighbor set in SNP_DB.

        Args:
            anchor (snp_t): Anchor SNP in "chromosome.position" format.
            rng (rng_t): Numpy random generator.

        Returns:
            snp_t: A randomly selected SNP from the same chromosome and within the specified neighbor set in SNP_DB
        """

        # make sure there is a '-' inside the snp string
        assert '.' in anchor

        # get neibhors for the anchor snp from the snp hub
        neighbors = self.snp_db.get_neighbors(anchor)
        assert len(neighbors) > 0, f"Anchor SNP {anchor} should have at least one neighbor in the SNP_DB to get a random SNP in the window. Got neighbors: {neighbors}"

        # roll a random snp from the list of snps
        return rng.choice(neighbors)

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

    def get_active_flag(self, interaction: interaction_t) -> bool:
        # is this interaction active?
        return self.epi_db.get_active(interaction)

    #todo: what do we want to store in the hub?
    def process_pruned_interactions(self, interactions: Set[interaction_t], snp_details_after_ld: Dict[snp_t, Dict], gen_pruned: int16_t) -> None:
        """
        Process pruned interactions by updating their status in the interaction hub and removing them from the consideration hub.

        Args:
            interactions (Set[interaction_t]): Set of interactions that have been pruned.
            snp_details_after_ld (Dict[snp_t, Dict]): Dictionary containing details for each pruned SNP, including reason, threshold, genomic distance, and anchor SNP.
            gen_pruned (int16_t): Generation number when the interactions were pruned.
        """

        process_start = time.time()
        print(f"[Timing] Processing {len(interactions)} pruned interactions...", flush=True)

        # go through each interaction and update the hub
        flip_total = 0.0
        add_details_total = 0.0
        remove_total = 0.0

        for interaction in interactions:
            # get the snp from the interaction
            snp = self.db.get_snp_from_interaction(interaction)

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
        print(f"[Timing] process_pruned_interactions completed: Flip flags={flip_total:.4f}s, Add LD details={add_details_total:.4f}s, Remove from consider={remove_total:.4f}s, Total={total_time:.4f}s", flush=True)

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

    def get_snp_ori_ray_id(self, snp: snp_t) -> ray.ObjectID:
        # get original feature from the snp hub (ray id)
        return self.snp_db.get_ori_ray_id(snp)

    def get_enc_ray_id(self, interaction: interaction_t) -> ray.ObjectID:
        # get encoded feature from the snp hub (ray id)
        return self.epi_db.get_enc_ray_id(interaction)

    def seen_interactions_count(self) -> None:
        """
        Function to print the count of seen interactions in the epi_hub
        """

        # print count of seen interactions in the epi_hub
        print(f"Seen interactions count: {len(self.epi_db.hub)}")
        return

    def get_unseen_interactions(self, interactions: Set[interaction_t]) -> Set[interaction_t]:
        """
        Function to get all unseen interactions from the given set.

        Parameters:
            interactions (Set[interaction_t]): Set of interactions to check for unseen status.

        Returns:
            Set[interaction_t]: A set of unseen interactions
        """
        return {interaction for interaction in interactions if not self.epi_db.does_interaction_exist(interaction)}

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
            if self.epi_db.get_active(branch):
                good_branches.add(branch)
        return good_branches

    def build_component_map(self, interactions: Set[interaction_t]) -> Dict[interaction_t, Dict]:
        """
        Function to build a component map for the given interactions.

        Parameters:
            interactions (Set[interaction_t]): Set of interactions to build the component map for.
        """

        component_map = {}

        for interaction in interactions:
            snp1, snp2 = interaction
            assert snp1 < snp2, f"Interactions should be ordered with snp1 < snp2 for consistency. Got {interaction}."

            component_map[interaction] = {
                'snp1_name': snp1,
                'snp2_name': snp2,
                'snp1_ray_id': self.get_snp_ori_ray_id(snp1),
                'snp2_ray_id': self.get_snp_ori_ray_id(snp2),
                'encoded_ray_id': self.get_enc_ray_id(interaction)
            }


        return component_map

    def generate_r2_set(self, interactions: Set[interaction_t]) -> List:
        """
        Generate a Set of interactions and their corresponding r2 values.

        Args:
            interactions (Set[interaction_t]): Set of interactions to generate r2 values for.

        Returns:
            List: A list of tuples containing interactions and their r2 values.
        """

        # make sure
        assert len(interactions) > 0

        return [(interaction, self.get_r2(interaction)) for interaction in interactions]
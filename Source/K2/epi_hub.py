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
import ray
import time

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
                       mdr_mapping: Dict[tuple, float32_t] | None) -> None:
            """
            Process Args and add to hub.

            Args:
                (k) interaction (interaction_t): chrm.pos1:chrm.pos2 string
                (0) r2 (float32_t): r2 value of the interaction
                (1) enc_rid (ray.ObjectID): ray id for encoded feature values of the corresponding interaction in the hub value list
                (2) enc_x (snp_t): the encoding type for the interaction
                (3) gen_seen (int16_t): generation in which the interaction was seen
                (4) active (bool): active value of the interaction
                (5) mdr_mapping (Dict[tuple, float32_t] | None): lookup table for MDR values of each snp in the interaction
                (6) gen_pruned (int16_t): generation when the interaction was pruned (-1 if not pruned)
                (7) pruned_reason (str): reason for pruning ("LD" or "CA" or "PE" or "" if not pruned)
                (8) ld_threshold (float32_t): LD threshold used for pruning
                (9) ld_genomic_distance (int32_t): genomic distance used for LD pruning
                (10) anchor_interaction (interaction_t or str): anchor interaction if pruned, empty string otherwise
            """
            # assert that the interaction does not contain the same snp twice (e.g., chr1.1000:chr1.1000)
            assert interaction[0] != interaction[1], f"Interaction {interaction} should not contain the same SNP twice."

            # add to hub with LD pruning fields initialized
            self.hub[interaction] = [r2, enc_rid, enc_x, gen_seen, active, mdr_mapping, int16_t(-1), "", float32_t(-1.0), int32_t(-1), ""]
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

        def get_mdr_mapping(self, interaction: interaction_t) -> Dict[tuple, float32_t] | None:
            # assert that interaction is in hub
            assert interaction in self.hub
            # return mdr mapping
            return self.hub[interaction][5]

        def does_interaction_exist(self, interaction: interaction_t) -> bool:
            return interaction in self.hub

        def get_gen_pruned(self, interaction: interaction_t) -> int16_t:
            """Get the generation when the interaction was pruned."""
            assert interaction in self.hub
            return self.hub[interaction][6]

        def flip_activate_flag_ld(self, interaction: interaction_t, gen_pruned: int16_t) -> None:
            """Mark an interaction as inactive due to LD pruning."""
            assert interaction in self.hub
            # Set active flag (index 4) to False and gen_pruned (index 6) to the generation
            self.hub[interaction][4] = False
            self.hub[interaction][6] = gen_pruned
            return

        def flip_activate_flag_pe(self, interaction: interaction_t, gen_pruned: int16_t) -> None:
            """Mark an interaction as inactive due to poor performance pruning."""
            assert interaction in self.hub
            # Set active flag (index 4) to False and gen_pruned (index 6) to the generation
            self.hub[interaction][4] = False
            self.hub[interaction][6] = gen_pruned
            self.hub[interaction][7] = "PE"
            return

        def flip_activate_flag_pre(self, interaction: interaction_t, gen_pruned: int16_t) -> None:
            """Mark an interaction as inactive due to poor performance pruning."""
            assert interaction in self.hub
            # Set active flag (index 4) to False and gen_pruned (index 6) to the generation
            self.hub[interaction][4] = False
            self.hub[interaction][6] = gen_pruned
            self.hub[interaction][7] = "PRE"
            return

        def add_ld_details(self, interaction: interaction_t, reason: str, threshold: float,
                          genomic_distance: int, anchor_interaction: interaction_t | str) -> None:
            """Add LD pruning details to the interaction hub."""
            assert interaction in self.hub
            # Update LD pruning details
            # Index 7: pruned_reason
            # Index 8: ld_threshold
            # Index 9: ld_genomic_distance
            # Index 10: anchor_interaction
            self.hub[interaction][7] = reason
            self.hub[interaction][8] = float32_t(threshold)
            self.hub[interaction][9] = int32_t(genomic_distance)
            self.hub[interaction][10] = anchor_interaction
            return

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
                       mdr_mapping) -> None:
        """
        Update SNP hub with the r2 and encoding type & vector (if applicable).

        Args:
            interaction (interaction_t): Interaction to be updated.
            r2 (float32_t): R2 value for the interaction.
            enc_rid (ray.ObjectID): Ray object ID for the encoded interaction values.
            enc_x (snp_t): The encoding type for the interaction.
            gen_seen (int16_t): Generation when the interaction was seen.
            explainability_threshold (float32_t): Threshold for SNP explainability.
            mdr_mapping (Dict | None): MDR mapping values if encoding is 'mdr'.
        """

        # if r2 is below the explainability threshold and the threshold is non-negative, add hub with active flag
        self.epi_db.add_interaction_to_hub(interaction=interaction,
                                           r2=r2,
                                           enc_rid=enc_rid,
                                           enc_x=enc_x,
                                           gen_seen=gen_seen,
                                           active=True,
                                           mdr_mapping=mdr_mapping)
        return

    def get_encoding(self, interaction: interaction_t) -> snp_t:
        # get best type of encoder for a given interaction
        return self.epi_db.get_enc_x(interaction)

    def get_r2(self, interaction: interaction_t) -> float32_t:
        # get r2 for a given interaction from epi hub
        return self.epi_db.get_r2(interaction)

    # save the epi_hub and snp_hub to a file
    def save_hubs(self, save_dir: str) -> None:
        """
        Save the interaction hub to a CSV file.

        EPI_DB structure (v = self.epi_db.hub[interaction]):
            v[0]: r2 (float32_t)
            v[1]: enc_rid (ray.ObjectID - not saved)
            v[2]: enc_x (snp_t - encoding type)
            v[3]: gen_seen (int16_t)
            v[4]: active (bool)
            v[5]: pager_lut (Dict or None)
            v[6]: gen_pruned (int16_t)
            v[7]: pruned_reason (str)
            v[8]: ld_threshold (float32_t)
            v[9]: ld_genomic_distance (int32_t)
            v[10]: anchor_interaction (interaction_t or str)
        """

        save_start = time.time()
        print(f"[Timing] Starting save_hubs to {save_dir}...", flush=True)

        # Save interaction hub with headers
        collect_start = time.time()
        interaction_data = []
        for k, v in self.epi_db.hub.items():
            # k: interaction tuple (snp1, snp2)
            # Extract SNP1 and SNP2 from the interaction tuple
            snp1, snp2 = k

            # Extract MDR mapping from mdr_mapping if it exists and is a dict
            mdr_mapping = v[5]
            if mdr_mapping is not None and isinstance(mdr_mapping, dict):
                # mdr_mapping maps genotype combos to risk values
                # Use repr() to get a properly escaped string representation
                mdr_str = repr(mdr_mapping)
            else:
                mdr_str = ''

            # Format anchor_interaction (v[10]) - could be tuple or string
            anchor_interaction = v[10]
            if isinstance(anchor_interaction, tuple):
                anchor_str = f"{anchor_interaction[0]}:{anchor_interaction[1]}"
            else:
                anchor_str = str(anchor_interaction)

            # Append row: [interaction_str, snp1, snp2, r2, encoding, gen_seen, active, gen_pruned, pruned_reason, ld_threshold, ld_genomic_distance, anchor_interaction, mdr_mapping]
            interaction_str = f"{snp1}:{snp2}"
            interaction_data.append([
                interaction_str,  # Combined interaction name
                snp1,            # First SNP
                snp2,            # Second SNP
                v[0],            # r2
                v[2],            # enc_x (encoding type)
                v[3],            # gen_seen
                v[4],            # active
                v[6],            # gen_pruned
                v[7],            # pruned_reason
                v[8],            # ld_threshold
                v[9],            # ld_genomic_distance
                anchor_str,      # anchor_interaction
                mdr_str          # mdr_mapping
            ])

        collect_time = time.time() - collect_start
        print(f"  - Data collection: {collect_time:.4f}s for {len(interaction_data)} interactions", flush=True)

        # Sort interaction_data by r2 (index 3 in the row)
        sort_start = time.time()
        interaction_data.sort(key=lambda x: x[3], reverse=True)  # reverse=True for descending order by r2
        sort_time = time.time() - sort_start
        print(f"  - Data sorting: {sort_time:.4f}s", flush=True)

        # Write interaction hub to file
        write_start = time.time()
        with open(save_dir+"interaction_hub.csv", 'w') as f:
            # Write the headers
            f.write("interaction,snp1,snp2,r2,encoding,gen_seen,active,gen_pruned,pruned_reason,ld_threshold,ld_genomic_distance,anchor_interaction,mdr_mapping\n")
            for row in interaction_data:
                # Add 'chr' prefix to interaction components
                # snp1_chr, snp1_pos = row[1].split('.')
                # snp2_chr, snp2_pos = row[2].split('.')
                interaction_with_chr = f"chr{row[1]}:chr{row[2]}"

                # Escape mdr_mapping field by wrapping in quotes (last field - row[12])
                mdr_mapping_escaped = row[12].replace('"', '""') if row[12] else ''  # Escape quotes by doubling them

                # Write all columns, with mdr_mapping wrapped in quotes to handle commas
                f.write(f"{interaction_with_chr},chr{row[1]},chr{row[2]},{row[3]},{row[4]},{row[5]},{row[6]},{row[7]},{row[8]},{row[9]},{row[10]},{row[11]},\"{mdr_mapping_escaped}\"\n")

        write_interaction_time = time.time() - write_start
        print(f"  - Writing interaction_hub.csv: {write_interaction_time:.4f}s", flush=True)

        # save csv with both seen and active (not pruned) interactions
        # Write consideration hub to file
        consider_write_start = time.time()
        with open(save_dir+"consideration_hub.csv", 'w') as f:
            # Write the headers
            f.write("interaction,r2,encoding\n")
            for row in interaction_data:
                # Check if active (index 6) is True
                if row[6] == True:
                    # Add 'chr' prefix to interaction
                    interaction_with_chr = f"chr{row[1]}:chr{row[2]}"
                    f.write(f"{interaction_with_chr},{row[3]},{row[4]}\n")

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

    def process_pruned_interactions(self, interactions: Set[interaction_t], snp_details_after_ld: Dict[interaction_t, Dict], gen_pruned: int16_t) -> None:
        """
        Process pruned interactions by updating their status in the interaction hub and removing them from the consideration hub.

        Args:
            interactions (Set[interaction_t]): Set of interactions that have been pruned.
            snp_details_after_ld (Dict[interaction_t, Dict]): Dictionary containing details for each pruned interaction, including reason, threshold, genomic distance, and anchor SNP.
            gen_pruned (int16_t): Generation number when the interactions were pruned.
        """

        process_start = time.time()
        print(f"[Timing] Processing {len(interactions)} pruned interactions...", flush=True)

        # go through each interaction and update the hub
        flip_total = 0.0
        add_details_total = 0.0
        remove_total = 0.0

        for interaction in interactions:
            # check to make sure we have not prunned this interaction before
            assert self.epi_db.get_gen_pruned(interaction) == int16_t(-1)

            # flip interaction to pruned
            flip_start = time.time()
            self.epi_db.flip_activate_flag_ld(interaction, gen_pruned)
            flip_total += time.time() - flip_start

            # add ld details to the snp hub
            add_start = time.time()
            self.epi_db.add_ld_details(
                interaction,
                snp_details_after_ld[interaction]["reason"],
                snp_details_after_ld[interaction]["threshold"],
                snp_details_after_ld[interaction]["genomic_distance"],
                snp_details_after_ld[interaction]["anchor_interaction"]
            )
            add_details_total += time.time() - add_start

            # delete interaction from non pruned
            remove_start = time.time()
            self.consider.remove_interaction(interaction)
            remove_total += time.time() - remove_start

        total_time = time.time() - process_start
        print(f"[Timing] process_pruned_interactions completed: Flip flags={flip_total:.4f}s, Add LD details={add_details_total:.4f}s, Remove from consider={remove_total:.4f}s, Total={total_time:.4f}s", flush=True)

    def at_least_one_active_snp(self, snps: List[interaction_t]) -> bool:
        """
        Function to check if at least one interaction in the list is active.

        Parameters:
            snps (List[interaction_t]): List of interactions to check (parameter name kept as 'snps' for compatibility, but actually interactions)

        Returns:
            bool: True if at least one interaction is active, False otherwise
        """

        # if one interaction is active return true
        for interaction in snps:
            if self.epi_db.get_active(interaction):
                return True
        # return false if all interactions are inactive
        return False

    def at_least_one_active_interaction(self, interactions: List[interaction_t]) -> bool:
        """
        Function to check if at least one interaction in the provided list is active (not pruned).

        Parameters:
            interactions (List[interaction_t]): List of interactions to check

        Returns:
            bool: True if at least one interaction is active, False otherwise
        """

        # Check if at least one interaction in the list is active
        for interaction in interactions:
            if self.epi_db.get_active(interaction):
                return True

        # Return false if all interactions are inactive
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

    def flip_active_flag_pe(self, interaction: interaction_t, gen_pruned: int16_t) -> None:
        """
        Function to flip the active flag of an interaction in the epi_hub.

        Parameters:
            interaction (interaction_t): The interaction for which to flip the active flag.
            gen_pruned (int16_t): The generation in which the interaction was pruned.
        """

        self.epi_db.flip_activate_flag_pe(interaction, gen_pruned)
        return

    def flip_active_flag_pre(self, interaction: interaction_t, gen_pruned: int16_t) -> None:
        """
        Function to flip the active flag of an interaction in the epi_hub.

        Parameters:
            interaction (interaction_t): The interaction for which to flip the active flag.
            gen_pruned (int16_t): The generation in which the interaction was pruned.
        """

        self.epi_db.flip_activate_flag_pre(interaction, gen_pruned)
        return

    def get_epi_db_size(self) -> uint32_t:
        """
        Function to get the size of the epi_db.

        Returns:
            uint32_t: The number of interactions stored in the epi_db.
        """
        return uint32_t(len(self.epi_db.hub))

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
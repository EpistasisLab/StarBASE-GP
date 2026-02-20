#####################################################################################################
#
# Container to maintain snps that have not been been flagged as inactive throughout the evolutionary process.
# This is used to ensure that we do not select snps that have been pruned by and get a speedup when selecting random snps.
# Hub in this class is broken down by chromosome and and sorted positions for fast querying of snps within a given distance.
#
#####################################################################################################


from ..Base.types import (snp_t, int32_t, rng_t, uint32_t, uint16_t)
from ..Base.utils import snp_chrm_pos

import time
from typing import List, Dict

# SortedList is a sorted list implementation in Python for fast insertion and deletion
# https://grantjenks.com/docs/sortedcontainers/sortedlist.html
from sortedcontainers import SortedList

class Considered:
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

    def calculate_nearest_neighbor(self, chrom: int32_t, position: int32_t) -> List[snp_t]:
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
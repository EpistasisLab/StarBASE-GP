#####################################################################################################
#
# This class provides stardard hub that is used to manage snps ordered by their chromosome and position.
# Class can be used to get snps within a given distance from a given snp.
# Users can incoprate this functionality into their own derived HUB class if needed (recommended).
#
#####################################################################################################

from typeguard import typechecked
from typing import List, Tuple
from .types import (snp_t, int32_t, uint32_t)
from .utils import snp_chrm_pos

@typechecked
class Ordered_Hub:
    def __init__(self) -> None:
        # {chrom: [pos1, pos2, ...]}
        # pos1 < pos2 < ... < posN
        self.order = {}
        return

    def get_in_window_positions(self, chrom: int32_t, position: int32_t, idx: int32_t, left_idx: int32_t, right_idx: int32_t) -> List[int32_t]:
        """
        Given an anchor SNP's chromosome number and position, return all positions within the left index and right index.
        Remove the original position from the in window list.

        Args:
            chrom (int32_t): chromosome number for the SNP
            position (int32_t): base pair position for the SNP
            idx (int32_t): index of the SNP in the ordered list for the chromosome (hub tracks this)
            left_idx (int32_t): left index in the ordered list that falls within the given distance from the SNP position (hub tracks this)
            right_idx (int32_t): right index in the ordered list that falls within the given distance from the SNP position (hub tracks this)
        Returns:
            List[int32_t]: List of positions within the left and right indices, excluding the original position.
        """

        assert chrom in self.order
        assert 0 <= idx < len(self.order[chrom])
        assert self.order[chrom][idx] == position, "Position does not match the index provided."
        assert 0 <= left_idx <= idx
        assert idx <= right_idx < len(self.order[chrom])
        assert left_idx <= right_idx

        in_window_positions = self.order[chrom][left_idx:right_idx + 1]
        in_window_positions.remove(position)
        return in_window_positions

    def get_out_window_positions(self, chrom: int32_t, position: int32_t, idx: int32_t, left_idx: int32_t, right_idx: int32_t) -> List[int32_t]:
        """
        Given an anchor SNP's chromosome number and position, return all positions outside the left index and right index.

        Args:
            chrom (int32_t): chromosome number for the SNP
            position (int32_t): base pair position for the SNP
            idx (int32_t): index of the SNP in the ordered list for the chromosome (hub tracks this)
            left_idx (int32_t): left index in the ordered list that falls within the given distance from the SNP position (hub tracks this)
            right_idx (int32_t): right index in the ordered list that falls within the given distance from the SNP position (hub tracks this)
        Returns:
            List[int32_t]: List of positions outside the left and right indices.
        """

        assert chrom in self.order
        assert 0 <= idx < len(self.order[chrom])
        assert self.order[chrom][idx] == position, "Position does not match the index provided."
        assert 0 <= left_idx <= idx
        assert idx <= right_idx < len(self.order[chrom])
        assert left_idx <= right_idx

        out_window_positions = []
        # positions before left_idx
        out_window_positions.extend(self.order[chrom][0:left_idx])
        # positions after right_idx
        out_window_positions.extend(self.order[chrom][right_idx + 1:])

        return out_window_positions

    def count_order_objs(self) -> uint32_t:
        """
        Count the total number of objects in the order dictionary.

        Returns:
            uint32_t: Total number of objects in the order dictionary.
        """
        count = uint32_t(0)
        for _, order in self.order.items():
            count += uint32_t(len(order))
        return count

    def get_window_indices(self, chrom: int32_t, position: int32_t, idx: int32_t, distance: int32_t) -> Tuple[int32_t, int32_t]:
        """
        Given an snp position and the left and right indicies in the ordered list that fall within the given distance from the position.

        Args:
            chrom (int32_t): chromosome number for the SNP
            position (int32_t): base pair position for the SNP
            idx (int32_t): index of the SNP in the ordered list for the chromosome (hub tracks this)
            distance (int32_t): distance from the SNP position to search for other SNPs

        Returns:
            Tuple[int32_t, int32_t]: left and right indices in the ordered list that fall within the given distance from the SNP position.
        """

        # make sure the chromosome exists
        assert chrom in self.order
        # make sure idx is within the bounds of the order dictionary for a chromosome
        assert 0 <= idx < len(self.order[chrom])
        # make sure the position exists at the given index
        assert self.order[chrom][idx] == position, "Position does not match the index provided."
        # make sure distance is positive
        assert distance > 0

        # find left index
        left_idx = idx - 1
        while left_idx >= 0 and abs(self.order[chrom][idx] - self.order[chrom][left_idx]) <= distance:
            left_idx -= 1
        left_idx += 1  # adjust to the last valid index

        # find right index
        right_idx = idx + 1
        while right_idx < len(self.order[chrom]) and abs(self.order[chrom][idx] - self.order[chrom][right_idx]) <= distance:
            right_idx += 1
        right_idx -= 1  # adjust to the last valid index

        # make sure left and right are not out of bounds
        assert 0 <= left_idx <= idx
        assert idx <= right_idx < len(self.order[chrom])
        assert left_idx <= right_idx

        return int32_t(left_idx), int32_t(right_idx)

    def generate_order(self, snps: List[snp_t]) -> List[Tuple[snp_t, int32_t]]:
        """
        Given a list of SNPs, generate the order dictionary and return a list of tuples
        containing each SNP and its index in the sorted order for its chromosome.

        Args:
            snps (List[snp_t]): List of SNPs to process.

        Returns:
            List[Tuple[snp_t, int32_t]]: List of tuples containing each SNP and its index in the sorted order for its chromosome.
        """

        # quick checks
        assert len(snps) > 0

        # load all snps into the order dictionary with chromosome as key and the values as lists of positions
        for snp in snps:
            chrom, pos = snp_chrm_pos(snp)
            if chrom not in self.order:
                self.order[chrom] = []
            self.order[chrom].append(pos)

        # sort all lists within the order dictionary
        for chrom, order in self.order.items():
            self.order[chrom] = sorted(order)

        # collect each snps bin number
        snp_bins = []
        # go through self.order and collect all snps, bin_num
        for chrom, order in self.order.items():
            for idx, pos in enumerate(order):
                snp_bins.append((snp_t(f"{chrom}.{pos}"), int32_t(idx)))

        # cast all bins to list for search efficiency
        for chrom, o in self.order.items():
            self.order[chrom] = [int32_t(p) for p in o]

        # make sure all SNPs are accounted for
        assert len(snps) == self.count_order_objs()
        # make sure snp_bins is the correct size
        assert len(snp_bins) == len(snps)

        return snp_bins # [(snp, idx_in_sorted_list), ...]
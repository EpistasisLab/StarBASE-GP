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

    # get window for a given SNP
    def get_window(self, chrom: int32_t, position: int32_t, idx: int32_t, distance: int32_t) -> List[int32_t]:
        # make sure the chromosome exists
        assert chrom in self.order
        # make sure idx is within the bounds of the order dictionary for a chromosome
        assert 0 <= idx < len(self.order[chrom])
        # make sure the position exists at the given index
        assert self.order[chrom][idx] == position, "Position does not match the index provided."
        # make sure distance is positive
        assert distance > 0

        # get all positions that fall within the distance from the given index
        window_list = []
        anchor = self.order[chrom][idx]
        assert snp_t(f"{chrom}.{anchor}") == snp_t(f"{chrom}.{position}"), "Position does not match the index provided."

        # get all positions to the left of the anchor position
        left_bound = idx - 1
        if left_bound >= 0:
            while abs(anchor - self.order[chrom][left_bound]) <= distance:
                window_list.append(self.order[chrom][left_bound])
                left_bound -= 1
                if left_bound < 0:
                    break

        # get all positions to the right of the anchor position
        right_bound = idx + 1
        if right_bound < len(self.order[chrom]):
            while abs(anchor - self.order[chrom][right_bound]) <= distance:
                window_list.append(self.order[chrom][right_bound])
                right_bound += 1
                if right_bound >= len(self.order[chrom]):
                    break

        return window_list

    # count the total number of objects in the order dictionary
    def count_order_objs(self) -> uint32_t:
        count = uint32_t(0)
        for _, order in self.order.items():
            count += uint32_t(len(order))
        return count

    # create ordered lists for each chromosome based on the snps provided
    def generate_order(self, snps: List[snp_t]) -> List[Tuple[snp_t, int32_t]]:
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
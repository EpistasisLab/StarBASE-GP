#####################################################################################################
#
# This class of helper functions used in various across various files.
#
#####################################################################################################

from .types import (snp_t, int32_t)
from typing import Tuple

def snp_chrm_pos(snp: snp_t) -> Tuple[int32_t, int32_t]:
    """
    Given a SNP in the format 'chrom.position', return the chromosome and position as integers.

    Args:
        snp (snp_t): SNP in the format 'chrom.position' (e.g. 1.12345)

    Returns:
        Tuple[int32_t, int32_t]: A tuple containing the chromosome and position as integers.
    """

    assert '.' in snp, "SNP must be in the format 'chrom.position' (e.g. 1.12345)."
    chrom, pos = snp.split('.')
    chrom, pos = int32_t(chrom), int32_t(pos)
    assert 1 <= chrom <= 22, "Chromosome number must be between 1 and 22."
    assert 1 <= pos <= 1000000000, "Position must be between 1 and 1000000000."
    return chrom, pos
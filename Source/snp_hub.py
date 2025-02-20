#####################################################################################################
#
# Interface for communicating with both EPI and SNP hubs.
#
#####################################################################################################

import numpy as np
from typing import List, Tuple, Set

from typeguard import typechecked
import numpy.typing as npt
from typing import List, Dict

# https://grantjenks.com/docs/sortedcontainers/sortedlist.html
from sortedcontainers import SortedList

### Static variables
mutation_tries = 20

### General Types

# chromosome number
gen_chrom_num_t = np.uint8
# chromosome snp position
gen_chrom_pos_t = np.uint32
# header snps type
gen_header_snps_t = npt.NDArray[np.str_]
# individual snp type
snp_t = np.str_
# numpy random number generator type
rng_t = np.random.Generator
# r2 score type
r2_t = np.float32

### SNP Hub Types

# bin id position type
snp_hub_bin_t = np.uint16
# count type
snp_hub_cnt_t = np.uint32
# header position
snp_hub_pos_t = np.uint32

# generation found/pruned type
snp_hub_gen_t = np.int32

# best individual r2 value type
snp_hub_res_t = np.float32
# best encoder type (in str)
snp_hub_enc_t = np.str_
# corresponding index of the snp position in the bin
snp_hub_idx_t = np.uint16

### Bin Hub Types

# type of object inside bins
bin_hub_arr_t = np.uint32
# type for bin size
bin_hub_size_t = np.uint16

@typechecked
class SnpHub:
    """
    Interface for communicating with Snp hubs and Bin hubs.
    We do not allow for the creation of new hubs, only the updating of existing ones.
    Treat as a private class.
    """

    class Non_Pruned_Hub:
        """
        Structure to keep track of snps that have not been pruned.
        """

        def __init__(self, snps: gen_header_snps_t) -> None:
            """
            Create a dictionary with all snps broken down by chromosome and position.
            Then we save them in a dictionary with the chromosome as the key and the snps in a sorted list.
            The individual lists will be updated by removing snps that have been pruned.
            """

            # create a dictionary to hold all snps
            self.non_pruned = {}

            for snp in snps:
                # split snp up by chromosome and position
                chrom, pos = self.snp_chrm_pos(snp)

                # check if key exists
                if chrom not in self.non_pruned:
                    self.non_pruned[chrom] = [pos]
                else:
                    self.non_pruned[chrom].append(pos)

            # sort all lists within dictionary
            for chrom, pos_l in self.non_pruned.items():
                # sort the list and store in SortedList and update the dictionary
                pos_l.sort()
                self.non_pruned[chrom] = SortedList(pos_l)
            return

        # remove snp from the non_pruned dictionary
        def remove_snp(self, snp: snp_t) -> None:
            # get chromosome and position
            chrom, pos = self.snp_chrm_pos(snp)

            # make sure the chromosome exists
            assert chrom in self.non_pruned

            # remove snp from the list
            self.non_pruned[chrom].remove(pos)

            return

        # print all snps in the non_pruned dictionary
        def print_hub(self) -> None:
            for chrom, pos_l in self.non_pruned.items():
                print(f"Chromosome: {chrom}")
                print(f"Positions: {pos_l}")
            return

        # helper to generate chromosome number and snp position
        def snp_chrm_pos(self, snp: snp_t) -> Tuple[gen_chrom_num_t, gen_chrom_pos_t]:
            chrom, pos = snp.split('.')
            chrom, pos = gen_chrom_num_t(chrom), gen_chrom_pos_t(pos)
            return chrom, pos

        # get a random snp from the hub
        def get_ran_snp(self, rng_: rng_t) -> snp_t:
            # make sure we have at least one snp
            assert self.get_total() > 0

            # initialize rng
            rng = np.random.default_rng(rng_)
            chrom = []

            for chrm in list(self.non_pruned.keys()):
                if len(self.non_pruned[chrm]) > 0:
                    chrom.append(chrm)


            # get a random chromosome key
            chrom = rng.choice(chrom)

            # get a random bin index from the chromosome
            position = rng.choice(self.non_pruned[chrom])

            return snp_t(f"{chrom}.{position}")

        # get total number of items in the non_pruned dictionary
        def get_total(self) -> np.uint32:
            sum = 0
            for _, pos_l in self.non_pruned.items():
                sum += len(pos_l)
            return np.uint32(sum)

        # get list of keys with at least one snp
        def get_keys_with_snps(self) -> List[gen_chrom_num_t]:
            keys = []
            for k, v in self.non_pruned.items():
                if len(v) > 0:
                    keys.append(k)
            return keys

    class Hub:
        """
        Hub to keep track of snp weights.
        """
        def __init__(self):
            """
            self.hub: dictionary to hold all snp and values
            assuming that all snps are already in the hub
            if we get a snp that is not in the hub, we throw an error in debug mode

                   res_pos = 0 # position for r2 recived from evaluation
                   bin_pos = 1 # id for bin assigned to
                   idx_pos = 2 # position for bin number in hub value list
                   pos_pos = 3 # position for header position in hub value list
                   enc_pos = 4 # position for the corresponding encoder types in hub value list
                  seen_pos = 5 # position for the seen flag in hub value list
                pruned_pos = 6 # position for the pruned flag in hub value list
              gen_seen_pos = 7 # position for the general seen flag in hub value list
            gen_pruned_pos = 8 # position for the general pruned flag in hub value list
            """

            # {snp: [res(np.float32),bin(np.uint32),idx(np.uint32),
            # pos(np.uint32),enc(np.str_),seen(bool)], prunned(bool),
            # gen_seen(np.unit32), gen_prunned(np.unit32)}
            self.hub = {}

        # will add snp, sum, bin, pos， idx, res, typ to the hub
        def add_to_hub(self,
                       snp: snp_t,
                       res: snp_hub_res_t,
                       bin: snp_hub_bin_t,
                       idx: snp_hub_idx_t,
                       pos: snp_hub_pos_t,
                       enc: snp_hub_enc_t,
                       seen=False,
                       prunned=False,
                       gen_seen: snp_hub_gen_t=snp_hub_gen_t(-1),
                       gen_pruned:snp_hub_gen_t=snp_hub_gen_t(-1)) -> None:
            """
            will take in a snp, sum, cnt, bin, and pos and add it to the hub

            Args:
                snp (snp_t): chrm.pos string
                res (snp_hub_res_t): best r2 result placeholder, default = -1
                bin (snp_hub_bin_t): what bin number of the snp in
                idx (snp_hub_idx_t): corresponding index of the snp position in the bin
                pos (snp_hub_pos_t): position of the snp in the csv header
                enc (snp_hub_enc_t): best encoder type placeholder, default = ''
                seen (bool): has this snp been seen before, default = False
                prunned (bool): has this snp been prunned, default = False
                gen_seen (snp_hub_gen_t): generation seen, default = -1
                gen_prunned (snp_hub_gen_t): generation prunned, default = -1
            """

            # add to hub
            self.hub[snp] = [res,bin,idx,pos,enc,seen,prunned,gen_seen,gen_pruned]
            return

        # get snp result r^2
        def get_uni_res(self, snp: snp_t) -> snp_hub_res_t:
            # check for snp existence
            assert snp in self.hub
            # return data
            return np.float32(self.hub[snp][0])

        # get snp bin number
        def get_snp_bin(self, snp: snp_t) -> bin_hub_size_t:
            # assert that snp is in hub
            assert snp in self.hub
            # return data
            return bin_hub_size_t(self.hub[snp][1])

        # get snp position idx in the bin
        def get_snp_idx(self, snp: snp_t) -> snp_hub_idx_t:
            # assert that snp is in hub
            assert snp in self.hub
            # return data
            return snp_hub_idx_t(self.hub[snp][2])

        # get snp position in csv header
        def get_snp_pos(self, snp: snp_t) -> snp_hub_pos_t:
            # assert that snp is in hub
            assert snp in self.hub
            # return data
            return snp_hub_pos_t(self.hub[snp][3])

        # get snp encoder type
        def get_uni_encoding(self, snp: snp_t) -> snp_hub_enc_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][4]

        # has this snp been seen before
        def has_been_seen(self, snp: snp_t) -> bool:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][5]

        # has this snp been prunned
        def has_been_prunned(self, snp: snp_t) -> bool:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][6]

        # flip the prunned flag
        def flip_prunned(self, snp: snp_t, gen_pruned: snp_hub_gen_t) -> None:
            # check snp exists in the hub
            assert snp in self.hub
            # make sure we have not seen this snp before
            assert self.has_been_prunned(snp) == False

            # flip the flag
            self.hub[snp][6] = True
            # record the generation prunned
            self.hub[snp][8] = gen_pruned

            return

        # flip the seen flag
        def flip_seen(self, snp: snp_t) -> None:
            # check snp exists in the hub
            assert snp in self.hub
            # make sure we have not seen this snp before
            assert self.has_been_seen(snp) == False

            # flip the flag
            self.hub[snp][5] = True
            return

        # update snp hub with the r2 and encoding type
        # assuming that this only gets called once per snp
        def update_snp(self, snp: snp_t, value: np.float32, encoding:np.str_, gen_seen: snp_hub_gen_t) -> None:
            # assert that snp is in hub
            assert snp in self.hub
            # make sure we have not seen this snp before
            assert self.has_been_seen(snp) == False

            # flip the seen flag
            self.flip_seen(snp)
            # update the results
            self.hub[snp][0] = value
            # update the encoder type
            self.hub[snp][4] = encoding
            # update the generation seen
            self.hub[snp][7] = gen_seen

            # if value is negative, set as prunned
            if value < 0.0:
                self.flip_prunned(snp, gen_seen)

            return

        # get snp based on r2 weight from all snps in hub with positive r2 and count > 0
        def get_snp_r2_weighted(self) -> Tuple[npt.NDArray[snp_t], npt.NDArray[r2_t]]:
            # get all snps with r2 > 0.0 and count greater than 0
            snps = [snp for snp in self.hub.keys() if self.get_uni_res(snp) > np.float32(0.0) and self.has_been_seen(snp)]

            # get all r2 scores
            r2 = np.array([self.get_uni_res(snp) for snp in snps], dtype=np.float32)
            r2 = r2 / np.sum(r2, dtype=np.float32)
            assert len(snps) == len(r2)

            # get a random snp based on r2 scores as weights
            return snps, r2

    class Bin:
        """
        Hub to all snps and their apporpriate bin.
        """
        def __init__(self) -> None:
            self.bins = {} # {chrom: [np.array([pos1, pos2, ...], dtype=bin_hub_arr_t), ...]}
            return

        def get_chrom_number_of_bins(self, chrom: gen_chrom_num_t) -> bin_hub_size_t:
            # make sure the chromosome exists
            assert chrom in self.bins
            # return the number of bins
            return bin_hub_size_t(len(self.bins[chrom]))

        def get_bin_size(self, chrom: gen_chrom_num_t, bin: bin_hub_size_t) -> bin_hub_size_t:
            # make sure the chromosome exists
            assert chrom in self.bins

            # make sure the bin exists
            assert bin < len(self.bins[chrom])
            assert bin >= 0

            # return the size of the bin
            return bin_hub_size_t(len(self.bins[chrom][bin]))

        def get_pos_in_bin(self, chrom: gen_chrom_num_t, bin: bin_hub_size_t, idx: bin_hub_size_t) -> gen_chrom_pos_t:
            # make sure the chromosome exists
            assert chrom in self.bins
            # make sure the bin exists
            assert bin < len(self.bins[chrom])
            assert bin >= 0
            # make sure the index is within the bin
            assert idx < len(self.bins[chrom][bin])
            assert idx >= 0

            # return the position in the bin
            return gen_chrom_pos_t(self.bins[chrom][bin][idx])

        # get bin for a specific choromosome and bin number
        def get_bin(self, chrom: gen_chrom_num_t, bin: bin_hub_size_t) -> npt.NDArray[gen_chrom_pos_t]:
            # make sure the chromosome exists
            assert chrom in self.bins
            # make sure the bin exists
            assert bin < len(self.bins[chrom])
            assert bin >= 0

            # return the bin
            return self.bins[chrom][bin]

        # create bins for snps
        # O(|snps| * log(|snps|)) time complexity
        def generate_bins(self, snps: gen_header_snps_t, bin_size: bin_hub_size_t) -> List:
            # quick checks
            assert len(snps) > 0
            assert bin_size > 0

            # collect all chromosomes and sort snp postions
            sorted_snps = {}
            for snp in snps:
                # split header into chromosome and position
                chrom, pos = self.snp_chrm_pos(snp)

                # if we get a new chromosome, intialize a new bin for it
                if chrom not in sorted_snps:
                    sorted_snps[chrom] = [pos]
                else:
                    sorted_snps[chrom].append(pos)

            # sort the positions and assigen them to a bin
            self.bins = {}
            for chrom, pos in sorted_snps.items():
                sorted_pos = np.sort(np.array(pos, dtype=gen_chrom_pos_t), kind='mergesort')

                for p in sorted_pos:
                    # if we get a new chromosome, intialize a new bin for it
                    if chrom not in self.bins:
                        self.bins[chrom] = [[p]]
                    else:
                        # check if the current bin is full
                        if len(self.bins[chrom][-1]) == bin_size:
                            self.bins[chrom].append([p])
                        else:
                            self.bins[chrom][-1].append(p)

            # collect each snps bin number
            snp_bins = []
            # go through self.bins and collect all snps, bin_num
            for chrom, bins in self.bins.items():
                # i: bin number
                for i in range(len(bins)):
                    # idx_in_bin: index of the snp in the bin
                    for idx_in_bin, pos in enumerate(bins[i]):
                        snp_bins.append((np.str_(f"{chrom}.{pos}"), i, idx_in_bin))


            # cast all bins to numpy arrays for efficiency
            for chrom, bins in self.bins.items():
                self.bins[chrom] = [np.array(b, dtype=gen_chrom_pos_t) for b in bins]

            # print("Length of snps: ", len(snps), flush=True)
            # print("Result from count_bin_objs: ", self.count_bin_objs(), flush=True)
            # make sure all SNPs are accounted for
            assert len(snps) == self.count_bin_objs()
            # make sure snp_bins is the correct size
            assert len(snp_bins) == len(snps)

            return snp_bins # [(snp, bin_num, idx_in_bin), ...]

        # helper to generate chromosome number and snp position
        def snp_chrm_pos(self, snp: snp_t) -> Tuple[gen_chrom_num_t, gen_chrom_pos_t]:
            chrom, pos = snp.split('.')
            chrom, pos = gen_chrom_num_t(chrom), gen_chrom_pos_t(pos)
            return chrom, pos

        # count all objects in bins
        def count_bin_objs(self) -> np.uint16:
            sum = 0
            for _, bins in self.bins.items():
                for bin in bins:
                    sum += len(bin)

            return np.uint32(sum)

        # get all snps in a given bin with r2 > 0.0                   SNPS              weighted r2 scores > 0
        def get_snps_r2_in_bin(self, snp: snp_t, snp_hub) -> Tuple[npt.NDArray[snp_t], npt.NDArray[r2_t]]:
            # make sure that snp_hub is the correct type
            assert isinstance(snp_hub, SnpHub.Hub)

            # get chromosome and position from snp
            chrom, pos = self.snp_chrm_pos(snp)
            bin = snp_hub.get_snp_bin(snp)

            # go thorugh all snps in the bin and collect the ones with r2 > 0.0
            snps = []
            r2 = []
            bin_snps = np.array([f"{chrom}.{p}" for p in self.bins[chrom][bin] if p != pos], dtype=snp_t)

            for s in bin_snps:
                # make sure this snp is in the hub
                assert s in snp_hub.hub

                # check if r2 is greater than 0.0 and count is greater than 0
                if snp_hub.get_uni_res(s) > r2_t(0.0) and snp_hub.has_been_seen(s):
                    snps.append(s)
                    r2.append(snp_hub.get_uni_res(s))

            # make sure snps and r2 are the same size
            assert len(snps) == len(r2)

            # get the snps in the bin
            return np.array(snps, dtype=np.str_), np.array(r2, dtype=r2_t) / np.sum(r2, dtype=r2_t)

        # return a random snp from the same chromosome and bin
        def get_ran_snp_in_bin(self, snp: snp_t, rng_: rng_t, snp_hub) -> snp_t:
            # make sure there is a '.' inside the snp string
            assert '.' in snp
            # make sure snp_hub is the correct type
            assert isinstance(snp_hub, SnpHub.Hub)

            # set rng
            rng = np.random.default_rng(rng_)

            # get chromosome and position from snp
            chrom, pos = self.snp_chrm_pos(snp)
            bin = snp_hub.get_snp_bin(snp)

            # collect all snps in the bin except the input snp
            candidates = [p for p in self.bins[chrom][bin] if p != pos]

            # if we no candidates, return original snp
            if len(candidates) == 0:
                return snp

            # return a random snp
            return snp_t(f"{chrom}.{rng.choice(candidates)}")

        # get all snps in the same chromosome but different bin
        def get_snps_r2_in_chrom(self, snp: snp_t, snp_hub) -> Tuple[npt.NDArray[snp_t], npt.NDArray[r2_t]]:
            # make sure there is a '.' inside the snp string
            assert '.' in snp
            # make sure snp_hub is the correct type
            assert isinstance(snp_hub, SnpHub.Hub)

            # get chromosome and position from snp
            chrom, _ = self.snp_chrm_pos(snp)
            bin = snp_hub.get_snp_bin(snp)

            # return all snps and r2 > 0.0 in the chromosome
            snps = []
            r2 = []

            # go through all bins in the chromosome
            for i, b in enumerate(self.bins[chrom]):
                if i == bin:
                    continue

                # go through each chromose position in the bin
                for pos in b:
                    s = snp_t(f"{chrom}.{pos}")

                    # make sure this snp is in the hub
                    assert s in snp_hub.hub

                    # check if r2 is greater than 0.0 and count is greater than 0
                    if snp_hub.get_uni_res(s) > r2_t(0.0) and snp_hub.has_been_seen(s):
                        snps.append(s)
                        r2.append(snp_hub.get_uni_res(s))

            # make sure snps and r2 are the same size
            assert len(snps) == len(r2)

            # return all snps in the chromosome
            return np.array(snps, dtype=snp_t) , np.array(r2, dtype=r2_t)/ np.sum(r2, dtype=r2_t)

        # get random snp from the same chromosome but different bin
        def get_ran_snp_in_chrom(self, snp: snp_t, rng_: rng_t, snp_hub) -> snp_t:
            # make sure there is a '.' inside the snp string
            assert '.' in snp
            # make sure snp_hub is the correct type
            assert isinstance(snp_hub, SnpHub.Hub)

            # initialize rng
            rng = np.random.default_rng(rng_)

            # get chromosome and position from snp
            chrom, _ = self.snp_chrm_pos(snp)
            bin = snp_hub.get_snp_bin(snp)

            # if there is only one bin for this chromosome return the original snp
            if len(self.bins[chrom]) == 1:
                return snp

            # get integer from 0 to len(self.bins[chrom]
            other_bin = rng.choice([i for i in range(len(self.bins[chrom])) if i != bin])
            # sample a random snp from the bin
            sample = rng.integers(0, len(self.bins[chrom][other_bin]), dtype=np.uint16)
            # get random snp from the bin
            pos = self.bins[chrom][other_bin][sample]

            # return a random snp
            return snp_t(f"{chrom}.{pos}")

        # get all snps outside the chromosome with r2 > 0.0
        def get_snps_r2_out_chrom(self, snp: snp_t, snp_hub) -> Tuple[npt.NDArray[snp_t], npt.NDArray[r2_t]]:
            # make sure there is a '.' inside the snp string
            assert '.' in snp
            # make sure snp_hub is the correct type
            assert isinstance(snp_hub, SnpHub.Hub)

            # get chromosome and position from snp
            chrom, _ = self.snp_chrm_pos(snp)

            # get all keys in the hub
            chrom_keys = list(self.bins.keys())

            # remove chrom from chorom_keys
            chrom_keys.remove(chrom)

            # go through all remaining keys and get all snps with r2 > 0.0
            snps = []
            r2 = []
            for c in chrom_keys:
                # go through all bins in the chromosome
                for bin in self.bins[c]:
                    # go through each chromosome position in the bin
                    for pos in bin:
                        s = snp_t(f"{c}.{pos}")

                        # make sure this snp is in the hub
                        assert s in snp_hub.hub

                        # check if r2 is greater than 0.0 and count is greater than 0
                        if snp_hub.get_uni_res(s) > r2_t(0.0) and snp_hub.has_been_seen(s):
                            snps.append(s)
                            r2.append(snp_hub.get_uni_res(s))

            # make sure snps and r2 are the same size
            assert len(snps) == len(r2)

            # return all snps outside the chromosome
            return np.array(snps, dtype=snp_t), np.array(r2, dtype=r2_t) / np.sum(r2, dtype=r2_t)

        # get random snp outside the chromosome
        def get_ran_snp_out_chrom(self, snp: snp_t, rng_: rng_t, snp_hub) -> snp_t:
            # make sure there is a '.' inside the snp string
            assert '.' in snp
            # make sure snp_hub is the correct type
            assert isinstance(snp_hub, SnpHub.Hub)

            # initialize rng
            rng = np.random.default_rng(rng_)

            # get chromosome and position from snp
            chrom, _ = self.snp_chrm_pos(snp)

            # get all keys in the hub
            chrom_keys = list(self.bins.keys())

            # remove chrom from chorom_keys
            chrom_keys.remove(chrom)

            # get random chromosome
            c = chrom_keys[rng.integers(0, len(chrom_keys), dtype=np.uint16)]

            # get random bin index from the chromosome
            i = rng.integers(0, len(self.bins[c]), dtype=np.uint16)

            # get random snp from the bin
            pos = self.bins[c][i][rng.integers(0, len(self.bins[c][i]), dtype=np.uint16)]

            # return a random snp
            return snp_t(f"{c}.{pos}")

        # get a random snp from the hub
        # def get_ran_snp(self, rng_: rng_t) -> snp_t:
        #     # initialize rng
        #     rng = np.random.default_rng(rng_)

        #     # get a random chromosome key
        #     c = rng.choice(list(self.bins.keys()))

        #     # get a random bin index from the chromosome
        #     i = rng.integers(0, len(self.bins[c]), dtype=np.uint16)

        #     return snp_t(f"{c}.{rng.choice(self.bins[c][i])}")

    # initialize all hubs
    def __init__(self, snps: gen_header_snps_t, bin_size: bin_hub_size_t) -> None:

        # initialize non pruned hub
        print('Initializing Non Pruned Hub')
        self.non_pruned = self.Non_Pruned_Hub(snps)
        print('Non Pruned Hub Initialized\n')

        # bin hub stuff
        print('Initializing SnpHub')
        self.bins = self.Bin()
        # get snps and their bin id
        snp_bin = self.bins.generate_bins(snps, bin_size)
        print('Bin Hub Initialized')

        # snp hub stuff
        self.hub = self.Hub()
        # update snp_hub with snp_bin and snp header positions
        for s in snp_bin:
            # find where snp is located in csv header (snps)
            h_pos = snp_hub_pos_t(np.where(snps == s[0])[0][0])

            # make sure that the snp is in the correct format
            assert isinstance(s[0], np.str_)
            # make sure header position matches the csv header
            assert s[0] == snps[h_pos]
            # make sure the snp bin poisition is correct
            assert self.bins.snp_chrm_pos(s[0])[1] == self.bins.bins[self.bins.snp_chrm_pos(s[0])[0]][s[1]][s[2]]

            # add snp to hub with all its data
            self.hub.add_to_hub(snp=s[0],
                                res=snp_hub_res_t(-1.0),
                                bin=snp_hub_bin_t(s[1]),
                                idx=s[2],
                                pos=h_pos,
                                enc=snp_hub_enc_t(''),
                                seen=False,
                                prunned=False,
                                gen_seen=snp_hub_gen_t(-1),
                                gen_pruned=snp_hub_gen_t(-1))
        print('SNP Hub Initialized')
        return

    # get best type of encoder for a given snp
    def get_uni_encoding(self, snp: snp_t) -> snp_hub_enc_t:
        return self.hub.get_uni_encoding(snp)

    # get r2 for a given snp from snp hub
    def get_uni_res(self, snp: snp_t) -> snp_hub_res_t:
        return self.hub.get_uni_res(snp)

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
        """

        # Save snp hub with headers
        snp_data = []
        for k, v in self.hub.hub.items():
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
            snp_data.append([k, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]])

        # Sort snp_data by the second column (AVG_R2)
        snp_data.sort(key=lambda x: x[1], reverse=True)  # reverse=True for descending order

        # Write snp hub to file
        with open(save_dir+"snp_hub.csv", 'w') as f:
            # Write the headers for the snp_file
            f.write("snp,chr,bp,r2,bin_num,bin_idx,encoding,seen,pruned,gen_seen,gen_pruned\n")
            for row in snp_data:
                # split snp into chromosome and position
                chrom, pos = row[0].split('.')
                f.write(f"{row[0]},{chrom},{pos},{row[1]},{row[2]},{row[4]},{row[5]},{row[6]},{row[7]},{row[8]},{row[9]}\n")

        # save csv with both seen and not prunned snps
        # Write snp hub to file
        with open(save_dir+"non_pruned_and_seen.csv", 'w') as f:
            # Write the headers for the snp_file
            f.write("snp,r2,encoding\n")
            for row in snp_data:
                if row[6] == True and row[7] == False:
                    f.write(f"{row[0]},{row[1]},{row[5]}\n")
        return

    # update snp hub with best univariate r2 result and corresponding encoder type
    def update_snp_hub(self, snp:snp_t, result:snp_hub_res_t, type: snp_hub_enc_t, gen_seen: snp_hub_gen_t) -> None:
        # update Hub object: if r2 is negative, flip prunned flag
        self.hub.update_snp(snp, result, type, gen_seen)
        # update Non_Pruned_Hub object: if r2 is negative, remove snp from non prunned
        if result < r2_t(0.0):
            self.non_pruned.remove_snp(snp)
        return

    # check if snp has encoder type recorded in the snp hub
    def is_encoder_in_hub(self, snp:snp_t) -> bool:
        return self.hub.has_been_seen(snp)

    # get snp position from the snp hub
    def get_snp_pos(self, snp: snp_t) -> snp_hub_pos_t:
        return self.hub.get_snp_pos(snp)

    # get a snp from the same chromosome and bin with r2 > 0.0 based on r2 weight
    def get_smt_snp_in_bin(self, snp: snp_t, rng_: rng_t) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # initialize rng
        rng = np.random.default_rng(rng_)

        # break snp into chromosome and position
        chrom, _ = self.bins.snp_chrm_pos(snp)

        # get the bin number for the snp
        bin_id = self.hub.get_snp_bin(snp)

        # get actual bin from Bin class
        bin = self.bins.get_bin(chrom, bin_id)

        # collect all snps that have not been prunned and have r2 > 0.0
        snps = []
        r2 = []

        for p in bin:
            s = snp_t(f"{chrom}.{p}")

            if self.hub.get_uni_res(s) > r2_t(0.0) and self.hub.has_been_prunned(s) == False and s != snp:
                snps.append(s)
                r2.append(self.hub.get_uni_res(s))

        # if no snps were returned, return a random one
        if len(snps) == 0:
            return self.get_ran_snp_in_bin(snp, rng)

        # get a random snp based on r2 scores as weights
        r2 = r2 / np.sum(r2, dtype=np.float32)
        choice = rng.choice(snps, p=r2)

        # try to get a random snp that is not the same as the input snp
        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = rng.choice(snps, p=r2)

        # get a random snp based on r2 scores as weights
        return snp

    # get a random snp from the same chromosome and bin
    def get_ran_snp_in_bin(self, snp: snp_t, rng_: rng_t) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # initialize rng
        rng = np.random.default_rng(rng_)

        # break snp into chromosome and position
        chrom, _ = self.bins.snp_chrm_pos(snp)

        # get the bin number for the snp
        bin_id = self.hub.get_snp_bin(snp)

        # get actual bin from Bin class
        bin = self.bins.get_bin(chrom, bin_id)

        # collect all snps that have (not prunned and seen) or (r2 > 0.0 and seen)
        snps = []
        for p in bin:
            s = snp_t(f"{chrom}.{p}")

            # not seen
            not_seen = self.hub.has_been_seen(s) == False
            # r2 > 0.0 and seen and not prunned
            seen_r2_np = self.hub.get_uni_res(s) > r2_t(0.0) and self.hub.has_been_seen(s) and self.hub.has_been_prunned(s) == False

            if (not_seen or seen_r2_np) and s != snp:
                snps.append(s)

        # if no snps were collected, return a random snp from non prunned
        if len(snps) == 0:
            return self.get_random_non_pruned_snp(rng)

        # try to get a random snp position that is not the same as the input snp
        choice = rng.choice(snps)

        for _ in range(mutation_tries):
            if choice != snp:
                return s
            choice = rng.choice(snps)

        # get a random snp based
        return snp

    # geta snp from the same chromosome but different bin
    def get_smt_snp_in_chrm(self, snp: snp_t, rng_: rng_t) -> snp_t:
        # initialize rng
        rng = np.random.default_rng(rng_)

        # break snp into chromosome and position
        chrom, _ = self.bins.snp_chrm_pos(snp)

        # get bin id for the snp
        bin_id = self.hub.get_snp_bin(snp)

        # collect all snps that have not been prunned and have r2 > 0.0
        snps = []
        r2 = []

        # loop through all non prunned snps and collect the ones with r2 > 0.0 and not prunned
        for pos in self.non_pruned.non_pruned[chrom]:
            s = snp_t(f"{chrom}.{pos}")

            if self.hub.get_uni_res(s) > r2_t(0.0) and self.hub.has_been_seen(s) and self.hub.get_snp_bin(s) != bin_id:
                snps.append(s)
                r2.append(self.hub.get_uni_res(s))

        r2 = r2 / np.sum(r2, dtype=np.float32)

        # if no snps were returned, return a random one
        if len(snps) == 0:
            return self.get_ran_snp_in_chrm(snp, rng)

        # get a random snp based on r2 scores as weights
        choice = rng.choice(snps, p=r2)

        # try to get a random snp that is not the same as the input snp
        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = rng.choice(snps, p=r2)

        # get a random snp based on r2 scores as weights
        return snp

    # get a random snp from the same chromosome but different bin
    def get_ran_snp_in_chrm(self, snp: snp_t, rng_: rng_t) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # initialize rng
        rng = np.random.default_rng(rng_)

        # break snp into chromosome and position
        chrom, _ = self.bins.snp_chrm_pos(snp)

        # get bin id for the snp
        bin_id = self.hub.get_snp_bin(snp)

        # collect all snps that have not been prunned and have r2 > 0.0
        snps = []

        # loop through all non prunned snps and collect the ones with r2 > 0.0 and not prunned
        for pos in self.non_pruned.non_pruned[chrom]:
            # make snps
            s = snp_t(f"{chrom}.{pos}")
            # not seen
            not_seen = self.hub.has_been_seen(s) == False
            # r2 > 0.0 and seen and not prunned
            seen_r2_np = self.hub.get_uni_res(s) > r2_t(0.0) and self.hub.has_been_seen(s) and self.hub.has_been_prunned(s) == False

            if (not_seen or seen_r2_np) and self.hub.get_snp_bin(s) != bin_id:
                snps.append(s)

        # if no snps were collected, return a random snp from non prunned
        if len(snps) == 0:
            return self.get_random_non_pruned_snp(rng)

        # try to get a random snp that is not the same as the input snp
        choice = rng.choice(snps)

        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = rng.choice(snps)

        # return same snp
        return snp

    # get a snp from outside the chromosome with r2 > 0.0 based on r2 weight
    def get_smt_snp_out_chrm(self, snp: snp_t, rng_: rng_t) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # set the random number generator
        rng = np.random.default_rng(rng_)

        # split up the snp into chromosome and position
        chrom, pos = self.bins.snp_chrm_pos(snp)

        # get keys for non pruned snps
        chrom_keys = self.non_pruned.get_keys_with_snps()

        # remove the current chromosome from the list
        if chrom in chrom_keys:
            chrom_keys.remove(chrom)

        # randomly select a chromosome
        c_pic = rng.choice(chrom_keys)

        # collect all snps that have not been prunned and have r2 > 0.0
        snps = []
        r2 = []

        # loop through all non prunned snps and collect the ones with r2 > 0.0 and not prunned
        for pos in self.non_pruned.non_pruned[c_pic]:
            s = snp_t(f"{c_pic}.{pos}")

            if self.hub.get_uni_res(s) > r2_t(0.0) and self.hub.has_been_seen(s):
                snps.append(s)
                r2.append(self.hub.get_uni_res(s))

        r2 = r2 / np.sum(r2, dtype=np.float32)

        # if no snps were returned, return a random one
        if len(snps) == 0:
            return self.get_ran_snp_out_chrm(snp, rng)

        # get a random snp based on r2 scores as weights
        choice = rng.choice(snps, p=r2)

        # try to get a random snp that is not the same as the input snp
        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = rng.choice(snps, p=r2)

        # get a random snp based on r2 scores as weights
        return snp

    # get random snp from outside the chromosome
    def get_ran_snp_out_chrm(self, snp: snp_t, rng_: rng_t) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # initialize rng
        rng = np.random.default_rng(rng_)

        # split up the snp into chromosome and position
        chrom, _ = self.bins.snp_chrm_pos(snp)

        # get keys for non pruned snps
        chrom_keys = self.non_pruned.get_keys_with_snps()

        # remove the current chromosome from the list
        if chrom in chrom_keys:
            chrom_keys.remove(chrom)

        # randomly select a chromosome
        c_pic = rng.choice(chrom_keys)

        # collect all snps that have not been prunned and have r2 > 0.0
        snps = []

        # loop through all non prunned snps and collect them
        for pos in self.non_pruned.non_pruned[c_pic]:
            # make snps
            s = snp_t(f"{c_pic}.{pos}")
            # not seen
            not_seen = self.hub.has_been_seen(s) == False
            # r2 > 0.0 and seen and not prunned
            seen_r2_np = self.hub.get_uni_res(s) > r2_t(0.0) and self.hub.has_been_seen(s) and self.hub.has_been_prunned(s) == False

            if (not_seen or seen_r2_np):
                snps.append(s)

        # if no snps were collected, return a random snp from non prunned
        if len(snps) == 0:
            return self.get_random_non_pruned_snp(rng)

        # try to get a random snp that is not the same as the input snp
        choice = rng.choice(snps)

        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = rng.choice(snps)

        return snp

    # get a random snp from all possible snps
    def get_ran_snp(self, rng_: rng_t, snp = None) -> snp_t:
        # initialize rng
        rng = np.random.default_rng(rng_)

        # if snp is None, return a random snp
        # None means we don't need to check if the snp is the same as the input snp
        if snp is None:
            return self.non_pruned.get_ran_snp(rng)
            # return self.bins.get_ran_snp(rng)

        # if snp is provided, return a random one that is not the same as the input snp
        choice = self.non_pruned.get_ran_snp(rng)
        # choice = self.bins.get_ran_snp(rng)

        # try to get a random snp that is not the same as the input snp
        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = self.non_pruned.get_ran_snp(rng)
            # choice = self.bins.get_ran_snp(rng)

        return choice

    def does_snp_exist(self, snp: snp_t) -> bool:
        return snp in self.hub.hub

    # has snp been prunned?
    def has_been_prunned(self, snp: snp_t) -> bool:
        return self.hub.has_been_prunned(snp)

    # process the prunned snps
    def process_prunned_snps(self, snps: Set[snp_t], gen_pruned: snp_hub_gen_t) -> None:
        # go through each snp and update the hub
        for snp in snps:
            # check to make sure we have not prunned this snp before
            assert self.hub.has_been_prunned(snp) == False

            # flip snp to prunned
            self.hub.flip_prunned(snp, gen_pruned)

            # delete snp from non pruned
            self.non_pruned.remove_snp(snp)

    # function to take in a list of snps and generate a dictionary of snps and their corresponding r2 values
    def generate_r2_dict(self, snps: Set[snp_t]) -> List:
        # make sure
        assert len(snps) > 0

        return [(snp, self.get_uni_res(snp)) for snp in snps]

    # check if all snps have been prunned
    def all_snps_prunned(self, snps: Set[snp_t]) -> bool:

        # if any snps has not been prunned return False
        for snp in snps:
            if self.hub.has_been_prunned(snp) == False:
                return False

        # return true if all snps have been prunned
        return True

    # print size of non pruned hub
    def pruned_hub_size(self) -> np.uint32:
        return self.non_pruned.get_total()

    # return a random snp from the non pruned hub
    def get_random_non_pruned_snp(self, rng_: rng_t) -> snp_t:
        rng = np.random.default_rng(rng_)
        return self.non_pruned.get_ran_snp(rng)
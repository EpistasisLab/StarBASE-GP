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
            bin_pos = 1 # position for count variable in hub value list
            idx_pos = 2 # position for bin number in hub value list
            pos_pos = 3 # position for header position in hub value list
            enc_pos = 4 # position for the corresponding encoder types in hub value list
            seen_pos = 5 # position for the seen flag in hub value list
            """

            # {snp: [res(np.float32),bin(np.uint32),idx(np.uint32),pos(np.uint32),enc(np.str_),seen(bool)],...}
            self.hub = {}

        # will add snp, sum, cnt, bin, pos， idx, res, typ to the hub #YF
        def add_to_hub(self,
                       snp: snp_t,
                       res: snp_hub_res_t,
                       bin: snp_hub_bin_t,
                       idx: snp_hub_idx_t,
                       pos: snp_hub_pos_t,
                       enc: snp_hub_enc_t,
                       seen=False) -> None:
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
            """

            # add to hub
            self.hub[snp] = [res,bin,idx,pos,enc,seen]
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
        def update_snp(self, snp: snp_t, value: np.float32, encoding:np.str_) -> None:
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
        def get_ran_snp(self, rng_: rng_t) -> snp_t:
            # initialize rng
            rng = np.random.default_rng(rng_)

            # get a random chromosome key
            c = rng.choice(list(self.bins.keys()))

            # get a random bin index from the chromosome
            i = rng.integers(0, len(self.bins[c]), dtype=np.uint16)

            return snp_t(f"{c}.{rng.choice(self.bins[c][i])}")

    # initialize all hubs
    def __init__(self, snps: gen_header_snps_t, bin_size: bin_hub_size_t) -> None:

        # bin_size: base pairs per bin

        # bin hub stuff
        print('Initializing SnpHub')
        self.bins = self.Bin()
        # get snps and their bin id
        snp_bin = self.bins.generate_bins(snps, bin_size)
        print('Bin Hub Initialized')

        #YF calculate the total number of possible combo
        chrom_size = len(self.bins.bins.keys())
        self.total_combo = np.uint32(chrom_size) * np.uint32(bin_size) # uint32 instead of uint16 to prevent overflow

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
                                seen=False)
        print('SNP Hub Initialized')
        return

    # get best type of encoder for a given snp
    def get_uni_encoding(self, snp: snp_t) -> snp_hub_enc_t:
        return self.hub.get_uni_encoding(snp)

    # get r2 for a given snp from snp hub
    def get_uni_res(self, snp: snp_t) -> snp_hub_res_t:
        return self.hub.get_uni_res(snp)

    # save the epi_hub and snp_hub to a file
    def save_hubs(self, snp_file: str) -> None:
        """
        Positional arguments for each row in the snp hub

        res_pos = 0 # position for r2 recived from evaluation
        bin_pos = 1 # position for count variable in hub value list
        idx_pos = 2 # position for bin number in hub value list
        pos_pos = 3 # position for header position in hub value list
        enc_pos = 4 # position for the corresponding encoder types in hub value list
        seen_pos = 5 # position for the seen flag in hub value list
        """

        # Save snp hub with headers
        snp_data = []
        for k, v in self.hub.hub.items():
            snp_data.append([k, v[0], v[1], v[2], v[3], v[4], v[5]])

        # Sort snp_data by the second column (AVG_R2)
        snp_data.sort(key=lambda x: x[1], reverse=True)  # reverse=True for descending order

        # Write snp hub to file
        with open(snp_file, 'w') as f:
            # Write the headers for the snp_file
            f.write("snp,res_r2,bin_num,bin_idx,enc_pos,seen_pos\n")
            for row in snp_data:
                f.write(f"{row[0]},{row[1]},{row[2]},{row[4]},{row[5]},{row[6]}\n")

        return

    # update snp hub with best univariate r2 result and corresponding encoder type
    def update_snp_hub(self, snp:snp_t, result:snp_hub_res_t, type: snp_hub_enc_t) -> None:
        self.hub.update_snp(snp, result, type)
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

        # get all snps and r2 scores for a given snp within the same chorosome and bin
        snps, r2 = self.bins.get_snps_r2_in_bin(snp, self.hub)
        assert(len(snps) == len(r2))

        # if no snps were returned, return a random one
        if len(snps) == 0:
            return self.get_ran_snp_in_bin(snp, rng)

        # get a random snp based on r2 scores as weights
        choice = rng.choice(snps, p=r2)

        # try to get a random snp that is not the same as the input snp
        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = rng.choice(snps, p=r2)

        # get a random snp based on r2 scores as weights
        return choice

    # get a random snp from the same chromosome and bin
    def get_ran_snp_in_bin(self, snp: snp_t, rng_: rng_t) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # initialize rng
        rng = np.random.default_rng(rng_)

        # try to get a random snp that is not the same as the input snp
        choice = self.bins.get_ran_snp_in_bin(snp, rng, self.hub)

        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = self.bins.get_ran_snp_in_bin(snp, rng, self.hub)

        # get a random snp based
        return choice

    # geta snp from the same chromosome but different bin
    def get_smt_snp_in_chrm(self, snp: snp_t, rng_: rng_t) -> snp_t:
        # initialize rng
        rng = np.random.default_rng(rng_)

        # get all snps and r2 scores for a given snp within the same chorosome and bin
        snps, r2 = self.bins.get_snps_r2_in_chrom(snp, self.hub)
        assert(len(snps) == len(r2))

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
        return choice

    # get a random snp from the same chromosome but different bin
    def get_ran_snp_in_chrm(self, snp: snp_t, rng_: rng_t) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # initialize rng
        rng = np.random.default_rng(rng_)

        # try to get a random snp that is not the same as the input snp
        choice = self.bins.get_ran_snp_in_chrom(snp, rng, self.hub)

        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = self.bins.get_ran_snp_in_chrom(snp, rng, self.hub)

        # get a random snp based
        return choice

    # get a snp from outside the chromosome with r2 > 0.0 based on r2 weight
    def get_smt_snp_out_chrm(self, snp: snp_t, rng_: rng_t) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # set the random number generator
        rng = np.random.default_rng(rng_)

        # get all snps and r2 scores for a given snp within the same chorosome and bin
        snps, r2 = self.bins.get_snps_r2_out_chrom(snp, self.hub)
        assert(len(snps) == len(r2))

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
        return choice

    # get random snp from outside the chromosome
    def get_ran_snp_out_chrm(self, snp: snp_t, rng_: rng_t) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # initialize rng
        rng = np.random.default_rng(rng_)

        # try to get a random snp that is not the same as the input snp
        choice = self.bins.get_ran_snp_out_chrom(snp, rng, self.hub)

        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = self.bins.get_ran_snp_out_chrom(snp, rng, self.hub)
        return choice

    # get a random snp from all possible snps
    def get_ran_snp(self, rng_: rng_t, snp = None) -> snp_t:
        # initialize rng
        rng = np.random.default_rng(rng_)

        # if snp is None, return a random snp
        # None means we don't need to check if the snp is the same as the input snp
        if snp is None:
            return self.bins.get_ran_snp(rng)

        # if snp is provided, return a random one that is not the same as the input snp
        choice = self.bins.get_ran_snp(rng)

        # try to get a random snp that is not the same as the input snp
        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = self.bins.get_ran_snp(rng)

        return choice

    # get snp from hub based on r2 weight
    # this is only called for to get the first snp -- no need to check if snp is the same
    def get_smt_snp(self, rng:rng_t):
        # call snp hub to get snps and r2 scores based on r2 weight > 0 and count > 0
        snps, r2 = self.hub.get_snp_r2_weighted()

        # if no snps were returned, return a random one
        if len(snps) == 0:
            return self.get_ran_snp(rng, None)
        # else return a random snp based on r2 scores as weights
        return rng.choice(snps, p=r2)

    def does_snp_exist(self, snp: snp_t) -> bool:
        return snp in self.hub.hub

    # get a snp from the same chromosome and bin within the wiggle range with r2 > 0.0 based on r2 weight
    def get_smt_snp_in_bin_wiggle(self, snp: snp_t, rng_: rng_t, step: np.uint16) -> snp_t:
        rng = np.random.default_rng(rng_)

        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # get a random snp based
        wiggle_snps = self.get_wiggle_range(snp, step)

        # collect all snps with a positive r2 score only
        snps, r2 = [], []
        for s in wiggle_snps:
            # make sure this snp is in the hub
            assert s in self.hub.hub

            # check if r2 is greater than 0.0
            if self.hub.get_uni_res(s) > np.float32(0.0):
                snps.append(s)
                r2.append(self.hub.get_uni_res(s))

        # if no snps have positve r2, return a random one from the wiggle range
        if len(snps) == 0:
            return self.get_ran_snp_in_bin_wiggle(snp, rng, step)

        # normalize r2 scores
        r2 = np.array(r2, dtype=np.float32) / np.sum(r2, dtype=np.float32)
        # get a snp based on r2 scores as weights
        choice = rng.choice(snps, p=r2)

        # try to get a random snp that is not the same as the input snp
        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = rng.choice(snps, p=r2)

        # get a random snp based on r2 scores as weights
        return self.get_ran_snp_in_bin_wiggle(snp, rng, step)

    # get a random snp from the same chromosome and bin that also considers neighboring bins within wiggle range
    def get_ran_snp_in_bin_wiggle(self, snp: snp_t, rng: rng_t, step: np.uint16) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # get a random snp based
        wiggle_snps = self.get_wiggle_range(snp, step)
        choice = rng.choice(wiggle_snps)

        # try to get a random snp that is not the same as the input snp
        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = rng.choice(wiggle_snps)

        # get a random snp based on r2 scores as weights
        return choice

    # get all snps from a given range that cross multiple bins if necessary
    def get_wiggle_range(self, snp: snp_t, step: np.uint16) -> npt.NDArray[snp_t]:
        # get chromosome and position from snp
        snp_chrom, _ = self.bins.snp_chrm_pos(snp)

        # wiggle snp holders
        right_wiggle_range = []
        left_wiggle_range = []

        # get the right wiggle range
        right_start = self.hub.get_snp_idx(snp) + 1
        original_bin_size = self.bins.get_bin_size(snp_chrom, self.hub.get_snp_bin(snp))

        # get all snps from the current bin within the right wiggle step range
        while len(right_wiggle_range) <= step and right_start < original_bin_size:
            right_wiggle_range.append(self.bins.get_pos_in_bin(snp_chrom, self.hub.get_snp_bin(snp), right_start))
            right_start += 1

        # get total number of bins in the chromosome
        chrm_num_of_bins = self.bins.get_chrom_number_of_bins(snp_chrom)
        # cross into the right neighboring bins if necessary to get the rest ÷of the snps
        right_snp_bin_num = self.hub.get_snp_bin(snp) + 1

        while len(right_wiggle_range) <= step and right_snp_bin_num < chrm_num_of_bins:
            # collect snps from the next bin until the wiggle range is full
            bin_start = 0
            while len(right_wiggle_range) <= step and bin_start < self.bins.get_bin_size(snp_chrom, right_snp_bin_num):
                right_wiggle_range.append(self.bins.get_pos_in_bin(snp_chrom, right_snp_bin_num, bin_start))
                bin_start += 1

            # move to the next bin to keep collecting snps
            right_snp_bin_num += 1

        # get all snps from the current bin within the left wiggle step range
        left_start = self.hub.get_snp_idx(snp) - 1
        while len(left_wiggle_range) <= step and 0 <= left_start:
            left_wiggle_range.append(self.bins.get_pos_in_bin(snp_chrom, self.hub.get_snp_bin(snp), left_start))
            left_start -= 1

        # cross into the left neighboring bins if necessary to get the rest of the snps
        left_snp_bin_num = self.hub.get_snp_bin(snp) - 1
        while len(left_wiggle_range) <= step and 0 <= left_snp_bin_num:
            # collect snps from the previous bin until the wiggle range is full
            bin_start = self.bins.get_bin_size(snp_chrom, left_snp_bin_num) - 1
            while len(left_wiggle_range) <= step and 0 <= bin_start:
                left_wiggle_range.append(self.bins.get_pos_in_bin(snp_chrom, left_snp_bin_num, bin_start))
                bin_start -= 1

            # move to previous bin to keep collecting snps
            left_snp_bin_num -= 1

        # combine the wiggle range and return new snps
        wiggle_range = left_wiggle_range + right_wiggle_range
        return np.array([f"{snp_chrom}.{pos}" for pos in wiggle_range], dtype=snp_t)


    # function to take in a list of snps and generate a dictionary of snps and their corresponding r2 values
    def generate_r2_dict(self, snps: Set[snp_t]) -> List:
        # make sure
        assert len(snps) > 0

        return [(snp, self.get_uni_res(snp)) for snp in snps]

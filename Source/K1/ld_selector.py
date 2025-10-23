##########################################################################################
############################ the ld classes ##############################################
##########################################################################################

from ..Base.selectors import SelectorNode
from ..Base.types import (rng_t, prob_t, int32_t, uint16_t, float32_t)

from decimal import Decimal
from typing import Dict
import numpy as np
from typeguard import typechecked
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

@typechecked
class LDSelector(SelectorNode):
    def __init__(self, rng: rng_t):

            # threshold values in steps of 0.05 between 0.5 and 0.95
            values = [float(Decimal('0.5') + Decimal('0.05') * i) for i in range(10)] # using Decimal for precision
            # genomic distances in steps of 100,000 between 500,000 and 1,000,000
            distance_choices = list(range(500000, 1000001, 100000))
            self.params = {
                'threshold': rng.choice(values),
                'genomic_distance': rng.choice(distance_choices)
            }

            # todo: only use params dictionary if possible?
            self.threshold = self.params['threshold']
            self.genomic_distance = self.params['genomic_distance']
            self.selected_features_ = None
            self.bool_mask = None
            self.name_of_selected_features = None

            # dictionary to store the details of SNPs after LD pruning - pruned flag, reason for pruning, threshold used, genomic distance used, and anchor SNP
            self.snp_details_after_ld = {}


    def fit(self, X_original, X_encoded, y, snp_r2_dict):
        if X_original.empty:
            self.selected_features_ = None
            return self

        # Function to calculate LD (R²) between two SNPs
        def calculate_ld(X, snp1: np.str_, snp2: np.str_):
            correlation = np.corrcoef(X[snp1], X[snp2])[0, 1]
            r_squared = correlation ** 2
            return r_squared

        # Function to remove subset groups from a list of groups
        def remove_subsets(groups):
            unique_groups = []
            for group in groups:
                is_subset = False
                for other_group in groups:
                    if set(group).issubset(set(other_group)) and group != other_group:
                        is_subset = True
                        break
                if not is_subset:
                    unique_groups.append(group)
            return unique_groups

        ld_threshold = self.threshold
        max_distance = self.genomic_distance
        final_selected_snps = []
        ld_removed_snps = set()
        anchor_snp_details = {}
        snp_details_after_ld = {}

        column_names = X_original.columns
        chr, pos = [], []

        # initialize the snp_details after_ld dictionary to have every SNP and set False, "", self.threshold, self.genomic_distance, and "" for anchor_snp
        for snp in column_names:
            snp_details_after_ld[snp] = {"pruned": False, "reason": "", "threshold": self.threshold, "genomic_distance": self.genomic_distance, "anchor_snp": ""}

        for snp in column_names:
            chr.append(int(snp.split('.')[0]))
            pos.append(int(snp.split('.')[1]))

        genotype_df_columns = pd.DataFrame({'chrom': chr, 'pos': pos, 'snp': column_names})
        genotype_df_columns = genotype_df_columns.sort_values(['chrom', 'pos'])
        sorted_snps = genotype_df_columns['snp'].tolist()
        genotype_df_original = X_original[sorted_snps]
        genotype_df_encoded = X_encoded[sorted_snps]
        chromosomes = genotype_df_columns['chrom'].unique()

        marginal_r2 = {snp: snp_r2_dict[snp] for snp in column_names}

        for chrom in chromosomes:
            chr_snps_df = genotype_df_columns[genotype_df_columns['chrom'] == chrom]
            chr_snps = chr_snps_df['snp'].tolist()

            groups = []
            current_group = [(chr_snps_df.iloc[0]['snp'], chr_snps_df.iloc[0]['pos'])]

            for i in range(1, len(chr_snps)):
                curr_snp = (chr_snps_df.iloc[i]['snp'], chr_snps_df.iloc[i]['pos'])
                prev_snp = (chr_snps_df.iloc[i - 1]['snp'], chr_snps_df.iloc[i - 1]['pos'])
                if abs(curr_snp[1] - prev_snp[1]) <= max_distance:
                    current_group.append(curr_snp)
                else:
                    groups.append(current_group)
                    current_group = [curr_snp]
            groups.append(current_group)

            groups = remove_subsets(groups)
            groups = [[snp[0] for snp in group] for group in groups]

            # LD Pruning and Conditional Analysis within each group
            for group in groups:
                ld_removed_snps_in_group = set()
                if len(group) == 1:
                    snp = group[0]
                    final_selected_snps.append(snp)
                    continue
                group_df = genotype_df_original[group]
                snp_list = group_df.columns.tolist()

                for i, snp1 in enumerate(snp_list):
                    if snp1 in ld_removed_snps:
                        continue
                    for j in range(i + 1, len(snp_list)):
                        snp2 = snp_list[j]
                        if snp2 in ld_removed_snps:
                            continue
                        ld_value = calculate_ld(genotype_df_original, snp1, snp2)
                        if ld_value > ld_threshold:
                            if marginal_r2[snp1] > marginal_r2[snp2]:
                                ld_removed_snps.add(snp2)
                                ld_removed_snps_in_group.add(snp2)
                                anchor_snp_details[snp2] = f"chr{snp1}" # hold the name of the anchor SNP which pruned the SNP
                            else:
                                ld_removed_snps.add(snp1)
                                ld_removed_snps_in_group.add(snp1)
                                anchor_snp_details[snp1] = f"chr{snp2}" # hold the name of the anchor SNP which pruned the SNP

                non_pruned_snps_in_group = [s for s in snp_list if s not in ld_removed_snps_in_group] # remaining SNPs after LD pruning
                # update the snp_details_after_ld dictionary for the pruned SNPs
                for snp in ld_removed_snps_in_group:
                    snp_details_after_ld[snp] = {
                        "pruned": True,
                        "reason": "LD",
                        "threshold": self.threshold,
                        "genomic_distance": self.genomic_distance,
                        "anchor_snp": anchor_snp_details[snp]
                    }

                if len(non_pruned_snps_in_group) == 0:
                    continue

                # Start Conditional Analysis
                snps_remaining_for_ca = non_pruned_snps_in_group[:]
                final_group_snps = []

                if len(snps_remaining_for_ca) < 2:
                    final_group_snps.extend(snps_remaining_for_ca)
                    final_selected_snps.extend(final_group_snps)
                    # move to next group
                    continue

                peak_snp = max(snps_remaining_for_ca, key=lambda x: marginal_r2[x])
                X_peak = genotype_df_encoded[[peak_snp]]
                p_values = []
                tested_snps = []

                for snp in snps_remaining_for_ca:
                    if snp == peak_snp:
                        continue
                    X_full = pd.concat([X_peak, genotype_df_encoded[[snp]]], axis=1)
                    X_full_const = sm.add_constant(X_full, has_constant='add') # add constant term for statsmodels
                    model = sm.OLS(y, X_full_const).fit() # fit the model
                    # Calculate p-value using Wald test
                    # get the index term of the snp to be checked
                    term_index = X_full_const.columns.get_loc(snp)
                    # r_matrix to test if the coefficient of the snp is equal to 0
                    r_matrix = np.zeros((1, X_full_const.shape[1]))
                    r_matrix[0, term_index] = 1.0
                    # perform the Wald test
                    wald_test = model.wald_test(r_matrix, scalar=False) # scalar=False returns a scalar value
                    # get the p-value from the Wald test
                    p_val = wald_test.pvalue
                    p_values.append(p_val)
                    tested_snps.append(snp)

                if not p_values: # If no SNPs were tested, break the while loop
                    break

                p_values = np.array(p_values).flatten()
                if len(p_values) == 1:
                    rejected = [p_values[0] < 0.05]
                else:
                    rejected, _, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

                to_remove = {s for s, r in zip(tested_snps, rejected) if not r} # if rejected is False, then we remove the SNP
                # update the snp_details_after_ld for the SNPs that are to be removed
                for snp in to_remove:
                    snp_details_after_ld[snp] = {
                        "pruned": True,
                        "reason": "CA",
                        "threshold": self.threshold,
                        "genomic_distance": self.genomic_distance,
                        "anchor_snp": f"chr{peak_snp}"
                    }
                    ld_removed_snps.add(snp)
                    anchor_snp_details[snp] = f"chr{peak_snp}"
                for snp in snps_remaining_for_ca:
                    if snp not in to_remove and snp != peak_snp:
                        final_group_snps.append(snp)
                # Add the peak SNP to the final group regardless of other SNPs
                final_group_snps.append(peak_snp)
                final_selected_snps.extend(final_group_snps) # final SNPs after LD pruning and CA added to the list

        self.final_selected_snps = list(set(final_selected_snps))
        self.name_of_selected_features = list(set(final_selected_snps))
        self.ld_removed_snps = ld_removed_snps
        self.bool_mask = genotype_df_original.columns.isin(self.final_selected_snps)
        self.snp_details_after_ld = snp_details_after_ld
        self.selected_features_ = np.array(self.final_selected_snps)
        return self

    def transform(self, X):
        """
        Transform the data to include only the selected features.

        Parameters:
        - X (array-like): The input features (2D array of shape [n_samples, n_features]).

        Returns:
        - X_transformed (array-like): The transformed array with only selected features.
        """
        if self.selected_features_ is None:
            raise RuntimeError("LDSelector has not been fitted yet.")
        return X[:, self.bool_mask]

    def mutate(self, rng: rng_t):

        # shift is a rng from normal distribution with a change in 1st decimal place
        shift = 0.05 * rng.choice([-1.0, 1.0])

        # check if the threshold is going to be less than 0.1
        if self.threshold + shift < float32_t(0.5):
            self.threshold = float32_t(0.5)
        # check if the threshold is going to be greater than 1
        elif self.threshold + shift > float32_t(0.95):
            self.threshold = float32_t(0.95)
        # if neither of the above, then we can just add the shift
        else:
            self.threshold = self.threshold + shift

        # increment genomic distance by 100000 with a minimum of 500000 and maximum of 1000000, in increments of 100000
        genomic_distance_shift = np.int32(rng.choice([-100000, 100000]))
        # check if the genomic_distance is going to be less than 500000
        if self.genomic_distance + genomic_distance_shift < 500000:
            self.genomic_distance = 500000
        # check if the genomic_distance is going to be greater than 1000000
        elif self.genomic_distance + genomic_distance_shift > 1000000:
            self.genomic_distance = 1000000
        # if neither of the above, then we can just add the shift
        else:
            self.genomic_distance = self.genomic_distance + genomic_distance_shift

        # initialize the selector with the new threshold
        self.params['threshold'] = float32_t(self.threshold)
        self.params['genomic_distance'] = int(self.genomic_distance)

    def get_feature_count(self):
        """
            Get the number of features selected by the selector.

            Returns:
            - int: The number of features selected. If the selector has not been fitted yet,
           raises a RuntimeError.
        """
        if self.selected_features_ is None:
            raise RuntimeError("LDSelector has not been fitted yet.")

        return len(self.selected_features_)
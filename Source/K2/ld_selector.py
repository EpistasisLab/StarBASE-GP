##########################################################################################
############################ the ld classes ##############################################
##########################################################################################

from ..Base.selectors import SelectorNode
from ..Base.types import (rng_t, float32_t, interaction_t, snp_t)

from decimal import Decimal
import numpy as np
from typeguard import typechecked
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import numba

@numba.njit(cache=True)
def calculate_ld_numba(snpa1, snpa2, snpb1, snpb2):
    """
    Fast LD (R²) calculation between two SNP interaction pairs using numba.
    Computes Pearson correlation coefficient and squares it.
    Calculates the coefficient between snpa1 and snpb1, and between snpa2 and snpb2, 
    and returns two R² coefficients. Both coefficients should be greater than the 
    threshold to consider the interaction terms redundant.
    
    Parameters:
        snpa1, snpa2: Genotype arrays for SNP pair A
        snpb1, snpb2: Genotype arrays for SNP pair B
    
    Returns:
        r2_a1_b1, r2_a2_b2: R² values for the two SNP pair correlations
    """
    n = len(snpa1)
    
    # Calculate means
    mean_a1 = np.mean(snpa1)
    mean_a2 = np.mean(snpa2)
    mean_b1 = np.mean(snpb1)
    mean_b2 = np.mean(snpb2)
    
    # Calculate sum of cross-products (numerator) and sum of squares (denominator components)
    sum_prod_a1_b1 = 0.0
    sum_sq_a1 = 0.0
    sum_sq_b1 = 0.0
    
    sum_prod_a2_b2 = 0.0
    sum_sq_a2 = 0.0
    sum_sq_b2 = 0.0
    
    for i in range(n):
        diff_a1 = snpa1[i] - mean_a1
        diff_b1 = snpb1[i] - mean_b1
        diff_a2 = snpa2[i] - mean_a2
        diff_b2 = snpb2[i] - mean_b2
        
        sum_prod_a1_b1 += diff_a1 * diff_b1
        sum_sq_a1 += diff_a1 * diff_a1
        sum_sq_b1 += diff_b1 * diff_b1
        
        sum_prod_a2_b2 += diff_a2 * diff_b2
        sum_sq_a2 += diff_a2 * diff_a2
        sum_sq_b2 += diff_b2 * diff_b2
    
    # Calculate correlation coefficients (no division by n needed - it cancels out!)
    # r = Σ[(x-μx)(y-μy)] / sqrt(Σ(x-μx)² * Σ(y-μy)²)
    r_a1_b1 = sum_prod_a1_b1 / np.sqrt(sum_sq_a1 * sum_sq_b1) if sum_sq_a1 > 0 and sum_sq_b1 > 0 else 0.0
    r_a2_b2 = sum_prod_a2_b2 / np.sqrt(sum_sq_a2 * sum_sq_b2) if sum_sq_a2 > 0 and sum_sq_b2 > 0 else 0.0
    
    return r_a1_b1 ** 2, r_a2_b2 ** 2

@numba.njit(cache=True, parallel=True)
def compute_ld_matrix(X_array, snp_indices):
    """
    Compute pairwise LD matrix for a subset of interaction terms.
    Uses parallel computation for speed.

    Parameters:
        X_array: 2D array of shape (n_samples, n_snps)
        snp_indices: indices of SNPs to compute LD for

    Returns:
        ld_matrix: symmetric matrix of LD values
    """

    n_snps = len(snp_indices)
    ld_matrix = np.zeros((n_snps, n_snps), dtype=np.float32)

    for i in numba.prange(n_snps):
        idx_i = snp_indices[i]
        for j in range(i + 1, n_snps):
            idx_j = snp_indices[j]
            ld_val = calculate_ld_numba(X_array[:, idx_i], X_array[:, idx_j])
            ld_matrix[i, j] = ld_val
            ld_matrix[j, i] = ld_val  # Symmetric

    return ld_matrix

@numba.njit(fastmath=True, cache=True)
def prune_snps_by_ld(ld_matrix, marginal_r2_array, ld_threshold):
    """
    Identify interaction terms to remove based on LD threshold.
    Keeps interaction term with higher marginal R².

    Parameters:
        ld_matrix: pairwise LD matrix
        marginal_r2_array: array of marginal R² values for each interaction term in their best inheritence model already computed
        ld_threshold: LD threshold for pruning

    Returns:
        pruned_indices: set of indices to remove
        anchor_indices: anchor interaction term index for each pruned interaction term (-1 if not pruned)
    """
    
    n_snps = ld_matrix.shape[0] # number of SNPs
    pruned = np.zeros(n_snps, dtype=np.bool_) # boolean array to keep track of which SNPs are pruned
    anchor_indices = np.full(n_snps, -1, dtype=np.int32) # array to store the index of the anchor SNP for each pruned SNP, initialized to -1 (indicating no anchor), required to store in the epi hub.

    for i in range(n_snps):
        if pruned[i]:
            continue
        for j in range(i + 1, n_snps):
            if pruned[j]:
                continue
            if ld_matrix[i, j] > ld_threshold:
                if marginal_r2_array[i] > marginal_r2_array[j]:
                    pruned[j] = True
                    anchor_indices[j] = i
                else:
                    pruned[i] = True
                    anchor_indices[i] = j
                    break  # Move to next i since i is pruned

    return pruned, anchor_indices

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

        self.threshold = self.params['threshold']
        self.genomic_distance = self.params['genomic_distance']
        self.selected_features_ = None
        self.bool_mask = None
        self.name_of_selected_features = None

        # dictionary to store the details of interactions after LD pruning - pruned flag, reason for pruning, threshold used, genomic distance used, and anchor interaction term
        self.interaction_details_after_ld = {}

    def fit(self, component_map, y, interaction_r2_dict):
        """
        Fit the LDSelector to the data by performing LD pruning and conditional analysis on interactions.
        Uses component map structure for efficient access to interaction components.
        
        Parameters:
            component_map: Dictionary mapping interaction names to their components:
                {interaction_name: {'snp1_name', 'snp2_name', 'snp1_data', 'snp2_data', 'encoded_data'}}
                Note: Data arrays should be numpy arrays, not Ray ObjectIDs (those should be resolved before calling fit)
            y: Target variable (phenotype)
            interaction_r2_dict: Dictionary mapping interaction names to their marginal R² values
        
        Returns:
            self: The fitted LDSelector instance with selected features and details stored.
        """
        
        if len(component_map) == 0:
            self.selected_features_ = None
            return self

        # Extract interaction names from component map - tuple containing SNP1 and SNP2 names (e.g., 2.191, 3.197)
        column_names = list(component_map.keys())
        #print("Column names from component map:", column_names)  # Debugging statement to check the format of column names
        #print("Type of column names:", type(column_names[0]))  # Check the type of the first column name to confirm if it's a tuple or string
        
        # Build DataFrame for all the snp1 data with their corresponding names, and similarly for snp2 data and encoded interaction data, using the component map for efficient access
        # Only the first part of the tuple should be the column name, not the entire tuple, since the entire tuple is the interaction name and we want to keep the original SNP names as column names for LD calculation
        inter_snp1_original = pd.DataFrame({
            key[0] if isinstance(key, tuple) else key: component_map[key]['snp1_data']
            for key in column_names
        }) 
        #print("Shape of inter_snp1_original:", inter_snp1_original.shape)  # Debugging statement to check the shape of the DataFrame
        #print("inter_snp1_original columns:", inter_snp1_original.columns)  # Debugging statement to check column names
        
        inter_snp2_original = pd.DataFrame({
            key[1] if isinstance(key, tuple) else key: component_map[key]['snp2_data']
            for key in column_names
        })
        #print("Shape of inter_snp2_original:", inter_snp2_original.shape)  # Debugging statement to check the shape of the DataFrame
        #print("inter_snp2_original columns:", inter_snp2_original.columns)  # Debugging statement to check column names
        
        # The column names for the encoded interaction data can be the same as the original interaction names (e.g., "2.191_3.197") since they are only used for conditional analysis and not for LD calculation, so we can keep them as is without needing to convert to string keys.
        inter_encoded = pd.DataFrame({
            f"{key[0]}_{key[1]}": component_map[key]['encoded_data']
            for key in column_names
        })
        #print("Shape of inter_encoded:", inter_encoded.shape)  # Debugging statement to check the shape of the DataFrame
        #print("inter_encoded columns:", inter_encoded.columns)  # Debugging statement to check column names

        ld_threshold = self.threshold
        max_distance = self.genomic_distance
        final_selected_interactions = []
        ld_removed_interactions = set()
        anchor_interaction_details = {}
        interaction_details_after_ld = {}
        
        # Initialize the interaction_details_after_ld dictionary (using original keys)
        for interaction in column_names:
            interaction_details_after_ld[interaction] = {
                "pruned": False, 
                "reason": "", 
                "threshold": self.threshold, 
                "genomic_distance": self.genomic_distance, 
                "anchor_interaction": ""
            }

        # Parse interaction names to extract chromosome and position information
        interaction_info = []
        for interaction in column_names:
            # Handle tuple format: (snp1, snp2) where each snp is like "chr.pos"
            assert isinstance(interaction, tuple)
            snp1, snp2 = interaction # interaction is a tuple of (snp1, snp2) where each snp is like "chr.pos"

            chr1, pos1 = int(snp1.split('.')[0]), int(snp1.split('.')[1])
            chr2, pos2 = int(snp2.split('.')[0]), int(snp2.split('.')[1])
            interaction_info.append({
                'interaction': interaction,
                'interaction_str': f"{snp1}_{snp2}",  # String representation for easier handling in DataFrames and dictionaries
                'chr1': chr1,
                'pos1': pos1,
                'chr2': chr2,
                'pos2': pos2,
                'chr_pair': (chr1, chr2)
            })
        
        interaction_df = pd.DataFrame(interaction_info)
        
        marginal_r2 = pd.Series({interaction: interaction_r2_dict[interaction] for interaction in column_names})
        # Create a string-indexed version for easier access
        marginal_r2_str = pd.Series({f"{k[0]}_{k[1]}": v for k, v in marginal_r2.items()})

        # Group interactions by chromosome pair
        chr_pair_groups = interaction_df.groupby('chr_pair')

        for chr_pair, group_df in chr_pair_groups:
            # Sort by both SNP positions for efficient grouping
            group_df = group_df.sort_values(['pos1', 'pos2']).reset_index(drop=True)
            
            if len(group_df) == 0:
                continue
            
            # Apply greedy grouping approach within chromosome pair
            # Two interactions are in the same group if BOTH pos1 and pos2 of the interactions are within max_distance
            groups = []
            current_group = [group_df.iloc[0]]
            
            for i in range(1, len(group_df)):
                curr_interaction = group_df.iloc[i]
                prev_interaction = group_df.iloc[i - 1]
                
                # Check if both positions are within genomic distance threshold
                pos1_within_distance = abs(curr_interaction['pos1'] - prev_interaction['pos1']) <= max_distance
                pos2_within_distance = abs(curr_interaction['pos2'] - prev_interaction['pos2']) <= max_distance
                
                if pos1_within_distance and pos2_within_distance:
                    # Both positions within distance - add to current group
                    current_group.append(curr_interaction)
                else:
                    # Distance threshold exceeded - finalize current group and start new one
                    groups.append(current_group)
                    current_group = [curr_interaction]
            
            # Don't forget the last group
            groups.append(current_group)

            # make sure there is no subset of groups - if there is a group that is a subset of another group, we can just keep the larger group and remove the smaller group
            # this is because if a group is a subset of another group, then the interactions in the smaller group will be pruned by the interactions in the larger group, so we can just keep the larger group and remove the smaller group
            # we can check for subsets by checking if the set of interactions in one group is a subset of the set of interactions in another group
            unique_groups = []
            for i, group in enumerate(groups):
                # Convert to native Python types to avoid numpy array issues
                group_interactions = set(str(row['interaction_str']) if hasattr(row['interaction_str'], 'item') else row['interaction_str'] for row in group)
                is_subset = False
                for j, other_group in enumerate(groups):
                    if i == j:  # Use index comparison instead of object comparison
                        continue
                    other_group_interactions = set(str(row['interaction_str']) if hasattr(row['interaction_str'], 'item') else row['interaction_str'] for row in other_group)
                    if group_interactions.issubset(other_group_interactions) and group_interactions != other_group_interactions:
                        is_subset = True
                        break
                if not is_subset:
                    unique_groups.append(group)
            
            # Process each sub-group for LD pruning and conditional analysis
            for group in unique_groups:
                # Convert to native Python strings to avoid numpy issues
                interactions_in_group = [
                    str(row['interaction_str']) if hasattr(row['interaction_str'], 'item') else row['interaction_str'] 
                    for row in group
                ]  # Use string keys
                final_group_interactions = [] # to store interactions that survive LD pruning and conditional analysis in this group
                
                # if only one interaction in the group, we can just keep it without LD checking or conditional analysis
                if len(interactions_in_group) == 1:
                    final_group_interactions.append(interactions_in_group[0])
                    final_selected_interactions.extend(final_group_interactions)
                    continue

                # Pairwise LD checking within this sub-group
                # Since we're in the same greedy group, all pairs need LD computation
                ld_removed_in_group = set() # to keep track of interactions removed in this group based on LD, required to store in the epi hub.
                
                for i in range(len(interactions_in_group)):
                    int_a_name = interactions_in_group[i]
                    #print(f"Processing interaction {int_a_name} in group with interactions: {interactions_in_group}")  # Debugging statement to track progress and check interaction names
                    
                    # Skip if already pruned
                    if int_a_name in ld_removed_in_group:
                        continue
                    
                    for j in range(i + 1, len(interactions_in_group)):
                        int_b_name = interactions_in_group[j]
                        #print(f"Comparing interaction {int_a_name} with interaction {int_b_name} for LD pruning")  # Debugging statement to track which interactions are being compared
                        
                        # Skip if already pruned
                        if int_b_name in ld_removed_in_group:
                            continue
                        
                        # Calculate LD between the constituent SNP pairs using the original SNP encodings
                        # Split interaction strings to get individual SNP names
                        snp_a1_name, snp_a2_name = int_a_name.split('_')
                        snp_b1_name, snp_b2_name = int_b_name.split('_')
                        
                        snp_a1 = inter_snp1_original[snp_a1_name].values
                        snp_a2 = inter_snp2_original[snp_a2_name].values
                        snp_b1 = inter_snp1_original[snp_b1_name].values
                        snp_b2 = inter_snp2_original[snp_b2_name].values

                        # Get both correlation coefficients (r² values)
                        r2_a1_b1, r2_a2_b2 = calculate_ld_numba(snp_a1, snp_a2, snp_b1, snp_b2)
                        
                        # Both coefficients must exceed threshold to consider interactions redundant
                        if r2_a1_b1 > ld_threshold and r2_a2_b2 > ld_threshold:
                            if marginal_r2_str[int_a_name] > marginal_r2_str[int_b_name]:
                                ld_removed_in_group.add(int_b_name)
                                ld_removed_interactions.add(int_b_name)  # Add string key
                                anchor_interaction_details[int_b_name] = int_a_name
                            else:
                                ld_removed_in_group.add(int_a_name)
                                ld_removed_interactions.add(int_a_name)  # Add string key
                                anchor_interaction_details[int_a_name] = int_b_name
                                break  # Move to next i since i is pruned

                # Update interaction details for pruned interactions (convert string keys back to original)
                for interaction_str in ld_removed_in_group:
                    # Convert string back to tuple for original key
                    original_key = tuple(interaction_str.split('_'))
                    anchor_str = anchor_interaction_details[interaction_str]
                    anchor_original = tuple(anchor_str.split('_'))
                    interaction_details_after_ld[original_key] = {
                        "pruned": True,
                        "reason": "LD",
                        "threshold": self.threshold,
                        "genomic_distance": self.genomic_distance,
                        "anchor_interaction": anchor_original
                    }

                # Get non-pruned interactions for conditional analysis
                non_pruned_interactions = [i for i in interactions_in_group if i not in ld_removed_in_group]
                
                # if no interactions remain after LD pruning, skip to next group - should not be happening since at least one interaction should remain as the anchor, but just in case
                if len(non_pruned_interactions) == 0:
                    continue
                
                # Conditional Analysis on remaining interactions in the group

                # if only one interaction remains after LD pruning, we can just keep it without conditional analysis
                if len(non_pruned_interactions) < 2:
                    final_group_interactions.extend(non_pruned_interactions)
                    final_selected_interactions.extend(final_group_interactions)
                    continue

                # Find peak interaction (highest marginal R²)
                peak_interaction = max(non_pruned_interactions, key=lambda x: marginal_r2_str[x])
                X_peak = inter_encoded[[peak_interaction]]
            
                p_values = []
                tested_interactions = []

                for interaction in non_pruned_interactions:
                    if interaction == peak_interaction:
                        continue
                    
                    # Test if interaction is significant after conditioning on peak
                    X_full = pd.concat([X_peak, inter_encoded[[interaction]]], axis=1)
                    X_full_const = sm.add_constant(X_full, has_constant='add')
                    model = sm.OLS(y, X_full_const).fit()
                    
                    # Wald test for the interaction term
                    term_index = X_full_const.columns.get_loc(interaction)
                    r_matrix = np.zeros((1, X_full_const.shape[1]))
                    r_matrix[0, term_index] = 1.0
                    wald_test = model.wald_test(r_matrix, scalar=False)
                    p_val = wald_test.pvalue
                    
                    p_values.append(p_val)
                    tested_interactions.append(interaction)

                if not p_values: # No interactions to test after LD pruning, just add the peak interaction
                    final_group_interactions.append(peak_interaction)
                    final_selected_interactions.extend(final_group_interactions)
                    continue

                # Multiple testing correction
                p_values = np.array(p_values).flatten()
                if len(p_values) == 1:
                    rejected = [p_values[0] < 0.05]
                else:
                    rejected, _, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh') # rejected is a boolean array indicating which interactions are significant after correction; if TRUE, then the interaction is significant after conditioning on the peak interaction, if FALSE, then the interaction is not significant after conditioning on the peak interaction and can be removed.

                # Remove interactions that are not significant after conditioning on the peak interaction
                to_remove = {i for i, r in zip(tested_interactions, rejected) if not r}
                
                for interaction_str in to_remove:
                    original_key = tuple(interaction_str.split('_'))
                    peak_original = tuple(peak_interaction.split('_'))
                    interaction_details_after_ld[original_key] = {
                        "pruned": True,
                        "reason": "CA",
                        "threshold": self.threshold,
                        "genomic_distance": self.genomic_distance,
                        "anchor_interaction": peak_original
                    }
                    ld_removed_interactions.add(interaction_str)
                    anchor_interaction_details[interaction_str] = peak_interaction

                # Add peak interaction and significant interactions
                final_group_interactions = [peak_interaction]
                for interaction in non_pruned_interactions:
                    if interaction not in to_remove and interaction != peak_interaction:
                        final_group_interactions.append(interaction)
                
                # Add this group's interactions to the final list
                final_selected_interactions.extend(final_group_interactions)
                
        # Store the interaction details after all chromosome pairs are processed
        self.interaction_details_after_ld = interaction_details_after_ld
        
        # Convert final selected interactions to list of tuples for selected_features_ (convert string keys back to tuple keys)
        self.selected_features_ = [tuple(str_key.split('_')) for str_key in final_selected_interactions]
        # Create bool mask by checking if column_names (tuples) are in selected_features_ (tuples)
        self.bool_mask = np.array([col in self.selected_features_ for col in column_names])
        self.name_of_selected_features = [column_names[i] for i in range(len(column_names)) if self.bool_mask[i]]

        
        return self

    def transform(self, X):
        """
        Transform the data to include only the selected features (interactions).

        Parameters:
            X (array-like): The input features (2D array of shape [n_samples, n_features]).

        Returns:
            X_transformed (array-like): The transformed array with only selected features.
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
            int: The number of features selected. If the selector has not been fitted yet, raises a RuntimeError.
        """

        if self.selected_features_ is None:
            raise RuntimeError("LDSelector has not been fitted yet.")

        return len(self.selected_features_)
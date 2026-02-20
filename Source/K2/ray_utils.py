from ..Base.selectors import SelectorNode
from ..Base.types import (float32_t, int16_t, snp_t, uint16_t, uint32_t, interaction_t)

import ray
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
from typing import List, Dict, Tuple, Set
import logging
from sklearn.inspection import permutation_importance
from mdr import ContinuousMDR
import pandas as pd
import numpy.typing as npt
import numba
from statsmodels.regression.linear_model import OLS


# Helper function to create interaction component mapping
def create_interaction_component_map(interaction_names: List[snp_t],
                                      snp1_ray_ids: List[ray.ObjectID],
                                      snp2_ray_ids: List[ray.ObjectID],
                                      encoded_ray_ids: List[ray.ObjectID]) -> Dict[snp_t, Dict]:
    """
    Create a mapping structure for interaction components.
    This efficiently handles shared SNPs across multiple interactions.

    Parameters:
        interaction_names: List of interaction names (format: 'snp1_snp2')
        snp1_ray_ids: List of Ray ObjectIDs for first component SNPs
        snp2_ray_ids: List of Ray ObjectIDs for second component SNPs
        encoded_ray_ids: List of Ray ObjectIDs for encoded interactions

    Returns:
        Dictionary mapping interaction names to their component info
    """
    component_map = {}
    for i, interaction_name in enumerate(interaction_names):
        snp1_name, snp2_name = interaction_name.split('_')
        component_map[interaction_name] = {
            'snp1_name': snp1_name,
            'snp2_name': snp2_name,
            'snp1_ray_id': snp1_ray_ids[i],
            'snp2_ray_id': snp2_ray_ids[i],
            'encoded_ray_id': encoded_ray_ids[i]
        }
    return component_map


# Function for Cartesian encoding for interactions
@numba.njit(cache=True)
def encode_cartesian(X1, X2):
    """
    Encode interactions using Cartesian product of two SNP vectors.
    For genotypes 0.0, 0.5, 1.0, the Cartesian encoding results in unique values.
    Optimized with numba for maximum speed.

    Parameters:
        X1, X2: Genotype vectors (numpy arrays) with values in {0.0, 0.5, 1.0}
    Returns:
        out: Encoded interaction vector - simple element-wise multiplication
    """
    return X1 * X2

# Function for XOR encoding for interactions
@numba.njit(cache=True)
def encode_xor(X1, X2):
    """
    Encode interactions using XOR logic: (SNP1 % 2 + SNP2 % 2) % 2
    Optimized with numba to minimize operations.

    Parameters:
        X1, X2: Genotype vectors (numpy arrays) with values in {0.0, 0.5, 1.0}
    Returns:
        out: XOR encoded vector with values in {0.0, 0.5, 1.0}
    """
    n = len(X1)
    out = np.empty(n, dtype=np.float64)

    for i in range(n):
        # Direct computation: (x1%2 + x2%2)%2
        # For 0.0, 0.5, 1.0: mod 2 gives 0.0, 0.5, 1.0
        val = (X1[i] % 2.0 + X2[i] % 2.0) % 2.0
        out[i] = val

    return out

# Function to return MDR mapping from training data
def build_mdr_mapping(X1_train, X2_train, y_train):
    """
    Build MDR mapping from training data for two SNPs and phenotype.
    Uses sklearn's ContinuousMDR to classify genotype combinations.

    Parameters:
        X1_train, X2_train: Training genotype vectors
        y_train: Training phenotype vector

    Returns:
        mdr: Fitted ContinuousMDR model containing the feature_map for encoding
    """
    mdr = ContinuousMDR()
    X_combined = np.column_stack((X1_train, X2_train))
    mdr.fit(X_combined, y_train)
    return mdr.feature_map

# Function for MDR encoding using pre-computed mapping
def encode_mdr(X1, X2, y=None):
    """
    Encode interactions using MDR mapping.
    If mdr_mapping is provided, uses it directly.
    Otherwise, builds mapping from training data using y.

    Parameters:
        X1, X2: Full genotype vectors
        y: Full phenotype vector (used for building mapping if mdr_mapping not provided)
        mdr_mapping: Optional pre-computed MDR feature_map
    Returns:
        out: MDR encoded vector with values in {0.0, 1.0} based on the mapping
        mdr: Fitted ContinuousMDR model containing the feature_map for encoding
        mapping: The MDR feature_map used for encoding
    """
    X_combined = np.column_stack((X1, X2))
    mdr = ContinuousMDR()
    mdr.fit(X_combined, y)
    out = mdr.transform(X_combined)
    mapping = mdr.feature_map

    return out, mdr, mapping

# # ray function to evaluate all encodings for a pair of snps in one call to reduce ray scheduling overhead
# @ray.remote
# def ray_interaction_eval_all_encodings(X1, X2, y, train_idx, valid_idx, snp):
#     """
#     Evaluate all 3 encoding types for a pair of SNPs in one Ray call.
#     This dramatically reduces Ray scheduling overhead by batching all encodings together.

#     For MDR: Builds mapping from training data, applies to validation.
#     For Cartesian/XOR: Direct encoding on full dataset.

#     Returns:
#         Dict[str, Tuple[float, str, str, float, dict|None]]:
#             Dictionary mapping encoding name to (r2_score, snp, encoding, error_flag, mdr_mapping)
#             mdr_mapping is only populated for 'mdr' encoding, None for others
#     """
#     assert isinstance(X1, np.ndarray), "X1 should be a numpy array"
#     assert isinstance(X2, np.ndarray), "X2 should be a numpy array"

#     results = {}

#     # Cartesian encoding
#     try:
#         X_encoded = encode_cartesian(X1, X2)

#         # Fit OLS model on training data
#         regressor = sm.OLS(y[train_idx], sm.add_constant(X_encoded[train_idx], has_constant='add'))
#         fit_results = regressor.fit()

#         # Score on validation set
#         y_pred = fit_results.predict(sm.add_constant(X_encoded[valid_idx], has_constant='add'))
#         score = r2_score(y[valid_idx], y_pred)

#         results['cartesian'] = (float32_t(score), snp, snp_t('cartesian'), float32_t(1.0), None)
#     except Exception as e:
#         logging.error(f"Error evaluating cartesian for SNP {snp}: {e}")
#         results['cartesian'] = (float32_t(0.0), snp, snp_t('cartesian'), float32_t(-1.0), None)

#     # XOR encoding
#     try:
#         X_encoded = encode_xor(X1, X2)

#         # Fit OLS model on training data
#         regressor = sm.OLS(y[train_idx], sm.add_constant(X_encoded[train_idx], has_constant='add'))
#         fit_results = regressor.fit()

#         # Score on validation set
#         y_pred = fit_results.predict(sm.add_constant(X_encoded[valid_idx], has_constant='add'))
#         score = r2_score(y[valid_idx], y_pred)
#         results['xor'] = (float32_t(score), snp, snp_t('xor'), float32_t(1.0), None)
#     except Exception as e:
#         logging.error(f"Error evaluating xor for SNP {snp}: {e}")
#         results['xor'] = (float32_t(0.0), snp, snp_t('xor'), float32_t(-1.0), None)

#     # MDR encoding
#     try:
#         X_encoded, mapping = encode_mdr(X1, X2, y)

#         # Fit OLS model on training data
#         regressor = sm.OLS(y[train_idx], sm.add_constant(X_encoded[train_idx], has_constant='add'))
#         fit_results = regressor.fit()

#         # Score on validation set
#         y_pred = fit_results.predict(sm.add_constant(X_encoded[valid_idx], has_constant='add'))
#         score = r2_score(y[valid_idx], y_pred)

#         results['mdr'] = (float32_t(score), snp, snp_t('mdr'), float32_t(1.0), mapping)
#     except Exception as e:
#         logging.error(f"Error evaluating MDR for SNP {snp}: {e}")
#         results['mdr'] = (float32_t(0.0), snp, snp_t('mdr'), float32_t(-1.0), None)

#     return results

# This function will be used when encoding the validation/test set not during initial evaluation of the interaction.
@ray.remote
def ray_interaction_encoder(X1, X2, y, train_idx, enc: snp_t, snp: snp_t, mdr_mapping=None) -> Tuple[np.ndarray, snp_t]:
    """
    Efficiently encode interaction features using specified encoding pattern.
    For MDR encoding, uses pre-computed mapping from evaluation phase.
    For Cartesian/XOR, encodes directly.
    Optimized with numba for maximum speed.

    Parameters:
        X1, X2: Full genotype vectors
        y: Full phenotype vector (used for MDR if mapping not provided)
        train_idx: Training indices (used for MDR if mapping not provided)
        enc: Encoding type ('cartesian', 'xor', or 'mdr')
        snp: SNP pair identifier

    Returns:
        X_encoded: Encoded interaction vector
        snp: SNP pair identifier
    """
    assert isinstance(X1, np.ndarray), "X1 should be a numpy array"
    assert isinstance(X2, np.ndarray), "X2 should be a numpy array"
    assert isinstance(y, np.ndarray), "y should be a numpy array"
    assert isinstance(enc, snp_t), "enc should be a numpy string"

    # Get the encoding string
    enc_str = str(enc) if isinstance(enc, np.str_) else enc

    if enc_str == 'cartesian':
        X_encoded = encode_cartesian(X1, X2)
        return X_encoded, snp

    elif enc_str == 'xor':
        X_encoded = encode_xor(X1, X2)
        return X_encoded, snp

    elif enc_str == 'mdr':
        # fit MDR mapping if not provided (should have been computed during evaluation)
        if mdr_mapping is None:
            mdr_mapping = encode_mdr(X1[train_idx], X2[train_idx], y[train_idx])[2]  # get the mapping from the tuple returned by encode_mdr
            mdr_fitted_object = encode_mdr(X1[train_idx], X2[train_idx], y[train_idx])[1]  # get the fitted MDR object to use for transform
        X_encoded = mdr_fitted_object.transform(np.column_stack((X1, X2))) # mdr package has built in transform function to apply the mapping to the full dataset
        return X_encoded, snp

    else:
        raise ValueError(f"Unknown encoding type: {enc_str}")

# ray remote function to pre-screen an interaction (MLG + Pearson correlation check)
@ray.remote
def ray_prescreen_interaction(X1: np.ndarray, X2: np.ndarray, full_train_idx: npt.NDArray, snp1: snp_t, snp2: snp_t) -> Tuple[bool, str, float32_t]:
    """
    Pre-screen an interaction by checking for missing multi-locus genotype (MLG) and high Pearson correlation.
    This is called ONCE per interaction (not per CV fold) to avoid redundant checks.

    Parameters:
        X1, X2: Genotype vectors
        full_train_idx: Full training indices for MLG and correlation checks
        snp1, snp2: SNP identifiers

    Returns:
        pass_flag (bool): True if passed both checks, False if failed
        failure_code (float32_t): -1.0 if MLG missing, -2.0 if high correlation, 1.0 if passed
        correlation_r2 (float32_t): Pearson's R² between SNPs (-1.0 if not computed)
    """

    correlation_r2 = float32_t(-1.0)

    # Step 1: Check for missing multi-locus genotype (MLG) in training set
    all_combinations = {(g1, g2) for g1 in [0.0, 0.5, 1.0] for g2 in [0.0, 0.5, 1.0]}
    full_train_combinations = set(zip(X1[full_train_idx], X2[full_train_idx]))
    missing_train = not all_combinations.issubset(full_train_combinations)

    if missing_train:
        return False, float32_t(-1.0), correlation_r2

    # Step 2: Check Pearson's correlation if SNPs are on the same chromosome
    snp1_chr = snp1.split('_')[0]
    snp2_chr = snp2.split('_')[0]

    if snp1_chr == snp2_chr:
        correlation_coef = float32_t(np.corrcoef(X1[full_train_idx], X2[full_train_idx])[0, 1])
        correlation_r2 = float32_t(correlation_coef ** 2)
        if correlation_r2 > 0.50:
            return False, 'pearson', correlation_r2

    # Passed both checks
    return True, float32_t(1.0), correlation_r2


# ray remote function to evaluate all encodings for a single CV fold
@ray.remote
def ray_evaluate_interaction_encodings(X1: ray.ObjectID, X2: ray.ObjectID, y: ray.ObjectID,
                                       train_idx: npt.NDArray, valid_idx: npt.NDArray,
                                       snp1: snp_t, snp2: snp_t) -> Tuple[Dict[str, float32_t], Dict]:
    """
    Evaluate all three encoding types (Cartesian, XOR, MDR) for an interaction on a single CV fold.
    Performs phantom epistasis check by fitting main effects first, then interaction effects on residuals.

    Parameters:
        X1_ray_id, X2_ray_id: Ray ObjectIDs for genotype vectors
        y_ray_id: Ray ObjectID for phenotype vector
        train_idx: Training indices for this fold
        valid_idx: Validation indices for this fold
        snp1, snp2: SNP identifiers

    Returns:
        results (Dict[str, float32_t]): R² scores for each encoding {'cartesian': r2, 'xor': r2, 'mdr': r2}
            Failed encodings have R² = -1.0
        mdr_mapping (Dict): MDR feature_map if MDR succeeded, else None
    """
    # Resolve Ray ObjectIDs
    # print(X1_ray_id)
    # X1 = ray.get(X1_ray_id)
    # X2 = ray.get(X2_ray_id)
    # y = ray.get(y_ray_id)

    results = {'cartesian': float32_t(-1.0), 'xor': float32_t(-1.0), 'mdr': float32_t(-1.0)}
    mdr_mapping = None

    # Step 1: Fit base model for phantom epistasis check (main effects only)
    try:
        X1_train_centered = X1[train_idx] - np.mean(X1[train_idx])
        X2_train_centered = X2[train_idx] - np.mean(X2[train_idx])
        y_train_centered = y[train_idx] - np.mean(y[train_idx])

        base_model = sm.OLS(y_train_centered, sm.add_constant(np.column_stack((X1_train_centered, X2_train_centered)), has_constant='add'))
        base_results = base_model.fit()

        # Get residuals on training data
        y_base_train_pred = base_results.predict(sm.add_constant(np.column_stack((X1_train_centered, X2_train_centered)), has_constant='add'))
        y_train_residuals = y_train_centered - y_base_train_pred

        # Get residuals on validation data
        X1_valid_centered = X1[valid_idx] - np.mean(X1[train_idx])
        X2_valid_centered = X2[valid_idx] - np.mean(X2[train_idx])
        y_valid_centered = y[valid_idx] - np.mean(y[train_idx])
        y_base_valid_pred = base_results.predict(sm.add_constant(np.column_stack((X1_valid_centered, X2_valid_centered)), has_constant='add'))
        y_valid_residuals = y_valid_centered - y_base_valid_pred
    except Exception as e:
        logging.error(f"Error fitting base model for phantom epistasis check for SNPs {snp1}, {snp2}: {e}")
        return results, mdr_mapping

    # Step 2: Evaluate Cartesian encoding
    try:
        X_encoded = encode_cartesian(X1, X2)
        X_encoded_train_centered = X_encoded[train_idx] - np.mean(X_encoded[train_idx])
        X_encoded_valid_centered = X_encoded[valid_idx] - np.mean(X_encoded[train_idx])

        regressor = sm.OLS(y_train_residuals, sm.add_constant(X_encoded_train_centered, has_constant='add'))
        fit_results = regressor.fit()
        y_pred = fit_results.predict(sm.add_constant(X_encoded_valid_centered, has_constant='add'))
        results['cartesian'] = float32_t(r2_score(y_valid_residuals, y_pred))
    except Exception as e:
        logging.error(f"Error evaluating cartesian for SNP pair {snp1}, {snp2}: {e}")

    # Step 3: Evaluate XOR encoding
    try:
        X_encoded = encode_xor(X1, X2)
        X_encoded_train_centered = X_encoded[train_idx] - np.mean(X_encoded[train_idx])
        X_encoded_valid_centered = X_encoded[valid_idx] - np.mean(X_encoded[train_idx])

        regressor = sm.OLS(y_train_residuals, sm.add_constant(X_encoded_train_centered, has_constant='add'))
        fit_results = regressor.fit()
        y_pred = fit_results.predict(sm.add_constant(X_encoded_valid_centered, has_constant='add'))
        results['xor'] = float32_t(r2_score(y_valid_residuals, y_pred))
    except Exception as e:
        logging.error(f"Error evaluating xor for SNP pair {snp1}, {snp2}: {e}")

    # Step 4: Evaluate MDR encoding
    try:
        X_encoded_train, mdr_fitted_object, temp_mdr_mapping = encode_mdr(X1[train_idx], X2[train_idx], y[train_idx])
        X_encoded_valid = mdr_fitted_object.transform(np.column_stack((X1[valid_idx], X2[valid_idx])))

        X_encoded_train_centered = X_encoded_train - np.mean(X_encoded_train)
        X_encoded_valid_centered = X_encoded_valid - np.mean(X_encoded_train)

        regressor = sm.OLS(y_train_residuals, sm.add_constant(X_encoded_train_centered, has_constant='add'))
        fit_results = regressor.fit()
        y_pred = fit_results.predict(sm.add_constant(X_encoded_valid_centered, has_constant='add'))
        results['mdr'] = float32_t(r2_score(y_valid_residuals, y_pred))
        mdr_mapping = temp_mdr_mapping
    except Exception as e:
        logging.error(f"Error evaluating MDR for SNP pair {snp1}, {snp2}: {e}")

    return results, mdr_mapping


# ray remote function to evaluate only cartesian encoding for a single CV fold (ablation study)
@ray.remote
def ray_evaluate_interaction_cartesian_fold(X1_ray_id: ray.ObjectID, X2_ray_id: ray.ObjectID, y_ray_id: ray.ObjectID,
                                           train_idx: npt.NDArray, valid_idx: npt.NDArray,
                                           snp1: snp_t, snp2: snp_t) -> float32_t:
    """
    Evaluate only Cartesian encoding for an interaction on a single CV fold.
    Used for ablation studies. Does NOT perform phantom epistasis check.

    Parameters:
        X1_ray_id, X2_ray_id: Ray ObjectIDs for genotype vectors
        y_ray_id: Ray ObjectID for phenotype vector
        train_idx: Training indices for this fold
        valid_idx: Validation indices for this fold
        snp1, snp2: SNP identifiers

    Returns:
        cartesian_r2 (float32_t): R² score for cartesian encoding (-1.0 if failed)
    """
    # Resolve Ray ObjectIDs
    X1 = ray.get(X1_ray_id)
    X2 = ray.get(X2_ray_id)
    y = ray.get(y_ray_id)

    # Step 1: Fit base model for phantom epistasis check (main effects only)
    try:
        X1_train_centered = X1[train_idx] - np.mean(X1[train_idx])
        X2_train_centered = X2[train_idx] - np.mean(X2[train_idx])
        y_train_centered = y[train_idx] - np.mean(y[train_idx])

        base_model = sm.OLS(y_train_centered, sm.add_constant(np.column_stack((X1_train_centered, X2_train_centered)), has_constant='add'))
        base_results = base_model.fit()

        # Get residuals on training data
        y_base_train_pred = base_results.predict(sm.add_constant(np.column_stack((X1_train_centered, X2_train_centered)), has_constant='add'))
        y_train_residuals = y_train_centered - y_base_train_pred

        # Get residuals on validation data
        X1_valid_centered = X1[valid_idx] - np.mean(X1[train_idx])
        X2_valid_centered = X2[valid_idx] - np.mean(X2[train_idx])
        y_valid_centered = y[valid_idx] - np.mean(y[train_idx])
        y_base_valid_pred = base_results.predict(sm.add_constant(np.column_stack((X1_valid_centered, X2_valid_centered)), has_constant='add'))
        y_valid_residuals = y_valid_centered - y_base_valid_pred
    except Exception as e:
        logging.error(f"Error fitting base model for phantom epistasis check for SNPs {snp1}, {snp2}: {e}")
        return float32_t(-1.0)

    try:
        X_encoded = encode_cartesian(X1, X2)
        X_encoded_train_centered = X_encoded[train_idx] - np.mean(X_encoded[train_idx])
        X_encoded_valid_centered = X_encoded[valid_idx] - np.mean(X_encoded[train_idx])

        regressor = sm.OLS(y_train_residuals, sm.add_constant(X_encoded_train_centered, has_constant='add'))
        fit_results = regressor.fit()
        y_pred = fit_results.predict(sm.add_constant(X_encoded_valid_centered, has_constant='add'))
        return float32_t(r2_score(y_valid_residuals, y_pred))
    except Exception as e:
        logging.error(f"Error evaluating cartesian for SNP pair {snp1}, {snp2}: {e}")
        return float32_t(-1.0)


# ray remote function to apply all the preprocessing steps when evaluating an interaction
@ray.remote
def ray_preprocess_interaction(X1, X2, y, train_idx, valid_idx, full_train_idx, snp1, snp2):
    """
    Preprocess interaction features for a pair of SNPs by evaluating all encodings and selecting the best one.
    This function combines evaluation and encoding steps to minimize Ray scheduling overhead.
    All encoding logic is inlined to avoid nested Ray calls.

    The following steps are performed:
    1. Missing multi-locus genotype: Check in both train and validation sets. If missing, the interaction is not evaluated.
    2. Check Pearson's correlation if the SNPs are in the same chromosome and do not evaluate if the correlation is above 0.50.
    3. Evaluate all 3 encoding types (Cartesian, XOR, MDR) inline with phantom epistasis check.

    Parameters:
        X1, X2: Genotype vectors for the two SNPs
        y: Phenotype vector
        train_idx: Training indices
        valid_idx: Validation indices
        full_train_idx: Full training indices used for MLG check and Pearson's correlation
        snp1, snp2: SNP identifiers for the two SNPs
    Returns:
       best_r2: Best R² score among the encodings (-1.0 if all failed or not evaluated)
       interaction name: snp1_bestlo_snp2 (e.g. "chr1.12345_cartesian_chr.67890")
       failure_code: -1.0 if MLG missing, -2.0 if high correlation, -3.0 if all encodings failed, 1.0 if success
       best_enc: Encoding type that achieved the best R² score (None if all failed or not evaluated)
       correlation_r2: Pearson's R² between the two SNPs (-1.0 if not computed)
       mdr_mapping: MDR feature_map if MDR was best encoding, else None
    """

    # Initialize the results variables
    best_r2 = float32_t(-1.0)
    best_enc = None
    failure_code = float32_t(1.0)  # assume success unless we hit a failure condition
    correlation_r2 = float32_t(-1.0)  # default value for correlation
    mdr_mapping = None

    # Assert that the inputs are numpy arrays
    assert isinstance(X1, np.ndarray), "X1 should be a numpy array"
    assert isinstance(X2, np.ndarray), "X2 should be a numpy array"

    # Step 1: Check for missing multi-locus genotype (MLG) in only training set
    # Create all possible genotype combinations for the two SNPs (0.0, 0.5, 1.0)
    all_combinations = {(g1, g2) for g1 in [0.0, 0.5, 1.0] for g2 in [0.0, 0.5, 1.0]}

    # Check if any combination is missing in the training set
    full_train_combinations = set(zip(X1[full_train_idx], X2[full_train_idx])) # checks for genotype combinations of X1 and X2 in the training set
    missing_train = not all_combinations.issubset(full_train_combinations)
    if missing_train:
        failure_code = float32_t(-1.0)  # MLG missing
        interaction_name = f"{snp1}_none_{snp2}"
        return best_r2, interaction_name, failure_code, best_enc, correlation_r2, mdr_mapping

    # Step 2: Compute Pearson's correlation if SNPs are in the same chromosome
    snp1_chr = snp1.split('_')[0]  # SNP name format is "chr_pos"
    snp2_chr = snp2.split('_')[0]

    # If chromosomes are the same, check for high correlation and skip if above threshold
    if snp1_chr == snp2_chr:
        correlation_coef = float32_t(np.corrcoef(X1[full_train_idx], X2[full_train_idx])[0, 1])
        correlation_r2 = float32_t(correlation_coef ** 2)
        if correlation_r2 > 0.50:
            failure_code = float32_t(-2.0)  # high correlation
            interaction_name = f"{snp1}_none_{snp2}"
            return best_r2, interaction_name, failure_code, best_enc, correlation_r2, mdr_mapping

    # Step 3: Best encoder and phantom epistasis check
    # Step 3a:Fit base model for phantom epistasis check

    # Build and fit the base model (main effects only)
    try:
        # Center the training data
        X1_train_centered = X1[train_idx] - np.mean(X1[train_idx])
        X2_train_centered = X2[train_idx] - np.mean(X2[train_idx])
        y_train_centered = y[train_idx] - np.mean(y[train_idx])
        base_model = sm.OLS(y_train_centered, sm.add_constant(np.column_stack((X1_train_centered, X2_train_centered)), has_constant='add'))
        base_results = base_model.fit()

        # Get residuals on training data
        y_base_train_pred = base_results.predict(sm.add_constant(np.column_stack((X1_train_centered, X2_train_centered)), has_constant='add'))
        y_train_residuals = y_train_centered - y_base_train_pred

        # Get residuals on validation data
        X1_valid_centered = X1[valid_idx] - np.mean(X1[train_idx])
        X2_valid_centered = X2[valid_idx] - np.mean(X2[train_idx])
        y_valid_centered = y[valid_idx] - np.mean(y[train_idx])
        y_base_valid_pred = base_results.predict(sm.add_constant(np.column_stack((X1_valid_centered, X2_valid_centered)), has_constant='add'))
        y_valid_residuals = y_valid_centered - y_base_valid_pred
    except Exception as e:
        logging.error(f"Error fitting base model for phantom epistasis check for SNPs {snp1}, {snp2}: {e}")
        failure_code = float32_t(-3.0)
        interaction_name = f"{snp1}_none_{snp2}"
        return best_r2, interaction_name, failure_code, best_enc, correlation_r2, mdr_mapping

    # Step 3b: Evaluate all 3 encodings inline (no nested ray calls)
    temp_mdr_mapping = None
    # make a dictionary to hold the r2 scores for each encoding to compare at the end and select the best one. This is needed to avoid code repetition and also to handle the case where all encodings fail.
    results = {}

    # Cartesian encoding
    try:
        X_encoded = encode_cartesian(X1, X2)

        # Center the encoded interaction based on training data
        X_encoded_train_centered = X_encoded[train_idx] - np.mean(X_encoded[train_idx])
        X_encoded_valid_centered = X_encoded[valid_idx] - np.mean(X_encoded[train_idx])

        # Fit OLS model on training residuals with centered interaction
        regressor = sm.OLS(y_train_residuals, sm.add_constant(X_encoded_train_centered, has_constant='add'))
        fit_results = regressor.fit()

        # Score on validation residuals with centered interaction
        y_pred = fit_results.predict(sm.add_constant(X_encoded_valid_centered, has_constant='add'))
        interaction_r2 = r2_score(y_valid_residuals, y_pred)
        results['cartesian'] = (float32_t(interaction_r2), interaction_t(snp1, snp2), snp_t('cartesian'), failure_code, temp_mdr_mapping)
    except Exception as e:
        logging.error(f"Error evaluating cartesian for SNP pair {snp1}, {snp2}: {e}")

    # XOR encoding
    try:
        X_encoded = encode_xor(X1, X2)

        # Center the encoded interaction based on training data
        X_encoded_train_centered = X_encoded[train_idx] - np.mean(X_encoded[train_idx])
        X_encoded_valid_centered = X_encoded[valid_idx] - np.mean(X_encoded[train_idx])

        # Fit OLS model on training residuals with centered interaction
        regressor = sm.OLS(y_train_residuals, sm.add_constant(X_encoded_train_centered, has_constant='add'))
        fit_results = regressor.fit()

        # Score on validation residuals with centered interaction
        y_pred = fit_results.predict(sm.add_constant(X_encoded_valid_centered, has_constant='add'))
        interaction_r2 = r2_score(y_valid_residuals, y_pred)
        results['xor'] = (float32_t(interaction_r2), interaction_t(snp1, snp2), snp_t('xor'), failure_code, temp_mdr_mapping)

    except Exception as e:
        logging.error(f"Error evaluating xor for SNP pair {snp1}, {snp2}: {e}")

    # MDR encoding
    try:
        # Build MDR mapping from training data only
        X_encoded_train, mdr_fitted_object, mdr_mapping = encode_mdr(X1[train_idx], X2[train_idx], y[train_idx])
        temp_mdr_mapping = mdr_mapping

        # Encode the validation set using the fitted MDR object
        X_encoded_valid = mdr_fitted_object.transform(np.column_stack((X1[valid_idx], X2[valid_idx])))

        # Center the encoded interaction based on training data
        X_encoded_train_centered = X_encoded_train - np.mean(X_encoded_train)
        X_encoded_valid_centered = X_encoded_valid - np.mean(X_encoded_train_centered)

        # Fit OLS model on training residuals with centered interaction
        regressor = sm.OLS(y_train_residuals, sm.add_constant(X_encoded_train_centered, has_constant='add'))
        fit_results = regressor.fit()

        # Score on validation residuals with centered interaction
        y_pred = fit_results.predict(sm.add_constant(X_encoded_valid_centered, has_constant='add'))
        interaction_r2 = r2_score(y_valid_residuals, y_pred)
        results['mdr'] = (float32_t(interaction_r2), interaction_t(snp1, snp2), snp_t('mdr'), failure_code, temp_mdr_mapping)

    except Exception as e:
        logging.error(f"Error evaluating MDR for SNP pair {snp1}, {snp2}: {e}")


    return results

# ray remote function to apply all the preprocessing steps and only evaluating cartesian encoding. This is required for ablation study
@ray.remote
def ray_preprocess_interaction_cartesian(X1, X2, y, train_idx, valid_idx, full_train_idx, snp1, snp2):
    """Same as ray_preprocess_interaction but only evaluates cartesian encoding for ablation study.
    This function is used to evaluate the impact of only using cartesian encoding.
    """

    # Initialize the results variables
    best_r2 = float32_t(-1.0)
    best_enc = None
    failure_code = float32_t(1.0)  # assume success unless we hit a failure condition
    correlation_r2 = float32_t(-1.0)  # default value for correlation
    mdr_mapping = None

    # Assert that the inputs are numpy arrays
    assert isinstance(X1, np.ndarray), "X1 should be a numpy array"
    assert isinstance(X2, np.ndarray), "X2 should be a numpy array"

    # Step 1: Check for missing multi-locus genotype (MLG) in only training set
    # Create all possible genotype combinations for the two SNPs (0.0, 0.5, 1.0)
    all_combinations = {(g1, g2) for g1 in [0.0, 0.5, 1.0] for g2 in [0.0, 0.5, 1.0]}

    # Check if any combination is missing in the training set
    full_train_combinations = set(zip(X1[full_train_idx], X2[full_train_idx]))
    missing_train = not all_combinations.issubset(full_train_combinations)
    if missing_train:
        failure_code = float32_t(-1.0)  # MLG missing
        interaction_name = f"{snp1}_none_{snp2}"
        return best_r2, interaction_name, failure_code, best_enc, correlation_r2, mdr_mapping

    # Step 2: Compute Pearson's correlation if SNPs are in the same chromosome
    snp1_chr = snp1.split('_')[0]  # SNP name format is "chr_pos"
    snp2_chr = snp2.split('_')[0]

    # If chromosomes are the same, check for high correlation and skip if above threshold
    if snp1_chr == snp2_chr:
        correlation_coef = float32_t(np.corrcoef(X1[full_train_idx], X2[full_train_idx])[0, 1])
        correlation_r2 = float32_t(correlation_coef ** 2)
        if correlation_r2 > 0.50:
            failure_code = float32_t(-2.0)  # high correlation
            interaction_name = f"{snp1}_none_{snp2}"
            return best_r2, interaction_name, failure_code, best_enc, correlation_r2, mdr_mapping
    # Step 3: Only evaluate Cartesian encoding for ablation study
    try:
        X_encoded = encode_cartesian(X1, X2)

        # Center the encoded interaction based on training data
        X_encoded_train_centered = X_encoded[train_idx] - np.mean(X_encoded[train_idx])
        X_encoded_valid_centered = X_encoded[valid_idx] - np.mean(X_encoded[train_idx])

        # Fit OLS model on training residuals with centered interaction
        regressor = sm.OLS(y[train_idx], sm.add_constant(X_encoded_train_centered, has_constant='add'))
        fit_results = regressor.fit()

        # Score on validation set with centered interaction
        y_pred = fit_results.predict(sm.add_constant(X_encoded_valid_centered, has_constant='add'))
        interaction_r2 = r2_score(y[valid_idx], y_pred)

        best_r2 = float32_t(interaction_r2)
        best_enc = snp_t('cartesian')
    except Exception as e:
        logging.error(f"Error evaluating cartesian for SNP pair {snp1}, {snp2}: {e}")
        failure_code = float32_t(-3.0)  # encoding failed

    # Construct final interaction name
    interaction_name = f"{snp1}_{best_enc}_{snp2}" if best_enc is not None else f"{snp1}_none_{snp2}"

    return best_r2, interaction_name, failure_code, best_enc, correlation_r2, mdr_mapping

@ray.remote
def ray_pfi(X, y, train_idx, valid_idx, new_column_names, root_node, random_state, pop_id):
    """
    Compute permutation feature importance (PFI) for a fitted model using validation data.

    Args:
        X (List[ray.ObjectID]): List of Ray ObjectIDs for feature data arrays.
        y (np.ndarray): Phenotype data array.
        train_idx (np.ndarray): Indices for training data.
        valid_idx (np.ndarray): Indices for validation data.
        new_column_names (List[snp_t]): Names of the features corresponding to X.
        root_node (SelectorNode): Fitted model node to evaluate.
        random_state (int): Random state for reproducibility.
        pop_id (uint32_t): Population ID for tracking.
    """


    # create dataset
    X_train = np.column_stack([ray.get(x)[train_idx] for x in X])
    X_train = sm.add_constant(X_train, has_constant='add')
    X_valid = np.column_stack([ray.get(x)[valid_idx] for x in X])
    X_valid = sm.add_constant(X_valid, has_constant='add')

    # to store pfi results
    pfi_results = {}

    fitted_model = root_node.fit(X_train, y[train_idx])
    # get permutation feature importance on the validation set
    pfi = permutation_importance(fitted_model, X_valid, y[valid_idx], n_repeats=100, random_state=random_state, scoring='r2')

    # Note: pfi.importances_mean has length = num_features + 1 (due to constant)
    # Skip the first element (constant) and map the rest to feature names
    for i in range(len(new_column_names)):
        # Add 1 to index to skip the constant column
        pfi_results[new_column_names[i]] = pfi.importances_mean[i + 1]

    return pfi_results, pop_id

@ray.remote
def ray_eval_pipeline_ld_fs(component_map: Dict[snp_t, Dict],
                            y_train: npt.NDArray,
                            train_idx: npt.NDArray,
                            selector_node: SelectorNode,
                            ld_node: SelectorNode,
                            pop_id: uint32_t,
                            interaction_r2_set: Set) -> Tuple[float32_t, int16_t, uint32_t, List[snp_t], Dict[snp_t, Dict]]:
    """
    Evaluate a pipeline with LD and feature selection nodes using Ray.
    Uses component map structure for efficient component SNP access.

    Parameters:
        component_map (Dict[snp_t, Dict]): Dictionary mapping interaction names to their components:
            {interaction_name: {'snp1_name', 'snp2_name', 'snp1_ray_id', 'snp2_ray_id', 'encoded_ray_id'}}
        y_train (npt.NDArray): Phenotype data array.
        train_idx (npt.NDArray): Indices for training data.
        selector_node (SelectorNode): Fitted feature selector node.
        ld_node (SelectorNode): Fitted LD node.
        pop_id (uint32_t): Population ID for tracking.
        interaction_r2_set (Set): Set of tuples (interaction_name, lo_r2) for interactions in the pipeline.

    Returns:
        Tuple containing:
            float32_t: Error value (-1.0 if failure, 1.0 if success).
            int16_t: Feature count after selection.
            uint32_t: Population ID.
            List[snp_t]: List of selected interaction names after LD and feature selection.
            Dict[snp_t, Dict]: Details of interactions after LD node.
    """

    # make dictionary to hold the interaction r2 scores
    interaction_r2_dict = {p[0]: p[1] for p in interaction_r2_set}

    # hold feature counts
    feature_count = 0
    features_final = []

    # Extract interaction names
    interaction_names = list(component_map.keys())

    # Build local component map with actual data (resolve Ray ObjectIDs) for LD node
    local_component_map = {}
    for name in interaction_names:
        local_component_map[name] = {
            'snp1_name': component_map[name]['snp1_name'],
            'snp2_name': component_map[name]['snp2_name'],
            'snp1_data': ray.get(component_map[name]['snp1_ray_id'])[train_idx],
            'snp2_data': ray.get(component_map[name]['snp2_ray_id'])[train_idx],
            'encoded_data': ray.get(component_map[name]['encoded_ray_id'])[train_idx]
        }

    # Fit the LD node using component map
    try:
        ld_node.fit(local_component_map, y_train[train_idx], interaction_r2_dict)
        selected_features_after_ld = ld_node.selected_features_

        # If no features selected, return early
        if selected_features_after_ld is None or len(selected_features_after_ld) == 0:
            logging.warning("No features selected after LD node")
            return float32_t(-1.0), int16_t(0), pop_id, [], ld_node.interaction_details_after_ld

        # Create dataframe with only selected features for feature selector
        interaction_transformed_df = pd.DataFrame({
            name: local_component_map[name]['encoded_data']
            for name in selected_features_after_ld
        })

    except Exception as e:
        logging.error(f"Exception while fitting LD node: {e}")
        return float32_t(-1.0), int16_t(0), pop_id, [], {}

    # adding the selector nodes
    try:
        # get snps from selector node
        selector_node.fit(interaction_transformed_df, y_train[train_idx])
        interaction_transformed_df = selector_node.transform(interaction_transformed_df) # this dataframe goes into regressor
        feature_count = selector_node.get_feature_count() # number of selected features after the selector node
        features_final = (selector_node.get_feature_names(selected_features_after_ld)) # get the names of the features after the selector node by sending the selected features after the LD node

    except Exception as e:
        logging.error(f"Exception while feature selector fits/transforms: {e}")
        return float32_t(-1.0), int16_t(0), pop_id, [], ld_node.snp_details_after_ld

    # need this bc the root node would tell us if nothing was passed to it with the old implementation
    if feature_count == 0:
        return float32_t(-1.0), int16_t(0), pop_id, [], ld_node.snp_details_after_ld

    # if features_final is not a list, convert it to a list
    if not isinstance(features_final, list):
        features_final = features_final.tolist()

    # return features that made it passed ld and fs for this pipeline
    return float32_t(1.0), int16_t(feature_count), pop_id, [snp_t(feature) for feature in features_final], ld_node.snp_details_after_ld

@ray.remote
def ray_eval_pipeline_fs(snp_names: List[snp_t],
                         interaction_train_enc: List[ray.ObjectID],
                         y_train: npt.NDArray,
                         train_idx: npt.NDArray,
                         selector_node: SelectorNode,   # error. feature count. pop_id. details after ld node. snp_after_ld (ignore for this one)
                         pop_id: uint32_t) ->     Tuple[float32_t, int16_t, uint32_t, List[snp_t], Dict[snp_t, Dict]]:
    """
    Evaluate a pipeline with only a feature selection node using Ray.

    Parameters:
        snp_names (List[snp_t]): List of SNP names in the pipeline.
        interaction_train_enc (List[ray.ObjectID]): List of Ray ObjectIDs for encoded interaction data.
        y_train (npt.NDArray): Phenotype data array.
        train_idx (npt.NDArray): Indices for training data.
        selector_node (SelectorNode): Fitted feature selector node.
        pop_id (uint32_t): Population ID for tracking.

    Returns:
        Tuple containing:
            float32_t: Error value (-1.0 if failure, 1.0 if success).
            int16_t: Feature count after selection.
            uint32_t: Population ID.
            List[snp_t]: List of selected SNP names after feature selection.
            Dict[snp_t, Dict]: Empty dictionary (no LD node details).
    """

    # hold feature counts across all folds
    feature_count = 0
    # holds feature list
    features_final = []
    # create both original and encoded dataframes from the ray object ids
    interaction_train_encoded_df = pd.DataFrame({name: ray.get(data_obj)[train_idx].tolist() for name, data_obj in zip(snp_names, interaction_train_enc)})

    # adding the selector and regressor nodes
    try:
        # get snps from selector node
        selector_node.fit(interaction_train_encoded_df, y_train[train_idx])
        interaction_train_encoded_df = selector_node.transform(interaction_train_encoded_df) # this dataframe goes into regressor
        feature_count = selector_node.get_feature_count() # number of selected features after the selector node
        features_final = (selector_node.get_feature_names(snp_names)) # get the names of the features after the selector node by sending the selected features after the LD node

    except Exception as e:
        logging.error(f"Exception while feature selector fits/transforms: {e}")
        return float32_t(-1.0), int16_t(0), pop_id, [], {}

    # need this bc the root node would tell us if nothing was passed to it with the old implementation
    if feature_count == 0:
        return float32_t(-1.0), int16_t(0), pop_id, [], {}

    # if features_final is not a list, convert it to a list
    if not isinstance(features_final, list):
        features_final = features_final.tolist()

    # return features that made it passed ld and fs for this pipeline
    return float32_t(1.0), int16_t(feature_count), pop_id, [snp_t(feature) for feature in features_final], {}

@ray.remote
def ray_eval_pipeline_r2(component_map: Dict[snp_t, Dict],
                      y: npt.NDArray,
                      train_idx: npt.NDArray,
                      valid_idx: npt.NDArray,
                      pop_id: uint32_t) -> Tuple[float32_t, uint32_t, float32_t]:
    """
    Evaluate a pipeline with only a regression node using Ray.

    Parameters:
        X_interaction (List[ray.ObjectID]): List of Ray ObjectIDs for encoded interaction data.
        X_univariate (List[ray.ObjectID]): List of Ray ObjectIDs for all the univariate features that make up the interactions in the pipeline (used for phantom epistasis check).
        y (ray.ObjectID): Ray ObjectID for phenotype data array.
        train_idx (np.ndarray): Indices for training data.
        valid_idx (np.ndarray): Indices for validation data.
        pop_id (uint32_t): Population ID for tracking.

    Returns:
        Tuple containing:
            float32_t: R² score on validation data.
            uint32_t: Population ID.
            float32_t: Error value (1.0 if success, -1.0 if failure).
    """

    # extract the interaction and univariate features from the component map
    interaction_names = list(component_map.keys())
    X_interaction = [component_map[name]['encoded_ray_id'] for name in interaction_names]
    X_univariate = []
    for name in interaction_names:
        X_univariate.append(component_map[name]['snp1_ray_id'])
        X_univariate.append(component_map[name]['snp2_ray_id'])

    # create dataset
    X_interaction_matrix = np.column_stack([ray.get(x) for x in X_interaction])
    X_univariate_matrix = np.column_stack([ray.get(x) for x in X_univariate])

    # remove any duplicate columns from the univariate matrix (can happen if the same SNP is involved in multiple interactions in the pipeline)
    _, unique_indices = np.unique(X_univariate_matrix, axis=1, return_index=True)
    X_univariate_matrix = X_univariate_matrix[:, np.sort(unique_indices)]

    # center all features based on training data for phantom epistasis check
    X_interaction_matrix_train_centered = X_interaction_matrix[train_idx] - np.mean(X_interaction_matrix[train_idx], axis=0)
    X_univariate_matrix_train_centered = X_univariate_matrix[train_idx] - np.mean(X_univariate_matrix[train_idx], axis=0)
    X_interaction_matrix_valid_centered = X_interaction_matrix[valid_idx] - np.mean(X_interaction_matrix[train_idx], axis=0)
    X_univariate_matrix_valid_centered = X_univariate_matrix[valid_idx] - np.mean(X_univariate_matrix[train_idx], axis=0)
    y_train_centered = ray.get(y)[train_idx] - np.mean(ray.get(y)[train_idx])
    y_valid_centered = ray.get(y)[valid_idx] - np.mean(ray.get(y)[train_idx])

    # Step 1: Fit a ridge regression model on the univariate features to get residuals for phantom epistasis check
    try:
        regressor = sm.OLS(y_train_centered, sm.add_constant(X_univariate_matrix_train_centered, has_constant='add'))
        results = regressor.fit_regularized(L1_wt=0.0, alpha=np.float32(1e-4))
        y_train_residuals = y_train_centered - results.predict(sm.add_constant(X_univariate_matrix_train_centered, has_constant='add'))
        y_valid_residuals = y_valid_centered - results.predict(sm.add_constant(X_univariate_matrix_valid_centered, has_constant='add')) # use the training centered univariate features to get the predictions for the validation set to compute the residuals for phantom epistasis check
    except Exception as e:
        logging.error(f"Exception while fitting the base model ridge regression: {e}")
        return float32_t(-1.0), pop_id, float32_t(-1.0)

    # Step 2: Fit OLS model on the interaction features using the residuals from the ridge regression and score on validation set to get the R² for the interaction while controlling for main effects (phantom epistasis check)
    try:
        regressor = sm.OLS(y_train_residuals, sm.add_constant(X_interaction_matrix_train_centered, has_constant='add'))
        fit_results = regressor.fit()
        y_pred = fit_results.predict(sm.add_constant(X_interaction_matrix_valid_centered, has_constant='add'))
        r2_score_value = r2_score(y_valid_residuals, y_pred)
    except Exception as e:
        logging.error(f"Error while scoring the pipeline: {e}")
        return float32_t(-1.0), pop_id, float32_t(-1.0)

    return float32_t(r2_score_value), pop_id, float32_t(1.0)

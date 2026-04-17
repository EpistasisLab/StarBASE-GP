from ..Base.selectors import SelectorNode
from ..Base.types import (float32_t, int16_t, snp_t, uint32_t)
from ..Base.utils import snp_chrm_pos

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


# Function for Cartesian encoding for interactions
@numba.njit(cache=True)
def encode_cartesian(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
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
def encode_xor(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
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

# Function for MDR encoding using pre-computed mapping
def encode_mdr(X1: np.ndarray, X2: np.ndarray, y: np.ndarray):
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

# This function will be used when encoding the validation/test set not during initial evaluation of the interaction.
@ray.remote
def ray_interaction_encoder(X1: np.ndarray,
                            X2: np.ndarray,
                            y: np.ndarray,
                            train_idx: npt.NDArray,
                            enc: snp_t,
                            snp: snp_t) -> Tuple[np.ndarray, snp_t]:
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

    # Get the encoding string
    enc_str = str(enc) if isinstance(enc, np.str_) else enc

    if enc_str == 'cartesian':
        X_encoded = encode_cartesian(X1, X2)
        return X_encoded, snp

    elif enc_str == 'xor':
        X_encoded = encode_xor(X1, X2)
        return X_encoded, snp

    elif enc_str == 'mdr':
        mdr_fitted_object = encode_mdr(X1[train_idx], X2[train_idx], y[train_idx])[1]  # get the fitted MDR object to use for transform
        X_encoded = mdr_fitted_object.transform(np.column_stack((X1, X2))) # mdr package has built in transform function to apply the mapping to the full dataset
        return X_encoded, snp

    else:
        raise ValueError(f"Unknown encoding type: {enc_str}")

# ray remote function to pre-screen an interaction (MLG + Pearson correlation check)
@ray.remote
def ray_prescreen_interaction(X1: np.ndarray,
                              X2: np.ndarray,
                              full_train_idx: npt.NDArray,
                              snp1: snp_t, snp2: snp_t) -> Tuple[bool, float32_t, float32_t]:
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
    snp1_chr = snp_chrm_pos(snp1)[0]
    snp2_chr = snp_chrm_pos(snp2)[0]

    if snp1_chr == snp2_chr:
        correlation_coef = float32_t(np.corrcoef(X1[full_train_idx], X2[full_train_idx])[0, 1])
        correlation_r2 = float32_t(correlation_coef ** 2)
        if correlation_r2 > 0.50:
            return False, float32_t(-2.0), correlation_r2

    # Passed both checks
    return True, float32_t(1.0), correlation_r2

# ray remote function to evaluate all encodings for a single CV fold
@ray.remote
def ray_evaluate_interaction_encodings(X1: np.ndarray,
                                       X2: np.ndarray,
                                       y: np.ndarray,
                                       train_idx: npt.NDArray,
                                       valid_idx: npt.NDArray,
                                       snp1: snp_t, snp2: snp_t) -> Tuple[Dict[str, float32_t], Dict | None, Dict | None]:
    """
    Evaluate all three encoding types (Cartesian, XOR, MDR) for an interaction on a single CV fold.
    Performs phantom epistasis check by fitting main effects first, then interaction effects on residuals.

    Parameters:
        X1_ray_id, X2_ray_id: Ray ObjectIDs for genotype vectors for the constituent SNPs
        y_ray_id: Ray ObjectID for phenotype vector
        train_idx: Training indices for this fold
        valid_idx: Validation indices for this fold
        snp1, snp2: SNP identifiers

    Returns:
        results (Dict[str, float32_t]): R² scores for each encoding {'cartesian': r2, 'xor': r2, 'mdr': r2}
            Failed encodings have R² = -1.0
        mdr_mapping (Dict): MDR feature_map if MDR succeeded, else None
    """
    results = {'cartesian': float32_t(-1.0), 'xor': float32_t(-1.0), 'mdr': float32_t(-1.0)}
    error = {'cartesian': False, 'xor': False, 'mdr': False, 'base_model': False}
    mdr_mapping = None

    # Step 1: Modified Calculation: Calculate base_r2 on the extact phenotype instead of extracting residuals
    try:
        # Center the training data for the main effects model
        X1_train_centered = X1[train_idx] - np.mean(X1[train_idx])
        X2_train_centered = X2[train_idx] - np.mean(X2[train_idx])
        y_train_centered = y[train_idx] - np.mean(y[train_idx])

        # training model with main effects (univariate) SNPs
        X_base_train = np.column_stack((X1_train_centered, X2_train_centered))
        base_model = sm.OLS(y_train_centered, sm.add_constant(X_base_train, has_constant='add'))
        base_results = base_model.fit()

        # Center the validation data using the training means 
        X1_valid_centered = X1[valid_idx] - np.mean(X1[train_idx])
        X2_valid_centered = X2[valid_idx] - np.mean(X2[train_idx])
        y_valid_centered = y[valid_idx] - np.mean(y[train_idx])
        
        # get the predictions on the validation set from the trained main effects model
        X_base_valid = np.column_stack((X1_valid_centered, X2_valid_centered))
        y_base_valid_pred = base_results.predict(sm.add_constant(X_base_valid, has_constant='add'))
        
        # compute the validation r2 of the main effects model to use as the baseline for phantom epistasis check
        base_r2 = r2_score(y_valid_centered, y_base_valid_pred)
    except Exception as e:
        logging.error(f"Error fitting base model for phantom epistasis check for SNPs {snp1}, {snp2}: {e}")
        error['base_model'] = True
        return results, mdr_mapping, error

    # Step 2: Evaluate Cartesian encoding - Modified: Fit joint model (main + cartesian) on y_train_centered, score on y_valid_centered and subtract the base_r2 to get the incremental r2 contributed by the interaction while controlling for main effects (phantom epistasis check)
    try:
        X_encoded = encode_cartesian(X1, X2)
        X_encoded_train_centered = X_encoded[train_idx] - np.mean(X_encoded[train_idx])
        X_encoded_valid_centered = X_encoded[valid_idx] - np.mean(X_encoded[train_idx])

        X_joint_train = np.column_stack((X_base_train, X_encoded_train_centered))
        X_joint_valid = np.column_stack((X_base_valid, X_encoded_valid_centered))

        regressor = sm.OLS(y_train_centered, sm.add_constant(X_joint_train, has_constant='add'))
        fit_results = regressor.fit()
        y_pred = fit_results.predict(sm.add_constant(X_joint_valid, has_constant='add'))

        joint_r2 = r2_score(y_valid_centered, y_pred)
        results['cartesian'] = float32_t(joint_r2 - base_r2)
    except Exception as e:
        logging.error(f"Error evaluating cartesian for SNP pair {snp1}, {snp2}: {e}")
        error['cartesian'] = True

    # Step 3: Evaluate XOR encoding: Modified: Fit joint model (main + xor) on y_train_centered, score on y_valid_centered and subtract the base_r2 to get the incremental r2 contributed by the interaction while controlling for main effects (phantom epistasis check)
    try:
        X_encoded = encode_xor(X1, X2)
        X_encoded_train_centered = X_encoded[train_idx] - np.mean(X_encoded[train_idx])
        X_encoded_valid_centered = X_encoded[valid_idx] - np.mean(X_encoded[train_idx])

        X_joint_train = np.column_stack((X_base_train, X_encoded_train_centered))
        X_joint_valid = np.column_stack((X_base_valid, X_encoded_valid_centered))
        regressor = sm.OLS(y_train_centered, sm.add_constant(X_joint_train, has_constant='add'))
        fit_results = regressor.fit()
        y_pred = fit_results.predict(sm.add_constant(X_joint_valid, has_constant='add'))
        joint_r2 = r2_score(y_valid_centered, y_pred)
        results['xor'] = float32_t(joint_r2 - base_r2)
    except Exception as e:
        logging.error(f"Error evaluating xor for SNP pair {snp1}, {snp2}: {e}")
        error['xor'] = True

    # Step 4: Evaluate MDR encoding: Modified: Fit joint model (main + mdr) on y_train_centered, score on y_valid_centered and subtract the base_r2 to get the incremental r2 contributed by the interaction while controlling for main effects (phantom epistasis check). Also return the MDR mapping if successful for encoding the validation/test set in downstream analyses.
    try:
        X_encoded_train, mdr_fitted_object, temp_mdr_mapping = encode_mdr(X1[train_idx], X2[train_idx], y[train_idx])
        X_encoded_valid = mdr_fitted_object.transform(np.column_stack((X1[valid_idx], X2[valid_idx])))

        X_encoded_train_centered = X_encoded_train - np.mean(X_encoded_train)
        X_encoded_valid_centered = X_encoded_valid - np.mean(X_encoded_train)

        X_joint_train = np.column_stack((X_base_train, X_encoded_train_centered))
        X_joint_valid = np.column_stack((X_base_valid, X_encoded_valid_centered))
        regressor = sm.OLS(y_train_centered, sm.add_constant(X_joint_train, has_constant='add'))
        fit_results = regressor.fit()
        y_pred = fit_results.predict(sm.add_constant(X_joint_valid, has_constant='add'))
        joint_r2 = r2_score(y_valid_centered, y_pred)
        results['mdr'] = float32_t(joint_r2 - base_r2)
        mdr_mapping = temp_mdr_mapping
    except Exception as e:
        logging.error(f"Error evaluating MDR for SNP pair {snp1}, {snp2}: {e}")
        error['mdr'] = True

    return results, mdr_mapping, error

# ray remote function to evaluate only cartesian encoding for a single CV fold (ablation study)
@ray.remote
def ray_evaluate_interaction_cartesian(X1: np.ndarray,
                                       X2: np.ndarray,
                                       y: np.ndarray,
                                       train_idx: npt.NDArray,
                                       valid_idx: npt.NDArray,
                                       snp1: snp_t, snp2: snp_t) -> Tuple[float32_t, bool]:
    """
    Evaluate only Cartesian encoding for an interaction on a single CV fold.
    Used for ablation studies. Does NOT perform phantom epistasis check.

    Parameters:
        X1, X2: Ray ObjectIDs for genotype vectors
        y: Ray ObjectID for phenotype vector
        train_idx: Training indices for this fold
        valid_idx: Validation indices for this fold
        snp1, snp2: SNP identifiers

    Returns:
        cartesian_r2 (float32_t): R² score for cartesian encoding (-1.0 if failed)
    """

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
        return float32_t(-1.0), True

    try:
        X_encoded = encode_cartesian(X1, X2)
        X_encoded_train_centered = X_encoded[train_idx] - np.mean(X_encoded[train_idx])
        X_encoded_valid_centered = X_encoded[valid_idx] - np.mean(X_encoded[train_idx])

        regressor = sm.OLS(y_train_residuals, sm.add_constant(X_encoded_train_centered, has_constant='add'))
        fit_results = regressor.fit()
        y_pred = fit_results.predict(sm.add_constant(X_encoded_valid_centered, has_constant='add'))
        return float32_t(r2_score(y_valid_residuals, y_pred)), False
    except Exception as e:
        logging.error(f"Error evaluating cartesian for SNP pair {snp1}, {snp2}: {e}")
        return float32_t(-1.0), True

@ray.remote
def ray_pfi(X: List[ray.ObjectID],
            y: np.ndarray,
            train_idx: npt.NDArray,
            valid_idx: npt.NDArray,
            new_column_names: List[snp_t],
            root_node: SelectorNode,
            random_state: int,
            pop_id: uint32_t) -> Tuple[Dict[snp_t, float32_t], uint32_t]:
    """
    Compute permutation feature importance (PFI) for a fitted model using validation data.

    Args:
        component_map (Dict[snp_t, Dict]): Dictionary mapping interaction names to their components:
            {interaction_name: {'snp1_name', 'snp2_name', 'snp1_ray_id', 'snp2_ray_id', 'encoded_ray_id'}}
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
                            interaction_r2_set: Set) -> Tuple[float32_t, int16_t, uint32_t, List, Dict]:
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

    # 1. HELPER: Ensure keys are standard Python types (Strings or Tuples of Strings)
    def sanitize_key(k):
        if isinstance(k, tuple):
            return tuple(str(x.item()) if hasattr(x, 'item') else str(x) for x in k)
        return str(k.item()) if hasattr(k, 'item') else str(k)

    # 2. SANITIZE INPUTS IMMEDIATELY
    # This prevents 'unhashable type' errors when creating dictionaries
    clean_component_map = {sanitize_key(k): v for k, v in component_map.items()}
    interaction_r2_dict = {sanitize_key(p[0]): p[1] for p in interaction_r2_set}

    feature_count = 0
    features_final = []
    interaction_names = list(clean_component_map.keys())

    # 3. BUILD LOCAL MAP (Resolving Ray ObjectIDs)
    local_component_map = {}
    try:
        for name in interaction_names:
            # Note: train_idx is used to slice the data immediately to save memory
            local_component_map[name] = {
                'snp1_name': clean_component_map[name]['snp1_name'],
                'snp2_name': clean_component_map[name]['snp2_name'],
                'snp1_data': ray.get(clean_component_map[name]['snp1_ray_id'])[train_idx].ravel(),
                'snp2_data': ray.get(clean_component_map[name]['snp2_ray_id'])[train_idx].ravel(),
                'encoded_data': ray.get(clean_component_map[name]['encoded_ray_id'])[train_idx].ravel()
            }
    except Exception as e:
        logging.error(f"Error resolving Ray objects: {e}")
        return float32_t(-1.0), int16_t(0), pop_id, [], {}

    # 4. FIT LD NODE
    try:
        # Pass the sanitized dictionary and r2_dict
        ld_node.fit(local_component_map, y_train[train_idx], interaction_r2_dict)
        selected_features_after_ld = ld_node.selected_features_


        if selected_features_after_ld is None or len(selected_features_after_ld) == 0:
            logging.warning("No features selected after LD node")
            return float32_t(-1.0), int16_t(0), pop_id, [], getattr(ld_node, 'interaction_details_after_ld', {})

        # Create dataframe with only selected features
        interaction_transformed_df = pd.DataFrame({
            name: local_component_map[name]['encoded_data']
            for name in selected_features_after_ld
        })

    except Exception as e:
        # Logging the specific error helps identify if hashing is still an issue
        logging.error(f"Exception while fitting LD node: {type(e).__name__}: {e}")
        return float32_t(-1.0), int16_t(0), pop_id, [], {}

    # 5. FIT SELECTOR NODE
    try:
        selector_node.fit(interaction_transformed_df, y_train[train_idx])
        interaction_transformed_df = selector_node.transform(interaction_transformed_df)
        feature_count = selector_node.get_feature_count()

        # Get final names and ensure they are tuples of snp_t (interaction_t)
        raw_features = selector_node.get_feature_names(selected_features_after_ld)
        if isinstance(raw_features, list):
            features_final = [tuple(f) if not isinstance(f, tuple) else f for f in raw_features]
        else:
            # .tolist() on numpy array converts rows to lists, so convert each to tuple
            features_final = [tuple(f) for f in raw_features.tolist()]

    except Exception as e:
        logging.error(f"Exception while feature selector fits/transforms: {e}")
        # Use getattr to safely handle case where interaction_details might not exist
        details = getattr(ld_node, 'interaction_details_after_ld', {})
        return float32_t(-1.0), int16_t(0), pop_id, [], details

    if feature_count == 0:
        return float32_t(-1.0), int16_t(0), pop_id, [], ld_node.interaction_details_after_ld

    return float32_t(1.0), int16_t(feature_count), pop_id, features_final, ld_node.interaction_details_after_ld

@ray.remote
def ray_eval_pipeline_fs(component_map: Dict[snp_t, Dict],
                         y_train: npt.NDArray,
                         train_idx: npt.NDArray,
                         selector_node: SelectorNode,
                         pop_id: uint32_t) -> Tuple[float32_t, int16_t, uint32_t, List, Dict]:
    """
    Evaluate a pipeline with only a feature selection node using Ray (no LD pruning).

    Parameters:
        component_map (Dict[snp_t, Dict]): Dictionary mapping interaction names to their component information
            (snp1_name, snp2_name, snp1_ray_id, snp2_ray_id, encoded_ray_id).
        y_train (npt.NDArray): Phenotype data array.
        train_idx (npt.NDArray): Indices for training data.
        selector_node (SelectorNode): Fitted feature selector node.
        pop_id (uint32_t): Population ID for tracking.

    Returns:
        Tuple containing:
            float32_t: Error value (-1.0 if failure, 1.0 if success).
            int16_t: Feature count after selection.
            uint32_t: Population ID.
            List[interaction_t]: List of selected interaction names after feature selection.
            Dict[snp_t, Dict]: Empty dictionary (no LD node details).
    """

    # 1. HELPER: Ensure keys are standard Python types (Strings or Tuples of Strings)
    def sanitize_key(k):
        if isinstance(k, tuple):
            return tuple(str(x.item()) if hasattr(x, 'item') else str(x) for x in k)
        return str(k.item()) if hasattr(k, 'item') else str(k)

    # 2. SANITIZE INPUTS IMMEDIATELY
    clean_component_map = {sanitize_key(k): v for k, v in component_map.items()}

    feature_count = 0
    features_final = []
    interaction_names = list(clean_component_map.keys())

    # 3. BUILD LOCAL MAP (Resolving Ray ObjectIDs)
    local_component_map = {}
    try:
        for name in interaction_names:
            # Note: train_idx is used to slice the data immediately to save memory
            local_component_map[name] = {
                'encoded_data': ray.get(clean_component_map[name]['encoded_ray_id'])[train_idx].ravel()
            }
    except Exception as e:
        logging.error(f"Error resolving Ray objects: {e}")
        return float32_t(-1.0), int16_t(0), pop_id, [], {}

    # 4. CREATE DATAFRAME WITH ENCODED INTERACTIONS
    try:
        interaction_train_encoded_df = pd.DataFrame({
            name: local_component_map[name]['encoded_data']
            for name in interaction_names
        })
    except Exception as e:
        logging.error(f"Error creating DataFrame: {e}")
        return float32_t(-1.0), int16_t(0), pop_id, [], {}

    # 5. FIT SELECTOR NODE
    try:
        selector_node.fit(interaction_train_encoded_df, y_train[train_idx])
        interaction_train_encoded_df = selector_node.transform(interaction_train_encoded_df)
        feature_count = selector_node.get_feature_count()

        # Get final names and ensure they are tuples of snp_t (interaction_t)
        raw_features = selector_node.get_feature_names(interaction_names)
        if isinstance(raw_features, list):
            features_final = [tuple(f) if not isinstance(f, tuple) else f for f in raw_features]
        else:
            # .tolist() on numpy array converts rows to lists, so convert each to tuple
            features_final = [tuple(f) for f in raw_features.tolist()]

    except Exception as e:
        logging.error(f"Exception while feature selector fits/transforms: {e}")
        return float32_t(-1.0), int16_t(0), pop_id, [], {}

    # need this bc the root node would tell us if nothing was passed to it with the old implementation
    if feature_count == 0:
        return float32_t(-1.0), int16_t(0), pop_id, [], {}

    # return features that made it passed fs for this pipeline
    return float32_t(1.0), int16_t(feature_count), pop_id, features_final, {}

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
    y_train_centered = y[train_idx] - np.mean(y[train_idx])
    y_valid_centered = y[valid_idx] - np.mean(y[train_idx])

    # Fit a base model and then a joint model to correctly calculate the pipeline epistasis R2.

    # Step 1: Fit base model (main effects only) to get baseline validation R2
    try:
        base_regressor = sm.OLS(y_train_centered, sm.add_constant(X_univariate_matrix_train_centered, has_constant='add'))
        base_results = base_regressor.fit_regularized(L1_wt=0.5, alpha=0.01) # alpha is a hyperparameter that controls the strength of regularization, can be tuned if needed but 0.1 is a common starting point for ridge regression
        base_pred = base_results.predict(sm.add_constant(X_univariate_matrix_valid_centered, has_constant='add'))
        base_r2 = r2_score(y_valid_centered, base_pred)
     
    except Exception as e:
        logging.error(f"Exception while fitting the base model ridge regression: {e}")
        print(f"Error fitting ridge regression for pipeline evaluation: {e}")
        return float32_t(-1.0), pop_id, float32_t(-1.0)

    # Step 2: Fit joint model (main effects + interactions) and calculate the incremental R2 contributed by the interactions while controlling for main effects (phantom epistasis check)
    try:
        X_joint_train = np.column_stack((X_univariate_matrix_train_centered, X_interaction_matrix_train_centered))
        X_joint_valid = np.column_stack((X_univariate_matrix_valid_centered, X_interaction_matrix_valid_centered))

        joint_regressor = sm.OLS(y_train_centered, sm.add_constant(X_joint_train, has_constant='add'))
        joint_results = joint_regressor.fit_regularized(L1_wt=0.5, alpha=0.015) # alpha is a hyperparameter that controls the strength of regularization, can be tuned if needed but 0.1 is a common starting point for ridge regression
        joint_pred = joint_results.predict(sm.add_constant(X_joint_valid, has_constant='add'))
        joint_r2 = r2_score(y_valid_centered, joint_pred)

        # Epistais R2 is strictly the variance added by the interaction features beyond the main effects, so we subtract the base_r2 from the joint_r2 to get the incremental R2 contributed by the interactions while controlling for main effects (phantom epistasis check)
        epistasis_r2 = float32_t(joint_r2 - base_r2)
    except Exception as e:
        logging.error(f"Error while scoring the pipeline: {e}")
        print(f"Error scoring the pipeline: {e}")
        return float32_t(-1.0), pop_id, float32_t(-1.0)

    return epistasis_r2, pop_id, float32_t(1.0)
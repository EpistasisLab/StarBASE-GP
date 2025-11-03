import time
from ..Base.selectors import SelectorNode
from ..Base.types import (float32_t, int16_t, prob_t, int32_t, snp_t, uint16_t)

import ray
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
from typing import List, Dict, Tuple, Set
import logging
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy.typing as npt
import numba

# Pre-defined LUTs (global constants for maximum performance) for encoders
LUT_DOMINANT = np.array([0.0, 1.0, 1.0], dtype=float32_t)
LUT_RECESSIVE = np.array([0.0, 0.0, 1.0], dtype=float32_t)
LUT_HETEROSIS = np.array([0.0, 1.0, 0.0], dtype=float32_t)
LUT_UNDERDOMINANT = np.array([0.5, 0.0, 1.0], dtype=float32_t)
LUT_OVERDOMINANT = np.array([0.0, 1.0, 0.5], dtype=float32_t)
LUT_SUBADDITIVE = np.array([0.0, 0.25, 1.0], dtype=float32_t)
LUT_SUPERADDITIVE = np.array([0.0, 0.75, 1.0], dtype=float32_t)
# LUT not needed for additive encoding as the data should already be in the correct format (0.0, 0.5, 1.0)

# Ultra-optimized numba encoding functions
@numba.njit(cache=True)
def _encode_with_lut_fast(X, lut):
    """
    Ultra-fast vectorized encoding using lookup table.
    Uses fastmath for SIMD optimization and cache=True to cache compiled code.
    """
    n = X.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        # Direct integer conversion: 0.0->0, 0.5->1, 1.0->2
        idx = int(X[i] * 2.0)
        out[i] = lut[idx]
    return out

@numba.njit(cache=True)
def encode_dominant(X):
    return _encode_with_lut_fast(X, LUT_DOMINANT)

@numba.njit(cache=True)
def encode_recessive(X):
    return _encode_with_lut_fast(X, LUT_RECESSIVE)

@numba.njit(cache=True)
def encode_heterosis(X):
    return _encode_with_lut_fast(X, LUT_HETEROSIS)

@numba.njit(cache=True)
def encode_underdominant(X):
    return _encode_with_lut_fast(X, LUT_UNDERDOMINANT)

@numba.njit(cache=True)
def encode_overdominant(X):
    return _encode_with_lut_fast(X, LUT_OVERDOMINANT)

@numba.njit(cache=True)
def encode_subadditive(X):
    return _encode_with_lut_fast(X, LUT_SUBADDITIVE)

@numba.njit(cache=True)
def encode_superadditive(X):
    return _encode_with_lut_fast(X, LUT_SUPERADDITIVE)

# No encode function needed for additive encoding

@numba.njit(cache=True)
def build_pager_lut(X, y):
    """
    Build a PAGER LUT (3 values for genotypes 0.0, 0.5, 1.0)
    based on phenotype means normalized relative to genotype 0 mean (anchor).
    Optimized with fastmath and cache for repeated calls.
    """
    means = np.zeros(3, dtype=float32_t)
    present = np.zeros(3, dtype=float32_t)
    geno_keys = np.array([0.0, 0.5, 1.0], dtype=float32_t)

    # Compute phenotype means per genotype
    for k, g in enumerate(geno_keys):
        mask = X == g
        n = np.sum(mask)
        if n > 0:
            means[k] = np.mean(y[mask])
            present[k] = 1

    # Determine anchor (genotype 0.0 if present)
    if present[0]:
        anchor = means[0]
    else:
        first_present = -1
        for i in range(3):
            if present[i]:
                first_present = i
                break
        anchor = means[first_present] if first_present >= 0 else 0.0

    # Compute relative differences
    rel = means - anchor

    # Normalize to [0, 1] (min-max scaling) among present genotypes
    if np.any(present):
        vals = rel[present == 1]
        mn, mx = np.min(vals), np.max(vals)
        scaled = np.empty(3, dtype=float32_t)
        if mx - mn == 0:
            for i in range(3):
                scaled[i] = 0.0 if present[i] else 0.5
        else:
            for i in range(3):
                scaled[i] = (rel[i] - mn) / (mx - mn) if present[i] else 0.5 # if a genotype is not present, assign 0.5
    else:
        scaled = np.full(3, 0.5, dtype=float32_t) 

    return scaled  # shape (3,)

@numba.njit(cache=True)
def encode_pager(X, lut):
    """
    Encode genotypes using a precomputed PAGER LUT.
    Ultra-fast with fastmath and cache optimizations.
    """
    n = X.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        genotype = int(X[i] * 2.0)
        out[i] = lut[genotype]
    return out

# -----------------------------
# Generic Ray worker template
# -----------------------------
def _ray_snp_eval_template(X, y, train_idx, valid_idx, snp, lo, encoder_func=None):
    assert isinstance(X, np.ndarray), "X should be a numpy array"

    # Special case: if encoder_func is None, data is already in correct format (e.g., additive)
    if encoder_func is None:
        X_encoded = X
    else:
        try:
            X_encoded = encoder_func(X)
        except Exception as e:
            logging.error(f"Encoding error for {lo}: {e}")
            return float32_t(0.0), snp, lo, float32_t(-1.0)

    try:
        regressor = sm.OLS(y[train_idx], sm.add_constant(X_encoded[train_idx], has_constant='add'))
        results = regressor.fit()
    except Exception as e:
        logging.error(f"OLS fitting error for {lo}: {e}")
        return float32_t(0.0), snp, lo, float32_t(-1.0)

    try:
        y_pred = results.predict(sm.add_constant(X_encoded[valid_idx], has_constant='add'))
        score = r2_score(y[valid_idx], y_pred)
    except Exception as e:
        logging.error(f"Scoring error for {lo}: {e}")
        return float32_t(0.0), snp, lo, float32_t(-1.0)

    return float32_t(score), snp, lo, float32_t(1.0)

# Ray-remote functions to evaluate SNPs with different encodings

@ray.remote
def ray_snp_eval_dom(X, y, train_idx, valid_idx, snp, lo=snp_t('dominant')):
    return _ray_snp_eval_template(X, y, train_idx, valid_idx, snp, lo, encode_dominant)

@ray.remote
def ray_snp_eval_rec(X, y, train_idx, valid_idx, snp, lo=snp_t('recessive')):
    return _ray_snp_eval_template(X, y, train_idx, valid_idx, snp, lo, encode_recessive)

@ray.remote
def ray_snp_eval_het(X, y, train_idx, valid_idx, snp, lo=snp_t('heterosis')):
    return _ray_snp_eval_template(X, y, train_idx, valid_idx, snp, lo, encode_heterosis)

@ray.remote
def ray_snp_eval_und(X, y, train_idx, valid_idx, snp, lo=snp_t('underdominant')):
    return _ray_snp_eval_template(X, y, train_idx, valid_idx, snp, lo, encode_underdominant)

@ray.remote
def ray_snp_eval_ovd(X, y, train_idx, valid_idx, snp, lo=snp_t('overdominant')):
    return _ray_snp_eval_template(X, y, train_idx, valid_idx, snp, lo, encode_overdominant)

@ray.remote
def ray_snp_eval_sub(X, y, train_idx, valid_idx, snp, lo=snp_t('subadditive')):
    return _ray_snp_eval_template(X, y, train_idx, valid_idx, snp, lo, encode_subadditive)

@ray.remote
def ray_snp_eval_sup(X, y, train_idx, valid_idx, snp, lo=snp_t('superadditive')):
    return _ray_snp_eval_template(X, y, train_idx, valid_idx, snp, lo, encode_superadditive)

@ray.remote
def ray_snp_eval_add(X, y, train_idx, valid_idx, snp, lo=snp_t('additive')):
    """
    Evaluate additive encoding - data is already in additive format (0.0, 0.5, 1.0),
    so no encoding transformation is needed. Pass None as encoder_func.
    """
    return _ray_snp_eval_template(X, y, train_idx, valid_idx, snp, lo, encoder_func=None)

@ray.remote
def ray_snp_eval_pager(X, y, train_idx, valid_idx, snp, lo=snp_t('pager')):
    return _ray_snp_eval_template(X, y, train_idx, valid_idx, snp, lo,
                                 lambda X_data: encode_pager(X_data, build_pager_lut(X_data[train_idx], y[train_idx])))

# permuation feature importance
@ray.remote
def ray_pfi(X, y, train_idx, valid_idx, new_column_names, root_node, random_state, pop_id) -> None:

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

# evaluate unseen snps and encode them efficiently using numba
@ray.remote
def ray_snp_encoder(X, y, train_idx, enc: snp_t, snp: snp_t) -> Tuple[np.ndarray, snp_t]:
    """
    Efficiently encode SNP data using pre-defined LUTs and numba.
    For PAGER encoding, builds LUT from training data and applies to entire X.
    Optimized for maximum speed with direct LUT access.
    """
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert isinstance(y, np.ndarray), "y should be a numpy array"
    assert isinstance(enc, snp_t), "enc should be a numpy string"

    # Get the encoding string (convert from numpy string if needed)
    enc_str = str(enc) if isinstance(enc, np.str_) else enc
    
    # Special case: additive - data is already in correct format
    if enc_str == 'additive':
        return X, snp
    
    # Special case: PAGER - build LUT from training data, apply to all data
    if enc_str == 'pager':
        lut = build_pager_lut(X[train_idx], y[train_idx])
        X_encoded = encode_pager(X, lut)
        return X_encoded, snp

    # Direct LUT mapping for maximum speed (no function call overhead)
    lut_map = {
        'dominant': LUT_DOMINANT,
        'recessive': LUT_RECESSIVE,
        'heterosis': LUT_HETEROSIS,
        'underdominant': LUT_UNDERDOMINANT,
        'overdominant': LUT_OVERDOMINANT,
        'subadditive': LUT_SUBADDITIVE,
        'superadditive': LUT_SUPERADDITIVE,
    }
    
    if enc_str in lut_map:
        # Direct encoding with pre-allocated LUT (fastest path)
        X_encoded = _encode_with_lut_fast(X, lut_map[enc_str])
        return X_encoded, snp
    else:
        logging.error(f"Encoding {enc} not recognized for SNP {snp}.")
        return X, snp

# all univariate snps with their best lo goes to the LD operator, then the feature selector
@ray.remote
def ray_eval_pipeline_ld_fs(snp_names: List[snp_t],
                            x_train_ori: List[ray.ObjectID],
                            x_train_enc: List[ray.ObjectID],
                            y_train: npt.NDArray,
                            train_idx: npt.NDArray,
                            selector_node: SelectorNode,
                            ld_node: SelectorNode,
                            pop_id: uint16_t,        # error. feature count. pop_id. details after ld node. snp_after_ld
                            snp_r2_set: Set) -> Tuple[float32_t, int16_t, uint16_t, List[snp_t], Dict[snp_t, Dict]]:

    # make dictionary to hold the snp r2 scores
    snp_r2_dict = {p[0]: p[1] for p in snp_r2_set}
    # hold feature counts across all folds
    feature_count = 0
    # holds feature list
    features_final = []
    # create both original and encoded dataframes from the ray object ids
    x_train_transformed_df = pd.DataFrame({name: ray.get(data_obj)[train_idx].tolist() for name, data_obj in zip(snp_names, x_train_enc)})
    x_train_original_df = pd.DataFrame({name: ray.get(data_obj)[train_idx].tolist() for name, data_obj in zip(snp_names, x_train_ori)})

    # finding out the time taken for LD node alone
    ld_start_time = time.time()
    # fit the LD node - send the unencoded snps for pearson's correlation calculation, the encoded data, the target and the snp r2 dictionary having the best lo r2
    try:
        ld_node.fit(x_train_original_df, x_train_transformed_df, y_train[train_idx], snp_r2_dict)
        selected_features_after_ld = ld_node.selected_features_
        # keeping only the selected features (not pruned out by LD) after the LD node
        x_train_transformed_df = pd.DataFrame(x_train_transformed_df[selected_features_after_ld], columns=selected_features_after_ld)
        if x_train_transformed_df.empty:
            print("No features selected after LD node")
            return float32_t(-1.0), int16_t(0), pop_id, [], ld_node.snp_details_after_ld # all SNPs in the pipeline were pruned out by LD, should not be happening but just a check

    except Exception as e:
        logging.error(f"Exception while fitting LD node: {e}")
        return float32_t(-1.0), int16_t(0), pop_id, [], {}
    ld_end_time = time.time()
    ld_duration = (ld_end_time - ld_start_time) / 60
    print(f"LD node processing time: {ld_duration:.4f} minutes")


    # adding the selector nodes
    # finding out the time taken for feature selector alone
    fs_start_time = time.time()
    try:
        # get snps from selector node
        selector_node.fit(x_train_transformed_df, y_train[train_idx])
        x_train_transformed_df = selector_node.transform(x_train_transformed_df) # this dataframe goes into regressor
        feature_count = selector_node.get_feature_count() # number of selected features after the selector node
        features_final = (selector_node.get_feature_names(selected_features_after_ld)) # get the names of the features after the selector node by sending the selected features after the LD node

    except Exception as e:
        logging.error(f"Exception while feature selector fits/transforms: {e}")
        return float32_t(-1.0), int16_t(0), pop_id, [], ld_node.snp_details_after_ld
    fs_end_time = time.time()
    fs_duration = (fs_end_time - fs_start_time) / 60
    print(f"Feature Selector: {selector_node.name} processing time: {fs_duration:.4f} minutes")

    # need this bc the root node would tell us if nothing was passed to it with the old implementation
    if feature_count == 0:
        return float32_t(-1.0), int16_t(0), pop_id, [], ld_node.snp_details_after_ld

    # if features_final is not a list, convert it to a list
    if not isinstance(features_final, list):
        features_final = features_final.tolist()

    # return features that made it passed ld and fs for this pipeline
    return float32_t(1.0), int16_t(feature_count), pop_id, [snp_t(feature) for feature in features_final], ld_node.snp_details_after_ld

# all univariate snps/nodes with their best lo go straight to the feature selector
@ray.remote
def ray_eval_pipeline_fs(snp_names: List[snp_t],
                         x_train_enc: List[ray.ObjectID],
                         y_train: npt.NDArray,
                         train_idx: npt.NDArray,
                         selector_node: SelectorNode,   # error. feature count. pop_id. details after ld node. snp_after_ld (ignore for this one)
                         pop_id: uint16_t) ->     Tuple[float32_t, int16_t, uint16_t, List[snp_t], Dict[snp_t, Dict]]:


    # hold feature counts across all folds
    feature_count = 0
    # holds feature list
    features_final = []
    # create both original and encoded dataframes from the ray object ids
    x_train_transformed_df = pd.DataFrame({name: ray.get(data_obj)[train_idx].tolist() for name, data_obj in zip(snp_names, x_train_enc)})

    # adding the selector and regressor nodes
    try:
        # get snps from selector node
        selector_node.fit(x_train_transformed_df, y_train[train_idx])
        x_train_transformed_df = selector_node.transform(x_train_transformed_df) # this dataframe goes into regressor
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

# evaluate piepline with only snps that make it passed ld and feature selector
@ray.remote
def ray_eval_pipeline_r2(X: List[ray.ObjectID],
                      y: ray.ObjectID,
                      train_idx,
                      valid_idx,                  #r2.       #id.     # error?
                      pop_id: uint16_t) -> Tuple[float32_t, uint16_t, float32_t]:

    # create dataset
    X_matrix = np.column_stack([ray.get(x) for x in X])

    # define the regressor
    regressor = sm.OLS(y[train_idx], sm.add_constant(X_matrix[train_idx], has_constant='add'))

    # try to fit the pipeline
    try:
        results = regressor.fit()
    except Exception as e:
        logging.error(f"Exception while fitting the pipeline: {e}")
        return float32_t(-1.0), pop_id, float32_t(-1.0)

    # try to score the pipeline
    try:
        y_pred = results.predict(sm.add_constant(X_matrix[valid_idx], has_constant='add'))
    except Exception as e:
        logging.error(f"Error while scoring the pipeline: {e}")
        return float32_t(-1.0), pop_id, float32_t(-1.0)

    return float32_t(r2_score(y[valid_idx], y_pred)), pop_id, float32_t(1.0)

# Optimized: Evaluate all 9 encodings in a single Ray call
@ray.remote
def ray_snp_eval_all_encodings(X, y, train_idx, valid_idx, snp):
    """
    Evaluate all 9 encoding types for a single SNP in one Ray call.
    This dramatically reduces Ray scheduling overhead by batching all encodings together.
    
    Returns:
        Dict[str, Tuple[float, str, str, float]]: 
            Dictionary mapping encoding name to (r2_score, snp, encoding, error_flag)
    """
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    
    results = {}
    
    # Define all encodings to evaluate
    encodings = [
        ('additive', None),  # No transformation needed
        ('dominant', encode_dominant),
        ('recessive', encode_recessive),
        ('heterosis', encode_heterosis),
        ('underdominant', encode_underdominant),
        ('overdominant', encode_overdominant),
        ('subadditive', encode_subadditive),
        ('superadditive', encode_superadditive),
        ('pager', lambda X_data: encode_pager(X_data, build_pager_lut(X_data[train_idx], y[train_idx])))
    ]
    
    for enc_name, encoder_func in encodings:
        try:
            # Encode the data
            if encoder_func is None:
                X_encoded = X
            else:
                X_encoded = encoder_func(X)
            
            # Fit OLS model
            regressor = sm.OLS(y[train_idx], sm.add_constant(X_encoded[train_idx], has_constant='add'))
            fit_results = regressor.fit()
            
            # Score on validation set
            y_pred = fit_results.predict(sm.add_constant(X_encoded[valid_idx], has_constant='add'))
            score = r2_score(y[valid_idx], y_pred)
            
            results[enc_name] = (float32_t(score), snp, snp_t(enc_name), float32_t(1.0))
            
        except Exception as e:
            logging.error(f"Error evaluating {enc_name} for SNP {snp}: {e}")
            results[enc_name] = (float32_t(0.0), snp, snp_t(enc_name), float32_t(-1.0))
    
    return results
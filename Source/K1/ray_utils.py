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

# evaluate unseen snps for additive encoding and a specific fold
@ray.remote                                                                          # r2     snp    lo     error
def ray_snp_eval_add(X, y, train_idx, valid_idx, snp, lo=snp_t('additive')) -> Tuple[float32_t,snp_t,snp_t,float32_t]:
    # quick checks
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert '.' in snp, "snp should be in the format 'chr.pos'"

    # transform data for regression model (adding constant for intercept)
    regressor = sm.OLS(y[train_idx], sm.add_constant(X[train_idx], has_constant='add'))

    # try to fit the pipeline
    try:
        results = regressor.fit()
    except Exception as e:
        logging.error(f"Exception while fitting the pipeline: {e}")
        return float32_t(0.0), snp, lo, float32_t(-1.0)

    # try to score the pipeline
    try:
        y_pred = results.predict(sm.add_constant(X[valid_idx], has_constant='add'))
    except Exception as e:
        logging.error(f"Error while scoring the pipeline: {e}")
        return float32_t(0.0), snp, lo, float32_t(-1.0)

    return float32_t(r2_score(y[valid_idx], y_pred)), snp, lo, float32_t(1.0)

# evaluate unseen snps for dominant encoding and a specific fold
@ray.remote                                                                          # r2     snp    lo     error
def ray_snp_eval_dom(X, y, train_idx, valid_idx, snp, lo=snp_t('dominant')) -> Tuple[float32_t,snp_t,snp_t,float32_t]:
    # quick checks
    assert isinstance(X, np.ndarray), "X should be a numpy array"

    # todo: need to add the best way to transform the data for encoding
    mapping = {float32_t(0.0): float32_t(0.0), float32_t(0.5): float32_t(1.0), float32_t(1.0): float32_t(1.0)}
    X_encoded = X.copy()
    for original, encoded in mapping.items():
        X_encoded[X_encoded == original] = encoded

    # transform data for regression model
    regressor = sm.OLS(y[train_idx], sm.add_constant(X_encoded[train_idx], has_constant='add'))

    # try to fit the pipeline
    try:
        results = regressor.fit()
    except Exception as e:
        logging.error(f"Exception while fitting the pipeline: {e}")
        return float32_t(0.0), snp, lo, float32_t(-1.0)

    # try to score the pipeline
    try:
        y_pred = results.predict(sm.add_constant(X_encoded[valid_idx], has_constant='add'))
    except Exception as e:
        logging.error(f"Error while scoring the pipeline: {e}")
        return float32_t(0.0), snp, lo, float32_t(-1.0)

    return float32_t(r2_score(y[valid_idx], y_pred)), snp, lo, float32_t(1.0)

# evaluate unseen snps for recessive encoding and a specific fold
@ray.remote                                                                            # r2     snp    lo     error
def ray_snp_eval_rec(X, y, train_idx, valid_idx, snp, lo=snp_t('recessive')) -> Tuple[float32_t,snp_t,snp_t,float32_t]:
    # quick checks
    assert isinstance(X, np.ndarray), "X should be a numpy array"

    # todo: need to add the best way to transform the data for encoding
    mapping = {float32_t(0.0): float32_t(0.0), float32_t(0.5): float32_t(0.0), float32_t(1.0): float32_t(1.0)}
    X_encoded = X.copy()
    for original, encoded in mapping.items():
        X_encoded[X_encoded == original] = encoded

    # transform data for regression model
    regressor = sm.OLS(y[train_idx], sm.add_constant(X_encoded[train_idx], has_constant='add'))

    # try to fit the pipeline
    try:
        results = regressor.fit()
    except Exception as e:
        logging.error(f"Exception while fitting the pipeline: {e}")
        return float32_t(0.0), snp, lo, float32_t(-1.0)

    # try to score the pipeline
    try:
        y_pred = results.predict(sm.add_constant(X_encoded[valid_idx], has_constant='add'))
    except Exception as e:
        logging.error(f"Error while scoring the pipeline: {e}")
        return float32_t(0.0), snp, lo, float32_t(-1.0)

    return float32_t(r2_score(y[valid_idx], y_pred)), snp, lo, float32_t(1.0)

# evaluate unseen snps for heterosis encoding and a specific fold
@ray.remote                                                                            # r2     snp    lo     error
def ray_snp_eval_het(X, y, train_idx, valid_idx, snp, lo=snp_t('heterosis')) -> Tuple[float32_t,snp_t,snp_t,float32_t]:
    # quick checks
    assert isinstance(X, np.ndarray), "X should be a numpy array"

    # todo: need to add the best way to transform the data for encoding
    mapping = {float32_t(0.0): float32_t(0.0), float32_t(0.5): float32_t(1.0), float32_t(1.0): float32_t(0.0)}
    X_encoded = X.copy()
    for original, encoded in mapping.items():
        X_encoded[X_encoded == original] = encoded

    # transform data for regression model
    regressor = sm.OLS(y[train_idx], sm.add_constant(X_encoded[train_idx], has_constant='add'))

    # try to fit the pipeline
    try:
        results = regressor.fit()
    except Exception as e:
        logging.error(f"Exception while fitting the pipeline: {e}")
        return float32_t(0.0), snp, lo, float32_t(-1.0)

    # try to score the pipeline
    try:
        y_pred = results.predict(sm.add_constant(X_encoded[valid_idx], has_constant='add'))
    except Exception as e:
        logging.error(f"Error while scoring the pipeline: {e}")
        return float32_t(0.0), snp, lo, float32_t(-1.0)

    return float32_t(r2_score(y[valid_idx], y_pred)), snp, lo, float32_t(1.0)

# evaluate unseen snps for underdominant encoding and a specific fold
@ray.remote                                                                                # r2     snp    lo     error
def ray_snp_eval_und(X, y, train_idx, valid_idx, snp, lo=snp_t('underdominant')) -> Tuple[float32_t,snp_t,snp_t,float32_t]:
    # quick checks
    assert isinstance(X, np.ndarray), "X should be a numpy array"

    # todo: need to add the best way to transform the data for encoding
    mapping = {float32_t(0.0): float32_t(0.5), float32_t(0.5): float32_t(0.0), float32_t(1.0): float32_t(1.0)}
    X_encoded = X.copy()
    for original, encoded in mapping.items():
        X_encoded[X_encoded == original] = encoded

    # transform data for regression model
    regressor = sm.OLS(y[train_idx], sm.add_constant(X_encoded[train_idx], has_constant='add'))

    # try to fit the pipeline
    try:
        results = regressor.fit()
    except Exception as e:
        logging.error(f"Exception while fitting the pipeline: {e}")
        return float32_t(0.0), snp, lo, float32_t(-1.0)

    # try to score the pipeline
    try:
        y_pred = results.predict(sm.add_constant(X_encoded[valid_idx], has_constant='add'))
    except Exception as e:
        logging.error(f"Error while scoring the pipeline: {e}")
        return float32_t(0.0), snp, lo, float32_t(-1.0)

    return float32_t(r2_score(y[valid_idx], y_pred)), snp, lo, float32_t(1.0)

# evaluate unseen snps for overdominant encoding and a specific fold
@ray.remote                                                                                # r2     snp    lo     error
def ray_snp_eval_ovd(X, y, train_idx, valid_idx, snp, lo=snp_t('overdominant')) -> Tuple[float32_t,snp_t,snp_t,float32_t]:
    # quick checks
    assert isinstance(X, np.ndarray), "X should be a numpy array"

    # todo: need to add the best way to transform the data for encoding
    mapping = {float32_t(0.0): float32_t(0.0), float32_t(0.5): float32_t(1.0), float32_t(1.0): float32_t(0.5)}
    X_encoded = X.copy()
    for original, encoded in mapping.items():
        X_encoded[X_encoded == original] = encoded

    # transform data for regression model
    regressor = sm.OLS(y[train_idx], sm.add_constant(X_encoded[train_idx], has_constant='add'))

    # try to fit the pipeline
    try:
        results = regressor.fit()
    except Exception as e:
        logging.error(f"Exception while fitting the pipeline: {e}")
        return float32_t(0.0), snp, lo, float32_t(-1.0)

    # try to score the pipeline
    try:
        y_pred = results.predict(sm.add_constant(X_encoded[valid_idx], has_constant='add'))
    except Exception as e:
        logging.error(f"Error while scoring the pipeline: {e}")
        return float32_t(0.0), snp, lo, float32_t(-1.0)

    return float32_t(r2_score(y[valid_idx], y_pred)), snp, lo, float32_t(1.0)

# evaluate unseen snps for subadditive encoding and a specific fold
@ray.remote                                                                              # r2     snp    lo     error
def ray_snp_eval_sub(X, y, train_idx, valid_idx, snp, lo=snp_t('subadditive')) -> Tuple[float32_t,snp_t,snp_t,float32_t]:
    # quick checks
    assert isinstance(X, np.ndarray), "X should be a numpy array"

    # todo: need to add the best way to transform the data for encoding
    mapping = {float32_t(0.0):float32_t(0.0), float32_t(0.5): float32_t(0.25), float32_t(1.0): float32_t(1.0)}
    X_encoded = X.copy()
    for original, encoded in mapping.items():
        X_encoded[X_encoded == original] = encoded

    # transform data for regression model
    regressor = sm.OLS(y[train_idx], sm.add_constant(X_encoded[train_idx], has_constant='add'))

    # try to fit the pipeline
    try:
        results = regressor.fit()
    except Exception as e:
        logging.error(f"Exception while fitting the pipeline: {e}")
        return float32_t(0.0), snp, lo, float32_t(-1.0)

    # try to score the pipeline
    try:
        y_pred = results.predict(sm.add_constant(X_encoded[valid_idx], has_constant='add'))
    except Exception as e:
        logging.error(f"Error while scoring the pipeline: {e}")
        return float32_t(0.0), snp, lo, float32_t(-1.0)

    return float32_t(r2_score(y[valid_idx], y_pred)), snp, lo, float32_t(1.0)

# evaluate unseen snps for superadditive encoding and a specific fold
@ray.remote                                                                                # r2     snp    lo     error
def ray_snp_eval_sup(X, y, train_idx, valid_idx, snp, lo=snp_t('superadditive')) -> Tuple[float32_t,snp_t,snp_t,float32_t]:
    # quick checks
    assert isinstance(X, np.ndarray), "X should be a numpy array"

    # todo: need to add the best way to transform the data for encoding
    mapping = {float32_t(0.0):float32_t(0.0), float32_t(0.5): float32_t(0.75), float32_t(1.0): float32_t(1.0)}
    X_encoded = X.copy()
    for original, encoded in mapping.items():
        X_encoded[X_encoded == original] = encoded

    # transform data for regression model
    regressor = sm.OLS(y[train_idx], sm.add_constant(X_encoded[train_idx], has_constant='add'))

    # try to fit the pipeline
    try:
        results = regressor.fit()
    except Exception as e:
        logging.error(f"Exception while fitting the pipeline: {e}")
        return float32_t(0.0), snp, lo, float32_t(-1.0)

    # try to score the pipeline
    try:
        y_pred = results.predict(sm.add_constant(X_encoded[valid_idx], has_constant='add'))
    except Exception as e:
        logging.error(f"Error while scoring the pipeline: {e}")
        return float32_t(0.0), snp, lo, float32_t(-1.0)

    return float32_t(r2_score(y[valid_idx], y_pred)), snp, lo, float32_t(1.0)

# evaluate unseen snps for PAGER encoding and a specific fold
# @ray.remote                                                                          # r2     snp    lo     error
# def ray_snp_eval_pager(X, y, train_idx, valid_idx, snp, lo=snp_t('pager')) -> Tuple[float32_t,snp_t,snp_t,float32_t]:
#     # quick checks
#     assert isinstance(X, np.ndarray), "X should be a numpy array"

#     uni_node = UniPAGERNode(X=X[train_idx], y=y[train_idx])
#     X_transformed = np.array(uni_node.transform(X[train_idx]).ravel(), dtype=float32_t)

#     # transform data for regression model
#     X_transformed = sm.add_constant(X_transformed, has_constant='add')
#     regressor = sm.OLS(y[train_idx], X_transformed)

#     # try to fit the pipeline
#     try:
#         results = regressor.fit()
#     except Exception as e:
#         logging.error(f"Exception while fitting the pipeline: {e}")
#         return float32_t(-100000.0), snp, lo

#     # try to score the pipeline
#     try:
#         X_val = np.array(uni_node.transform(X[valid_idx]).ravel(), dtype=float32_t)
#         X_val = sm.add_constant(X_val, has_constant='add')
#         y_pred = results.predict(X_val)
#     except Exception as e:
#         logging.error(f"Error while scoring the pipeline: {e}")
#         return float32_t(-100000.0), snp, lo

#     return float32_t(r2_score(y[valid_idx], y_pred)), snp, lo

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

    for i in range(len(new_column_names)):
        if pfi.importances_mean[i] > 0:
            pfi_results[new_column_names[i]] = pfi.importances_mean[i]

    return pfi_results, pop_id

# evaluate unseen snps (additive is not needed as it is the original encoding)
@ray.remote
def ray_snp_encoder(X, y,  train_idx, enc:snp_t, snp: snp_t) -> Tuple[np.ndarray,snp_t]:
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert isinstance(y, np.ndarray), "y should be a numpy array"
    assert isinstance(enc, snp_t), "enc should be a numpy string"

    # copy to modify based on encoding
    X_encoded = X.copy()

    # todo: special case for pager encoding
    if enc == snp_t('pager'):
        return X, snp

    mapping = None
    if enc == snp_t('dominant'):
        mapping = {float32_t(0.0): float32_t(0.0), float32_t(0.5): float32_t(1.0), float32_t(1.0): float32_t(1.0)}
    elif enc == snp_t('recessive'):
        mapping = {float32_t(0.0): float32_t(0.0), float32_t(0.5): float32_t(0.0), float32_t(1.0): float32_t(1.0)}
    elif enc == snp_t('heterosis'):
        mapping = {float32_t(0.0): float32_t(0.0), float32_t(0.5): float32_t(1.0), float32_t(1.0): float32_t(0.0)}
    elif enc == snp_t('underdominant'):
        mapping = {float32_t(0.0): float32_t(0.5), float32_t(0.5): float32_t(0.0), float32_t(1.0): float32_t(1.0)}
    elif enc == snp_t('overdominant'):
        mapping = {float32_t(0.0): float32_t(0.0), float32_t(0.5): float32_t(1.0), float32_t(1.0): float32_t(0.5)}
    elif enc == snp_t('subadditive'):
        mapping = {float32_t(0.0):float32_t(0.0), float32_t(0.5): float32_t(0.25), float32_t(1.0): float32_t(1.0)}
    elif enc == snp_t('superadditive'):
        mapping = {float32_t(0.0):float32_t(0.0), float32_t(0.5): float32_t(0.75), float32_t(1.0): float32_t(1.0)}
    else:
        logging.error(f"Encoding {enc} not recognized for SNP {snp}.")
        return X, snp

    # encode the SNP based on the mapping
    for original, encoded in mapping.items():
        X_encoded[X_encoded == original] = encoded
    return X_encoded, snp

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

    # adding the selector and regressor nodes
    try:
        # get snps from selector node
        selector_node.fit(x_train_transformed_df, y_train[train_idx])
        x_train_transformed_df = selector_node.transform(x_train_transformed_df) # this dataframe goes into regressor
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
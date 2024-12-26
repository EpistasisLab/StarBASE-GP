# This class is responsible for all the post analysis functions - focusing on Univariate SNP analysis

import shap
import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import ray
import logging
import warnings
from . import snp_hub

from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.inspection import permutation_importance

# function to get the shap values
@ray.remote
def get_shap_values(pipeline, uni_snps_df, uni_nodes, X_train_id, y_train_id, X_val_id, y_val_id, snp_r2_dict, id, seed) -> npt.NDArray[np.float32]:
        
        # printing pipeline details from SHAP function
        print("Pipeline ID from PERM function: ", id, flush=True)
        print("Pipeline R2 Score from PERM function:", pipeline.get_trait_r2(), flush=True)
        print("Pipeline Feature Count from PERM function:", pipeline.get_trait_feature_cnt(), flush=True)
        print("Pipeline feature names after LD node from PERM function:", pipeline.get_trait_feature_names(), flush=True)

        features_final = pipeline.get_trait_feature_names()
        root_node = pipeline.get_root_node()
        root_name = root_node.name
        selector_name = pipeline.get_selector_node().name


        # create the pipeline to get the epi features created and selected by the selector before applying the root node
        steps = []

        # make a list of uni node names
        uni_node_names = [uni_node.get_snp_name() for uni_node in uni_nodes]
        # change the feature names to have the inheritance information
        new_column_names = [ f'{row["feature"]}_{row["inheritence"]}' for _, row in uni_snps_df.iterrows()]
        # make a dictionary of the new column names where key is the old column name and value is the new column name
        new_column_names_dict = {uni_node_names[i]: new_column_names[i] for i in range(len(uni_node_names))}


        try:
             # create the pipeline
            steps = []
            # uni nodes into one sklearn union
            steps.append(('snp_union', FeatureUnion([(uni_node.name, uni_node) for uni_node in uni_nodes if uni_node.get_snp_name() in features_final])))
            # pass to regressor
            steps.append(('regressor', root_node.regressor))

            # create the pipeline without refitting the regressor
            pipeline = SklearnPipeline(steps=steps)
            pipeline.fit(X_train_id, y_train_id)
        except Exception as e:
            logging.error(f"Exception while fitting the pipeline: {e}")
            return pd.DataFrame()

        try:
            # getting the SHAP feature importance values
            number_of_features = len(features_final)
            print("Number of features from Permutation Testing function: ", number_of_features, flush=True)
            # perform permutation importance using sklearn permutation_importance
            result = permutation_importance(pipeline, X_val_id, y_val_id, n_repeats=1, random_state=seed, n_jobs=None)
            # make a dataframe of the permutation importance with column names
            perm_importance_df = pd.DataFrame(list(zip(new_column_names, result.importances_mean)), columns=['feature', 'perm_importance'])

            # max_evals = max(500, 2 * number_of_features + 1)
            # #max_evals = 1200
            # #root_node.fit(x_final_train, y_train_id)
            # explainer = shap.Explainer(root_node.predict, X_val_id)
            # shap_values = explainer(X_val_id, max_evals=max_evals)
            # # get the mean absolute shap values used in the summary plot
            # shap_values = np.abs(shap_values.values).mean(0)
            # shap_values_df = pd.DataFrame(list(zip(new_column_names, shap_values)), columns=['feature', 'shap_value'])
            # # sort the shap_values_df by the shap_value
            # shap_values_df = shap_values_df.sort_values(by='shap_value', ascending=False)

        except Exception as e:
            logging.error(f"Exception while getting SHAP values: {e}")
            return pd.DataFrame()

        return perm_importance_df, selector_name, root_name, id

class Poster:
    def __init__(self, X_train, y_train, X_val,  y_val, hub: snp_hub.SnpHub):
        self.X_train_id = X_train
        self.y_train_id = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.hub = hub

    def run_poster(self, pipeline, uni_nodes, X_train_id, y_train_id, X_val_id, y_val_id, id, seed):
        # call the get_epi_pairs function to create the epi_pairs_df
        uni_snps_df = self.get_uni_snp(pipeline) # required to get the names of the interactions
        # get the snp_r2_dict
        snp_r2_dict = self.get_snp_hub_r2(pipeline)
        # create subsets
        results_refs = get_shap_values.remote(pipeline, uni_snps_df, uni_nodes, X_train_id, y_train_id, X_val_id, y_val_id, snp_r2_dict, id, seed)
        return results_refs


    def get_uni_snp(self, pipeline):

        # define an empty list to store the uni_spns
        uni_spns_list = []

        # get the epi pairs in the pipeline
        uni_snp = pipeline.get_uni_snps()
        print("uni_snp: ", uni_snp, flush=True)

        for snp_name in uni_snp:
            # get the best_lo for the snp
            best_lo = self.hub.get_uni_encoding(snp_name)

            # Add the interaction to the epi_pairs_list as a dictionary
            uni_spns_list.append({'feature': snp_name, 'inheritence': best_lo})

        # convert the epi_pairs_list to a dataframe
        uni_snps_df = pd.DataFrame(uni_spns_list)

        return uni_snps_df

    def get_snp_hub_r2(self, pipeline):
        # get the r2 values for the snps in the pipeline
        snp_r2_set=self.hub.generate_r2_dict(pipeline.get_uni_snps())
        snp_r2_dict = {p[0]: p[1] for p in snp_r2_set}

        return snp_r2_dict

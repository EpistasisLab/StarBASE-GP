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

# function to get the shap values
@ray.remote
def get_shap_values(pipeline, uni_snps_df, uni_nodes, X_train_id, y_train_id, X_val_id, y_val_id, snp_r2_dict, id) -> npt.NDArray[np.float32]:

        # create the pipeline to get the epi features created and selected by the selector before applying the root node
        steps = []

        # combine the epi nodes into a sklearn union
        steps.append(('uni_union', FeatureUnion([(uni_node.name, uni_node) for uni_node in uni_nodes])))
        # add the selector node
        steps.append(('selector', pipeline.get_selector_node()))
        selector_name = pipeline.get_selector_node().name

        root_node = pipeline.get_root_node()
        root_name = root_node.name

        ld_node = pipeline.get_ld_node()

        if root_node is None:
            raise ValueError("Root node is None, ensure the pipeline is correctly returning a model.")

        transformer_pipeline = SklearnPipeline(steps=steps) # pipeline with the epi features and selector
        # make a list of uni node names
        uni_node_names = [uni_node.get_snp_name() for uni_node in uni_nodes]
        # change the feature names to have the inheritance information
        new_column_names = [ f'{row["feature"]}_{row["inheritence"]}' for _, row in uni_snps_df.iterrows()]
        # make a dictionary of the new column names where key is the old column name and value is the new column name
        new_column_names_dict = {uni_node_names[i]: new_column_names[i] for i in range(len(uni_node_names))}

        # attempt to fit the pipeline
        try:
            # Fit the pipeline with warnings captured as exceptions
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=ConvergenceWarning)
                pipeline_fitted = transformer_pipeline.fit(X_train_id, y_train_id)
                # get the transformed dataset
                selected_features = pipeline_fitted.named_steps['selector'].get_feature_names(uni_node_names)
                x_train_transformed = pipeline_fitted.transform(X_train_id)
                x_train_transformed_df = pd.DataFrame(x_train_transformed, columns=selected_features)
                x_test_transformed = pipeline_fitted.transform(X_val_id)
                x_test_transformed_df = pd.DataFrame(x_test_transformed, columns=selected_features)
                # x_train original dataframe
                x_train_original_df = pd.DataFrame(X_train_id, columns=selected_features)
                # x_test original dataframe
                x_test_original_df = pd.DataFrame(X_val_id, columns=selected_features)

                # # reinitialize the LD node with the selected features, and the other features already in the LD node
                ld_node.fit(x_train_original_df, x_train_transformed_df, y_train_id, snp_r2_dict)
                # data after LD node
                features_final = ld_node.selected_features_
                # print("Features after LD node: ", features_final, flush=True)
                features_final_column_names = [new_column_names_dict[feature] for feature in features_final]
                x_final_train = pd.DataFrame(x_train_transformed_df[features_final], columns=features_final)
                x_final_test = pd.DataFrame(x_test_transformed_df[features_final], columns=features_final)

                x_final_train.columns = features_final_column_names
                x_final_test.columns = features_final_column_names


        except ConvergenceWarning as cw:
            logging.error(f"ConvergenceWarning while fitting model: {cw}")
            logging.error(f"uni_nodes: {len(uni_nodes)}")
            # return an empty NDArray
            return np.array([], dtype=np.float32)
        except NotFittedError as nfe:
            logging.error(f"NotFittedError occurred: {nfe}")
            return np.array([], dtype=np.float32)
        except Exception as e:
            # Catch all other exceptions and log error with relevant context
            logging.error(f"Exception while fitting model: {e}")
            return np.array([], dtype=np.float32)

        # try:
        #      # create the pipeline
        #     steps = []
        #     # uni nodes into one sklearn union
        #     steps.append(('snp_union', FeatureUnion([(uni_node.name, uni_node) for uni_node in uni_nodes if uni_node.get_snp_name() in features_final])))
        #     # pass to regressor
        #     steps.append(('regressor', root_node.regressor))

        #     # create the pipeline without refitting the regressor
        #     pipeline = SklearnPipeline(steps=steps)
        # except Exception as e:
        #     logging.error(f"Exception while getting features to run SHAP: {e}")
        #     return pd.DataFrame()

        try:
            # getting the SHAP feature importance values
            number_of_features = len(features_final)
            print("Number of features from SHAP function: ", number_of_features, flush=True)
            max_evals = max(500, 2 * number_of_features + 1)
            #max_evals = 1200
            root_node.fit(x_final_train, y_train_id)
            explainer = shap.Explainer(root_node.predict, x_final_test)
            shap_values = explainer(x_final_test, max_evals=max_evals)
            # get the mean absolute shap values used in the summary plot
            shap_values = np.abs(shap_values.values).mean(0)
            shap_values_df = pd.DataFrame(list(zip(x_final_test.columns, shap_values)), columns=['feature', 'shap_value'])
            # sort the shap_values_df by the shap_value
            shap_values_df = shap_values_df.sort_values(by='shap_value', ascending=False)

        except Exception as e:
            logging.error(f"Exception while getting SHAP values: {e}")
            return pd.DataFrame()

        return x_final_test, shap_values_df, selector_name, root_name, id

class Poster:
    def __init__(self, X_train, y_train, X_val,  y_val, hub: snp_hub.SnpHub):
        self.X_train_id = X_train
        self.y_train_id = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.hub = hub

    def run_poster(self, pipeline, uni_nodes, X_train_id, y_train_id, X_val_id, y_val_id, id):
        # call the get_epi_pairs function to create the epi_pairs_df
        uni_snps_df = self.get_uni_snp(pipeline) # required to get the names of the interactions
        # get the snp_r2_dict
        snp_r2_dict = self.get_snp_hub_r2(pipeline)
        # create subsets
        results_refs = get_shap_values.remote(pipeline, uni_snps_df, uni_nodes, X_train_id, y_train_id, X_val_id, y_val_id, snp_r2_dict, id)
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

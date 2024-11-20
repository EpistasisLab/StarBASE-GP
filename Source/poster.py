# This class is responsible for all the post analysis functions
# todo: must focus on only univariate snps, not any epistatic snps

import shap
import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import ray
import logging
import warnings
from . import geno_hub

from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline as SklearnPipeline

# function to get the shap values
@ray.remote
def get_shap_values(pipeline, epi_pairs_df, epi_nodes, X_train_id, y_train_id, id) -> npt.NDArray[np.float32]:

        # create the pipeline to get the epi features created and selected by the selector before applying the root node
        steps = []

        # combine the epi nodes into a sklearn union
        steps.append(('epi_union', FeatureUnion([(epi_node.name, epi_node) for epi_node in epi_nodes])))
        # add the selector node
        steps.append(('selector', pipeline.get_selector_node()))
        selector_name = pipeline.get_selector_node().name

        root_node = pipeline.get_root_node()
        root_name = root_node.name

        if root_node is None:
            raise ValueError("Root node is None, ensure the pipeline is correctly returning a model.")

        transformer_pipeline = SklearnPipeline(steps=steps) # pipeline with the epi features and selector

        # attempt to fit the pipeline
        try:
            # Fit the pipeline with warnings captured as exceptions
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=ConvergenceWarning)
                pipeline_fitted = transformer_pipeline.fit(X_train_id, y_train_id)
        except ConvergenceWarning as cw:
            logging.error(f"ConvergenceWarning while fitting model: {cw}")
            logging.error(f"epi_nodes: {len(epi_nodes)}")
            # return an empty NDArray
            return np.array([], dtype=np.float32)
        except NotFittedError as nfe:
            logging.error(f"NotFittedError occurred: {nfe}")
            return np.array([], dtype=np.float32)
        except Exception as e:
            # Catch all other exceptions and log error with relevant context
            logging.error(f"Exception while fitting model: {e}")
            return np.array([], dtype=np.float32)

        try:
            # Transform X_train using the epi_union (before the selector)
            transformed_features = pipeline_fitted.named_steps['epi_union'].transform(X_train_id)

            # Get the mask of the selected features from the selector node
            support_mask = pipeline_fitted.named_steps['selector'].selector.get_support()

            # Apply the mask to the transformed features
            selected_features = transformed_features[:, support_mask]

            # Get epi pairs for naming columns
            column_names = [
            f'{row["feature1"]}_{row["interaction"]}_{row["feature2"]}'
            for _, row in epi_pairs_df.iterrows()
            ]

            # Convert column_names to a NumPy array to use with the support_mask
            column_names_selected = np.array(column_names)[support_mask]
            #print("Column names selected: ", column_names_selected)

            # Make sure the length of column names matches the selected features
            if len(column_names_selected) != selected_features.shape[1]:
                logging.error("Mismatch in number of selected features and column names")
                return pd.DataFrame()

            # Create a DataFrame with custom column names
            feature_df = pd.DataFrame(selected_features, columns=column_names_selected)

            # Print the size of the selected features
            print(f"Selected features size: {feature_df.shape}")
        except Exception as e:
            logging.error(f"Exception while getting features to run SHAP: {e}")
            return pd.DataFrame()

        try:
            # getting the SHAP feature importance values
            number_of_features = len(feature_df.columns)
            max_evals = max(500, 2 * number_of_features + 1)
            root_node.fit(feature_df, y_train_id)
            explainer = shap.Explainer(root_node.predict, feature_df)
            shap_values = explainer(feature_df)
            # get the mean absolute shap values used in the summary plot
            shap_values = np.abs(shap_values.values).mean(0)
            shap_values_df = pd.DataFrame(list(zip(feature_df.columns, shap_values)), columns=['feature', 'shap_value'])
            # sort the shap_values_df by the shap_value
            shap_values_df = shap_values_df.sort_values(by='shap_value', ascending=False)

        except Exception as e:
            logging.error(f"Exception while getting SHAP values: {e}")
            return pd.DataFrame()

        return feature_df, shap_values_df, selector_name, root_name, id


class Poster:
    def __init__(self, X_train, y_train, X_val,  y_val, hub: geno_hub.GenoHub):
        self.X_train_id = X_train
        self.y_train_id = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.hub = hub

    def run_poster(self, pipeline, epi_nodes, X_train_id, y_train_id, id):
        # call the get_epi_pairs function to create the epi_pairs_df
        epi_pairs_df = self.get_epi_pairs(pipeline) # required to get the names of the inetractions
        # create subsets
        results_refs = get_shap_values.remote(pipeline, epi_pairs_df, epi_nodes, X_train_id, y_train_id, id)
        return results_refs


    def get_epi_pairs(self, pipeline):

        # define an empty list to store the epi_pairs
        epi_pairs_list = []

        # get the epi pairs in the pipeline
        epi_pairs = pipeline.get_epi_pairs() # set of tuples

        for snp_pair in epi_pairs:
            snp1, snp2 = snp_pair
            # get the best lo for the epi_pairs
            best_lo = self.hub.get_interaction_lo(snp1, snp2)

            # Add the interaction to the epi_pairs_list as a dictionary
            epi_pairs_list.append({'feature1': snp1, 'feature2': snp2, 'interaction': best_lo})

        # convert the epi_pairs_list to a dataframe
        epi_pairs_df = pd.DataFrame(epi_pairs_list)

        return epi_pairs_df

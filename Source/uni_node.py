# This script has the implementation of univariate encoders
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

class UniNode(BaseEstimator, TransformerMixin, ABC):
    def __init__(self, name: np.str_, snp_name: np.str_, snp_pos: np.uint32):
        self.name = name
        self.snp_name = snp_name
        self.snp_pos = snp_pos

        self.encoded_feature = None # store the output of process_data, the actual encoded feature
        self.mapping = None # required for PAGER
        self.epi_feature_name = None
        self.fit_flag = False

    @abstractmethod
    def fit(self, X, y=None):
        pass

    def print_encoded_feature(self):
        print(self.encoded_feature)

    def transform(self, X):
        pass

    def get_encoding(self) -> np.str_:
        return np.str_(f"{self.get_snp_name}_{self.get_encoder}")

    @abstractmethod
    def get_encoder(self) -> np.str_:
        pass

    def get_snp_name(self) -> np.str_:
        return self.snp_name

    def get_node_name(self) -> np.str_:
        return self.name

    @abstractmethod
    def predict(self, X):
        pass

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class UniAdditiveNode(UniNode):
    def fit(self, X, y=None):
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]

        # no changes required for the snp
        self.mapping = {0:0, 0.5: 0.5, 1: 1} # will be used in the transform and predict methods

        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        # mark the node as fitted
        self.fit_flag = True

        return self

    def transform(self, X):
        # always recompute the encoded univariate feature, regardless of train or validation data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]

        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        # return the computed feature for this dataset
        return np.array(snp.astype(np.float32), dtype=np.float32).reshape(-1, 1)

    def predict(self, X):
        # does the same operation as fit but with the test data
        if self.fit_flag == False:
            raise ValueError("Model not fitted yet. Please fit the model first")
                # Get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]


        snp = snp.replace(self.mapping)

        #store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        return self.encoded_feature.reshape(-1, 1)

    def get_encoder(self) -> np.str_:
        return np.str_("additive")

class UniDominantNode(UniNode):
    def fit(self, X, y=None):
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]

        # change all the 0.5 to 1
        self.mapping = {0:0, 0.5: 1, 1: 1} # will be used in the transform and predict methods

        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        # mark the node as fitted
        self.fit_flag = True

        return self

    def transform(self, X):
        # always recompute the encoded univariate feature, regardless of train or validation data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]

        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        # return the computed feature for this dataset
        return np.array(snp.astype(np.float32), dtype=np.float32).reshape(-1, 1)

    def predict(self, X):
        # does the same operation as fit but with the test data
        if self.fit_flag == False:
            raise ValueError("Model not fitted yet. Please fit the model first")
                # Get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]


        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        return self.encoded_feature.reshape(-1, 1)

    def get_encoder(self) -> np.str_:
        return np.str_("dominant")

class UniRecessiveNode(UniNode):
    def fit(self, X, y=None):
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]

        # change all the 0.5 to 0
        self.mapping = {0:0, 0.5: 0, 1: 1} # will be used in the transform and predict methods

        snp = snp.replace(self.mapping)

        # Store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        # Mark the node as fitted
        self.fit_flag = True

        return self

    def transform(self, X):
        # always recompute the encoded univariate feature, regardless of train or validation data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]

        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        # return the computed feature for this dataset
        return np.array(snp.astype(np.float32), dtype=np.float32).reshape(-1, 1)

    def predict(self, X):
        # does the same operation as fit but with the test data
        if self.fit_flag == False:
            raise ValueError("Model not fitted yet. Please fit the model first")
                # Get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]

        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        return self.encoded_feature.reshape(-1, 1)

    def get_encoder(self) -> np.str_:
        return np.str_("recessive")

class UniHeterosisNode(UniNode):
    def fit(self, X, y=None):
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]

        # change all the 0.5 to 1 and 1 to 0 in snp simultaneously
        self.mapping = {0:0, 0.5: 1, 1: 0} # will be used in the transform and predict methods

        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        # mark the node as fitted
        self.fit_flag = True

        return self

    def transform(self, X):
        # always recompute the encoded univariate feature, regardless of train or validation data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]

        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        # return the computed feature for this dataset
        return np.array(snp.astype(np.float32), dtype=np.float32).reshape(-1, 1)

    def predict(self, X):
        # does the same operation as fit but with the test data
        if self.fit_flag == False:
            raise ValueError("Model not fitted yet. Please fit the model first")
                # Get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]


        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        return self.encoded_feature.reshape(-1, 1)

    def get_encoder(self) -> np.str_:
        return np.str_("heterosis")

class UniUnderDominantNode(UniNode):
    def fit(self, X, y=None):
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]

        # change all the 0 to 0.5, 0.5 to 0
        self.mapping = {0:0.5, 0.5: 0, 1: 1} # will be used in the transform and predict methods

        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        # mark the node as fitted
        self.fit_flag = True

        return self

    def transform(self, X):
        # always recompute the encoded univariate feature, regardless of train or validation data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]

        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        # return the computed feature for this dataset
        return np.array(snp.astype(np.float32), dtype=np.float32).reshape(-1, 1)

    def predict(self, X):
        # does the same operation as fit but with the test data
        if self.fit_flag == False:
            raise ValueError("Model not fitted yet. Please fit the model first")
                # Get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]


        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        return self.encoded_feature.reshape(-1, 1)

    def get_encoder(self) -> np.str_:
        return np.str_("underdominant")
    
class UniOverDominantNode(UniNode):
    def fit(self, X, y=None):
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]

        # change all the 0 to 0, 0.5 to 1 and 1 to 0.5
        self.mapping = {0:0, 0.5: 1, 1: 0.5} # will be used in the transform and predict methods

        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        # mark the node as fitted
        self.fit_flag = True

        return self

    def transform(self, X):
        # always recompute the encoded univariate feature, regardless of train or validation data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]

        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        # return the computed feature for this dataset
        return np.array(snp.astype(np.float32), dtype=np.float32).reshape(-1, 1)

    def predict(self, X):
        # does the same operation as fit but with the test data
        if self.fit_flag == False:
            raise ValueError("Model not fitted yet. Please fit the model first")
                # Get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]


        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        return self.encoded_feature.reshape(-1, 1)

    def get_encoder(self) -> np.str_:
        return np.str_("overdominant")

class UniSubAdditiveNode(UniNode):
    def fit(self, X, y=None):
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]

        # change all the 0.5 to 0.25
        self.mapping = {0:0, 0.5: 0.25, 1: 1} # will be used in the transform and predict methods

        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        # mark the node as fitted
        self.fit_flag = True

        return self

    def transform(self, X):
        # always recompute the encoded univariate feature, regardless of train or validation data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]


        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        # return the computed feature for this dataset
        return np.array(snp.astype(np.float32), dtype=np.float32).reshape(-1, 1)

    def predict(self, X):
        # does the same operation as fit but with the test data
        if self.fit_flag == False:
            raise ValueError("Model not fitted yet. Please fit the model first")
                # Get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]


        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        return self.encoded_feature.reshape(-1, 1)

    def get_encoder(self) -> np.str_:
        return np.str_("subadditive")

class UniSuperAdditiveNode(UniNode):
    def fit(self, X, y=None):
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]

        # change all the 0.5 to 0.75
        self.mapping = {0:0, 0.5: 0.75, 1: 1} # will be used in the transform and predict methods

        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        # Mark the node as fitted
        self.fit_flag = True

        return self

    def transform(self, X):
        # always recompute the encoded univariate feature, regardless of train or validation data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]

        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        # return the computed feature for this dataset
        return np.array(snp.astype(np.float32), dtype=np.float32).reshape(-1, 1)

    def predict(self, X):
        # does the same operation as fit but with the test data
        if self.fit_flag == False:
            raise ValueError("Model not fitted yet. Please fit the model first")
                # Get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp = X.iloc[:, self.snp_pos]
        else:
            snp = X[:, self.snp_pos]

        snp = snp.replace(self.mapping)

        # store the result in self.encoded_feature for training data
        self.encoded_feature = np.array(snp.astype(np.float32), dtype=np.float32)

        return self.encoded_feature.reshape(-1, 1)

    def get_encoder(self) -> np.str_:
        return np.str_("superadditive")

# Need to change the PAGER implementation due to training/testing data
class UniPAGERNode(UniNode):
    def fit(self, X, y=None):
        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp_train = X.iloc[:, self.snp_pos]
        else:
            snp_train = X[:, self.snp_pos]
        # create a DataFrame with the snp columns and the phenotype
        train_data = pd.DataFrame({
            'snp': snp_train,
            'Phenotype': y
        })

        # calculate the mean phenotype for each snp pair
        geno_aggregations_train = train_data.groupby(['snp']).agg(
            mean_phenotype=('Phenotype', 'mean')
        ).reset_index()

        # calculate the relative distance of the mean phenotype to the anchor
        anchor_mean = geno_aggregations_train.loc[geno_aggregations_train['snp'].idxmin(), 'mean_phenotype'] # anchor = 0 when all three genotypes are present
        geno_aggregations_train['rel_dist'] = (geno_aggregations_train['mean_phenotype'] - (anchor_mean))

        # normalize the relative distance
        scaler_train = MinMaxScaler()
        geno_aggregations_train['normalized_rel_dist'] = scaler_train.fit_transform(
            geno_aggregations_train['rel_dist'].values.reshape(-1, 1)
        )

        # store the PAGER values generated by using the train data
        self.mapping = geno_aggregations_train[['snp', 'normalized_rel_dist']] # store the PAGER values generated by using the train data

        train_data = pd.merge(train_data, geno_aggregations_train, on=['snp'], how='left')
        pager_encoded_interactions_train = train_data['normalized_rel_dist'].values
        self.encoded_feature = np.nan_to_num(pager_encoded_interactions_train, nan=0.5).astype(np.float32)
        self.fit_flag = True

        return self

    def transform(self, X):
        if self.fit_flag == False or self.mapping is None:
            raise ValueError("Mapping not set. Please fit the model first.")

        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp_train = X.iloc[:, self.snp_pos]
        else:
            snp_train = X[:, self.snp_pos]

        train_data = pd.DataFrame({
            'snp': snp_train,
        })

        train_data = pd.merge(train_data, self.mapping, on=['snp'], how='left') # using the mapping generated by the train data
        pager_encoded_interactions_train = train_data['normalized_rel_dist'].values
        self.encoded_feature = np.nan_to_num(pager_encoded_interactions_train, nan=0.5).astype(np.float32)

        return self.encoded_feature.reshape(-1, 1)

    def predict(self, X):
        if self.fit_flag == False or self.mapping is None:
            raise ValueError("Mapping not set. Please fit the model first.")

        # get the snp columns from the input data
        if isinstance(X, pd.DataFrame):
            snp_test = X.iloc[:, self.snp1_pos]
        else:
            snp_test = X[:, self.snp1_pos]

        test_data = pd.DataFrame({
            'snp': snp_test,
        })
        test_data = pd.merge(test_data, self.mapping, on=['snp'], how='left') # using the mapping generated by the train data
        pager_encoded_interactions_test = test_data['normalized_rel_dist'].values
        self.encoded_feature = np.nan_to_num(pager_encoded_interactions_test, nan=0.5).astype(np.float32)

        return self.encoded_feature.reshape(-1, 1)

    def get_encoder(self) -> np.str_:
        return np.str_("pager")
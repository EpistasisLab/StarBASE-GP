# This class is a wrapper for the scikit-learn library. It provides a set of nodes (regressors and feature selectors) that can be used in the pipeline.
# Will contain a abstract class and then the various regressors and feature selectors will be implemented as subclasses of this abstract class.

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, SelectFwe, SelectFromModel, SequentialFeatureSelector, f_regression
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from typeguard import typechecked
from typing import Dict, List
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

rng_t = np.random.Generator
name_t = np.str_

class ScikitNode(BaseEstimator, ABC):
    def __init__(self, name, features=None):
        self.name = name
        self.features = features if features is not None else []

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    @abstractmethod
    def mutate(self, rng_):
        pass

    def get_feature_names(self, feature_names):
        # Ensure feature_names is a NumPy array
        feature_names = np.array(feature_names)

        # Ensure the length of feature_names matches the number of features in the data
        support_mask = self.selector.get_support()
        if len(feature_names) != len(support_mask):
            raise ValueError("Length of feature_names does not match the number of features in the data.")

        # Print debug information
        # print("Length of feature names: ", len(feature_names))
        # print("Mask: ", support_mask)
        # print("Mask type: ", type(support_mask))
        # print("Length of mask: ", len(support_mask))

        # Use the Boolean mask to filter feature names
        return feature_names[support_mask]

##########################################################################################
########################## the feature selector classes ##################################
##########################################################################################

# variance threshold
@typechecked
class VarianceThresholdNode(ScikitNode, TransformerMixin):
    def __init__(self,
                 rng_: rng_t,
                 params: Dict = {},
                 name: name_t = name_t('VarianceThreshold')):
        super().__init__(name)
        rng = np.random.default_rng(rng_)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'threshold': np.float32(rng.uniform(low=0.0001, high=0.05))} # high is set low to allow models to have a chance to learn
        else:
            # make sure params is correct
            assert 'threshold' in params
            assert len(params) == 1
            assert isinstance(params['threshold'], np.float32)
            self.params = params

        self.selector = VarianceThreshold(threshold=self.params['threshold'])

    def fit(self, X, y=None):
        self.selector.fit(X)

    def transform(self, X):
        return self.selector.transform(X)

    def mutate(self, rng_: rng_t):
        rng = np.random.default_rng(rng_)

        # get a random number from a normal distribution
        shift = np.float32(rng.normal(loc=0.0, scale=0.025))

        # check if the threshold is going to be negative
        if self.params['threshold'] + shift < np.float32(0.0):
            self.params['threshold'] = np.float32(0.0)
        # check if the threshold is going to be greater than 1
        elif self.params['threshold'] + shift > np.float32(1.0):
            self.params['threshold'] = np.float32(1.0)
        # if neither of the above, then we can just add the shift
        else:
            self.params['threshold'] = self.params['threshold'] + shift

        self.selector = VarianceThreshold(threshold=self.params['threshold'])

    def get_feature_count(self):
        return self.selector.get_support().sum()

    # def get_feature_names(self, feature_names):
    #     # # Ensure feature_names is a NumPy array
    #     # feature_names = np.array(feature_names)
    #     # # Ensure the length matches the number of features in the original input
    #     # if len(feature_names) != len(self.selector.get_support()):
    #     #     raise ValueError("Length of feature_names does not match the number of features in the data.")

    #     # print the length of the feature names
    #     print("Length of feature names: ", len(feature_names))
    #     # print the lenth of filtered feature names
    #     print("Length of filtered feature names: ", len(feature_names[self.selector.get_support()]))

    #     # Use the Boolean mask to filter feature names
    #     return feature_names[self.selector.get_support()]

# select percentile
class SelectPercentileNode(ScikitNode, TransformerMixin):
    def __init__(self,
                 rng_: rng_t,
                 params: Dict = {},
                 name: name_t = name_t('SelectPercentile')):
        super().__init__(name)
        rng = np.random.default_rng(rng_)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'percentile': np.int8(rng.integers(low=50, high=100)), 'score_func': f_regression}
        else:
            # make sure params is correct
            assert 'percentile' in params
            assert 'score_func' in params
            assert len(params) == 2
            assert isinstance(params['percentile'], np.int8)
            assert isinstance(params['score_func'], np.ufunc)
            self.params = params

        self.selector = SelectPercentile(score_func=self.params['score_func'], percentile=self.params['percentile'])

    def fit(self, X, y):
        self.selector.fit(X, y)

    def transform(self, X):
        # self.selector.transform(X)
        # self.features = self.selector.get_support()
        return self.selector.transform(X)

    def mutate(self, rng_: rng_t):
        rng = np.random.default_rng(rng_)

        # maginitude we are shfiting the percentile
        shift = np.int8(rng.integers(low=-5, high=5, endpoint=True))

        # check if the percentile is going to be less than 1
        if self.params['percentile'] + shift < np.int8(1):
            self.params['percentile'] = np.int8(1)
        # check if the percentile is going to be greater than 100
        elif self.params['percentile'] + shift > np.int8(100):
            self.params['percentile'] = np.int8(100)
        # if neither of the above, then we can just add the shift
        else:
            self.params['percentile'] = self.params['percentile'] + shift

        self.selector = SelectPercentile(score_func=self.params['score_func'], percentile=self.params['percentile'])

    def get_feature_count(self):
        return self.selector.get_support().sum()

    # def get_feature_names(self, feature_names):
    #     # # Ensure feature_names is a NumPy array
    #     # feature_names = np.array(feature_names)
    #     # # Ensure the length matches the number of features in the original input
    #     # if len(feature_names) != len(self.selector.get_support()):
    #     #     raise ValueError("Length of feature_names does not match the number of features in the data.")

    #     # print the length of the feature names
    #     print("Length of feature names: ", len(feature_names))
    #     # print the lenth of filtered feature names
    #     print("Length of filtered feature names: ", len(feature_names[self.selector.get_support()]))

    #     # Use the Boolean mask to filter feature names
    #     return feature_names[self.selector.get_support()]

# select fwe
class SelectFweNode(ScikitNode, TransformerMixin):
    def __init__(self,
                 rng_: rng_t,
                 params: Dict = {},
                 name: name_t = name_t('SelectFwe')):
        super().__init__(name)
        rng = np.random.default_rng(rng_)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'alpha': np.float32(rng.uniform(low=1e-4, high=0.05)), 'score_func': f_regression}
        else:
            # make sure params is correct
            assert 'alpha' in params
            assert 'score_func' in params
            assert len(params) == 2
            assert isinstance(params['alpha'], np.float32)
            assert isinstance(params['score_func'], np.ufunc)
            self.params = params

        self.selector = SelectFwe(score_func=self.params['score_func'], alpha=self.params['alpha'])

    def fit(self, X, y):
        self.selector.fit(X, y)

    def transform(self, X):
        return self.selector.transform(X)

    def mutate(self, rng_: rng_t):
        rng = np.random.default_rng(rng_)

        # get a random number from a normal distribution
        shift = np.float32(rng.normal(loc=0.0, scale=0.005))

        # check if the alpha is going to be negative
        if self.params['alpha'] + shift < np.float32(0.0001):
            self.params['alpha'] = np.float32(0.0001)
        # check if the alpha is going to be greater than .99
        elif self.params['alpha'] + shift > np.float32(0.99):
            self.params['alpha'] = np.float32(0.99)
        # if neither of the above, then we can just add the shift
        else:
            self.params['alpha'] = self.params['alpha'] + shift

        # new selector configuration
        self.selector = SelectFwe(score_func=self.params['score_func'], alpha=self.params['alpha'])

    def get_feature_count(self):
        return self.selector.get_support().sum()

    # def get_feature_names(self, feature_names):
    #     # Ensure feature_names is a NumPy array
    #     feature_names = np.array(feature_names)
    #     # Ensure the length matches the number of features in the original input
    #     if len(feature_names) != len(self.selector.get_support()):
    #         raise ValueError("Length of feature_names does not match the number of features in the data.")

    #     # print the length of the feature names
    #     print("Length of feature names: ", len(feature_names))
    #     # print the lenth of filtered feature names
    #     print("Length of filtered feature names: ", len(feature_names[self.selector.get_support()]))

    #     # Use the Boolean mask to filter feature names
    #     return feature_names[self.selector.get_support()]

# select from model using L1-based feature selection (model is lasso regression)
class SelectFromModelLasso(ScikitNode, TransformerMixin):
    def __init__(self,
                 rng_: rng_t,
                 seed: int = -1,
                 params: Dict = {},
                 name: name_t = name_t('SelectFromLasso')):
        super().__init__(name)
        rng = np.random.default_rng(rng_)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'estimator': Lasso(random_state=seed), 'threshold': rng.choice([name_t('mean'), name_t('median')])}
        else:
            # make sure params is correct
            assert 'estimator' in params
            assert 'threshold' in params
            assert len(params) == 2
            assert isinstance(params['estimator'], Lasso)
            assert isinstance(params['threshold'], np.str_)
            self.params = params

        self.selector = SelectFromModel(estimator = self.params['estimator'], threshold=self.params['threshold'])

    def fit(self, X, y):
        self.selector.fit(X, y)

    def transform(self, X):
        return self.selector.transform(X)

    def mutate(self, rng_):
        rng = np.random.default_rng(rng_)

        # randomly select threshold
        self.params['threshold'] = rng.choice([name_t('mean'), name_t('median')])
        # new selector configuration
        self.selector = SelectFromModel(estimator = self.params['estimator'], threshold=self.params['threshold'])

    def get_feature_count(self,):
        return self.selector.get_support().sum()

    # def get_feature_names(self, feature_names):
    #     # Ensure feature_names is a NumPy array
    #     feature_names = np.array(feature_names)
    #     # Ensure the length matches the number of features in the original input
    #     if len(feature_names) != len(self.selector.get_support()):
    #         raise ValueError("Length of feature_names does not match the number of features in the data.")

    #     # print the length of the feature names
    #     print("Length of feature names: ", len(feature_names))
    #     # print the lenth of filtered feature names
    #     print("Length of filtered feature names: ", len(feature_names[self.selector.get_support()]))

    #     # Use the Boolean mask to filter feature names
    #     return feature_names[self.selector.get_support()]

# select from model using tree-based feature selection (model is ExtraTreesRegressor)
class SelectFromModelTree(ScikitNode, TransformerMixin):
    def __init__(self,
                 rng_: rng_t,
                 seed: int = -1,
                 params: Dict = {},
                 name: name_t = name_t('SelectFromExtraTrees')):
        super().__init__(name)
        rng = np.random.default_rng(rng_)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'estimator': ExtraTreesRegressor(random_state=seed), 'threshold': rng.choice([name_t('mean'), name_t('median')])}
        else:
            # make sure params is correct
            assert 'estimator' in params
            assert 'threshold' in params
            assert len(params) == 2
            assert isinstance(params['estimator'], ExtraTreesRegressor)
            assert isinstance(params['threshold'], np.str_)
            self.params = params

        self.selector = SelectFromModel(estimator = self.params['estimator'], threshold=self.params['threshold'])

    def fit(self, X, y):
        self.selector.fit(X, y)

    def transform(self, X):
        return self.selector.transform(X)

    def mutate(self, rng_):
        rng = np.random.default_rng(rng_)
        # randomly select threshold
        self.params['threshold'] = rng.choice([name_t('mean'), name_t('median')])
        # new selector configuration
        self.selector = SelectFromModel(estimator = self.params['estimator'], threshold=self.params['threshold'])

    def get_feature_count(self):
        return self.selector.get_support().sum()

    # def get_feature_names(self, feature_names):
    #     # Ensure feature_names is a NumPy array
    #     feature_names = np.array(feature_names)
    #     # Ensure the length matches the number of features in the original input
    #     if len(feature_names) != len(self.selector.get_support()):
    #         raise ValueError("Length of feature_names does not match the number of features in the data.")

    #     # print the length of the feature names
    #     print("Length of feature names: ", len(feature_names))
    #     # print the lenth of filtered feature names
    #     print("Length of filtered feature names: ", len(feature_names[self.selector.get_support()]))

    #     # Use the Boolean mask to filter feature names
    #     return feature_names[self.selector.get_support()]

# sequential feature selector, model = RandomForestRegressor
class SequentialFeatureSelectorNode(ScikitNode, TransformerMixin):
    def __init__(self,
                 rng_: rng_t,
                 seed: int = -1,
                 params: Dict = {},
                 name: name_t = name_t('SequentialFeatureSelectorRF')):
        super().__init__(name)
        rng = np.random.default_rng(rng_)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'estimator': RandomForestRegressor(random_state=seed), 'tol': np.float32(rng.uniform(low=1e-5, high=0.5))}
        else:
            assert 'estimator' in params
            assert 'tol' in params
            assert len(params) == 2
            assert isinstance(params['estimator'], RandomForestRegressor)
            assert isinstance(params['tol'], np.float32)
            self.params = params

        self.selector = SequentialFeatureSelector(estimator=self.params['estimator'], tol=self.params['tol'], cv=5)

    def fit(self, X, y):
        self.selector.fit(X, y)

    def transform(self, X):
        return self.selector.transform(X)

    def mutate(self, rng_):
        rng = np.random.default_rng(rng_)

        # get a random number from a normal distribution
        shift = np.float32(rng.normal(loc=0.0, scale=0.05))

        # check if the tol is going to be less than 1e-5
        if self.params['tol'] + shift < np.float32(1e-5):
            self.params['tol'] = np.float32(1e-5)
        # check if the tol is going to be greater than 0.5
        elif self.params['tol'] + shift > np.float32(0.5):
            self.params['tol'] = np.float32(0.5)
        # if neither of the above, then we can just add the shift
        else:
            self.params['tol'] = self.params['tol'] + shift

        # new selector configuration
        self.selector = SequentialFeatureSelector(estimator=self.params['estimator'], tol=self.params['tol'], cv=5)

    def get_feature_count(self):
        return self.selector.get_support().sum()

    # def get_feature_names(self, feature_names):
    #     # Ensure feature_names is a NumPy array
    #     feature_names = np.array(feature_names)
    #     # Ensure the length matches the number of features in the original input
    #     if len(feature_names) != len(self.selector.get_support()):
    #         raise ValueError("Length of feature_names does not match the number of features in the data.")

    #     # print the length of the feature names
    #     print("Length of feature names: ", len(feature_names))
    #     # print the lenth of filtered feature names
    #     print("Length of filtered feature names: ", len(feature_names[self.selector.get_support()]))

    #     # Use the Boolean mask to filter feature names
    #     return feature_names[self.selector.get_support()]

# custom feature selector based on feature encoding frequency
class FeatureEncodingFrequencySelector(ScikitNode, TransformerMixin):
    """Feature selector based on Encoding Frequency. Encoding frequency is the frequency of each unique element(0/1/2/3) present in a feature set.
     Features are selected on the basis of a threshold assigned for encoding frequency. If frequency of any unique element is less than or equal to threshold,
     the feature is removed.  """
    def __init__(self,
                 rng_: rng_t,
                 seed: int = -1,
                 params: Dict = {},
                 name: name_t = name_t('FeatureEncodingFrequencySelector')):
        super().__init__(name)
        # if params is an empty dictionary, then we will initialize the params

        rng = np.random.default_rng(rng_)

        if params == {}:
            self.params = {'threshold': np.float32(rng.uniform(low=0.0, high=0.3))} # increments of 0.05
        else:
            # make sure params is correct
            assert 'threshold' in params
            assert len(params) == 1
            assert isinstance(params['threshold'], np.float32)
            self.params = params
        self.threshold = self.params['threshold']
        self.seed = seed
        self.selector = 'FeatureEncodingFrequencySelector'
        self.boolean_mask = None

    def fit(self, X, y=None):
        """
        Fit the feature selector to the data.
        Parameters:
        - X (array-like): The input features (2D array of shape [n_samples, n_features]).
        - y (ignored): The target variable (not used in this selector).
        Returns:
        - self: The fitted selector.
        """
        X = np.asarray(X)  # Ensure input is a numpy array
        n_samples, n_features = X.shape
        selected_features = []
        for i in range(n_features):
            unique_values, counts = np.unique(X[:, i], return_counts=True)
            frequencies = counts / n_samples
            if np.all(frequencies >= self.threshold):
                selected_features.append(i)
        self.selected_features_ = np.array(selected_features)
        self.selected_features_ = np.array(self.selected_features_, dtype=int)
        # make a boolean mask of the selected features
        self.boolean_mask = np.zeros(X.shape[1], dtype=bool)
        self.boolean_mask[self.selected_features_] = True
        # print("Boolean mask: ", self.boolean_mask)
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
            raise RuntimeError("FeatureEncodingFrequencySelector has not been fitted yet.")
        X = np.asarray(X)  # Ensure input is a numpy array

        if X.shape[1] != len(self.boolean_mask):
            raise ValueError("Number of features in X does not match the number of features in the selector.")

        return X[:, self.boolean_mask]

    def mutate(self, rng: rng_t):
        # shift is a rng from normal distribution with a change in 2nd decimal place
        shift = np.float32(rng.normal(loc=0.0, scale=0.01))
        # check if the threshold is going to be less than 0.0
        if self.threshold + shift < np.float32(0.0):
            self.threshold = np.float32(0.0)
        # check if the threshold is going to be greater than 0.3
        elif self.threshold + shift > np.float32(0.3):
            self.threshold = np.float32(0.3)
        self.threshold = self.params['threshold']

    def get_feature_count(self):
        """
            Get the number of features selected by the selector.
            Returns:
            - int: The number of features selected. If the selector has not been fitted yet,
           raises a RuntimeError.
        """
        if self.selected_features_ is None:
            raise RuntimeError("FeatureEncodingFrequencySelector has not been fitted yet.")
        return len(self.selected_features_)

    def get_feature_names(self, feature_names):
        """
            Get the names of the features selected by the selector.
            Parameters:
            - feature_names (array-like): The names of the features.
            Returns:
            - array-like: The names of the selected features.
        """
        if self.selected_features_ is None:
            raise RuntimeError("FeatureEncodingFrequencySelector has not been fitted yet.")
        # # Ensure feature_names is a NumPy array
        # feature_names = np.array(feature_names)
        # # Ensure the length matches the number of features in the original input
        # if len(feature_names) != len(self.boolean_mask):
        #     raise ValueError("Length of feature_names does not match the number of features in the data.")

        # # print the length of the feature names
        # print("Length of feature names: ", len(feature_names))
        # print('Type of feature names: ', type(feature_names))
        # print('Type of boolean mask: ', type(self.boolean_mask))
        # print the lenth of filtered feature names
        #print("Length of filtered feature names: ", len(feature_names[self.selector.get_support()]))

        # Use the Boolean mask to filter feature names
        final_features = []
        for i in range(len(self.boolean_mask)):
            if self.boolean_mask[i]:
                final_features.append(feature_names[i])

        return final_features

##########################################################################################
############################ the regressor classes #######################################
##########################################################################################

# Linear regression
class LinearRegressionNode(ScikitNode, RegressorMixin):
    def __init__(self,
                 rng_: rng_t,
                 params: Dict = {},
                 name: name_t = name_t('LinearRegression')):
        super().__init__(name)
        rng = np.random.default_rng(rng_)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'fit_intercept': rng.choice([True, False])}
        else:
            assert len(params) == 1
            assert 'fit_intercept' in params
            self.params = params

        self.regressor = LinearRegression(fit_intercept=self.params['fit_intercept'])

    def fit(self, X, y):
        self.regressor.fit(X, y)
        return self.regressor

    def predict(self, X):
        return self.regressor.predict(X)

    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)

    def mutate(self, rng_: rng_t):
        rng = np.random.default_rng(rng_)

        # randomly pick fit_intercept
        self.params['fit_intercept'] = rng.choice([True, False])

        # new regressor configuration
        self.regressor = LinearRegression(fit_intercept=self.params['fit_intercept'])

# ElasticNet regression
class ElasticNetNode(ScikitNode, RegressorMixin):
    def __init__(self,
                 rng_: rng_t,
                 seed: int = -1,
                 params: Dict = {},
                 name: name_t = name_t('LinearRegression')):
        super().__init__(name)
        rng = np.random.default_rng(rng_)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            # l1_ratio should not be 0 or 1, use Lasso or Ridge instead
            self.params = {'alpha': np.float32(rng.uniform(low=0.0001, high=10.00)),
                          'l1_ratio': np.float32(rng.uniform(low=0.02, high=0.98)),
                          'fit_intercept': rng.choice([True, False]),
                          'selection': rng.choice([name_t('cyclic'), name_t('random')]),
                          'random_state': seed}
        else:
            assert len(params) == 5
            assert 'alpha' in params
            assert 'l1_ratio' in params
            assert 'fit_intercept' in params
            assert 'selection' in params
            assert 'random_state' in params
            self.params = params

        self.regressor = ElasticNet(**self.params)

    def fit(self, X, y):
        return self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)

    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)

    def mutate(self, rng):
        # get a random number from a normal distribution
        alpha_shift = np.float32(rng.normal(loc=0.0, scale=1.0))

        # check if the alpha is going to be less than 0.0001
        if self.params['alpha'] + alpha_shift < np.float32(0.0001):
            self.params['alpha'] = np.float32(0.0001)
        # check if the alpha is going to be greater than 10
        elif self.params['alpha'] + alpha_shift > np.float32(10.0):
            self.params['alpha'] = np.float32(10.0)
        # if neither of the above, then we can just add the shift
        else:
            self.params['alpha'] = self.params['alpha'] + alpha_shift

        # get a random number from a normal distribution
        l1_ratio_shift = np.float32(rng.normal(loc=0.0, scale=0.05))

        # check if the l1_ratio is going to be less than 0.02
        if self.params['l1_ratio'] + l1_ratio_shift < np.float32(0.02):
            self.params['l1_ratio'] = np.float32(0.02)
        # check if the l1_ratio is going to be greater than 0.98
        elif self.params['l1_ratio'] + l1_ratio_shift > np.float32(0.98):
            self.params['l1_ratio'] = np.float32(0.98)
        # if neither of the above, then we can just add the shift
        else:
            self.params['l1_ratio'] = self.params['l1_ratio'] + l1_ratio_shift

        # randomly pick fit_intercept
        self.params['fit_intercept'] = rng.choice([True, False])
        # randomly pick selection
        self.params['selection'] = rng.choice([name_t('cyclic'), name_t('random')])

        # new regressor configuration
        self.regressor = ElasticNet(**self.params)

# SGD regression
class SGDRegressorNode(ScikitNode, RegressorMixin):
    def __init__(self,
                 rng_: rng_t,
                 seed: int = -1,
                 params: Dict = {},
                 name: name_t = name_t('SGDRegressor')):
        super().__init__(name)
        rng = np.random.default_rng(rng_)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'loss': rng.choice(['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive']),
                           'penalty': rng.choice(['l2','l1','elasticnet',None]),
                           'alpha': np.float32(rng.uniform(low=0.0001, high=10.00)),
                           'l1_ratio': np.float32(rng.uniform(low=0.02, high=0.98)),
                           'fit_intercept': rng.choice([True, False]),
                           'epsilon': np.float32(rng.uniform(low=0.0001, high=10.00)),
                           'learning_rate': rng.choice(['constant','optimal','invscaling','adaptive']),
                           'eta0': np.float32(rng.uniform(low=1e-7, high=0.01)),
                           'random_state': seed}
        else:
            assert len(params) == 9
            assert 'alpha' in params
            assert 'l1_ratio' in params
            assert 'epsilon' in params
            assert 'loss' in params
            assert 'eta0' in params
            assert 'random_state' in params
            assert 'fit_intercept' in params
            assert 'penalty' in params
            assert 'learning_rate' in params
            self.params = params

        self.regressor = SGDRegressor(**self.params)

    def fit(self, X, y):
        return self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)

    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)

    def mutate(self, rng_):
        rng = np.random.default_rng(rng_)

        # get a random number from a normal distribution
        alpha_shift = np.float32(rng.normal(loc=0.0, scale=1.0))

        # check if the alpha is going to be less than 0.0001
        if self.params['alpha'] + alpha_shift < np.float32(0.0001):
            self.params['alpha'] = np.float32(0.0001)
        # check if the alpha is going to be greater than 10
        elif self.params['alpha'] + alpha_shift > np.float32(10.0):
            self.params['alpha'] = np.float32(10.0)
        # if neither of the above, then we can just add the shift
        else:
            self.params['alpha'] = self.params['alpha'] + alpha_shift

        # get a random number from a normal distribution
        l1_ratio_shift = np.float32(rng.normal(loc=0.0, scale=0.05))

        # check if the l1_ratio is going to be less than 0.02
        if self.params['l1_ratio'] + l1_ratio_shift < np.float32(0.02):
            self.params['l1_ratio'] = np.float32(0.02)
        # check if the l1_ratio is going to be greater than 0.98
        elif self.params['l1_ratio'] + l1_ratio_shift > np.float32(0.98):
            self.params['l1_ratio'] = np.float32(0.98)
        # if neither of the above, then we can just add the shift
        else:
            self.params['l1_ratio'] = self.params['l1_ratio'] + l1_ratio_shift

        # get a random number from a normal distribution
        epsilon_shift = np.float32(rng.normal(loc=0.0, scale=1.0))

        # check if the epsilon is going to be less than 0.0001
        if self.params['epsilon'] + epsilon_shift < np.float32(0.0001):
            self.params['epsilon'] = np.float32(0.0001)
        # check if the epsilon is going to be greater than 10
        elif self.params['epsilon'] + epsilon_shift > np.float32(10.0):
            self.params['epsilon'] = np.float32(10.0)
        # if neither of the above, then we can just add the shift
        else:
            self.params['epsilon'] = self.params['epsilon'] + epsilon_shift

        # get a random number from a normal distribution
        eta0_shift = np.float32(rng.normal(loc=0.0, scale=1.0))

        # check if the eta0 is going to be less than 0.0001
        if self.params['eta0'] + eta0_shift < np.float32(0.0001):
            self.params['eta0'] = np.float32(0.0001)
        # check if the eta0 is going to be greater than 10
        elif self.params['eta0'] + eta0_shift > np.float32(10.0):
            self.params['eta0'] = np.float32(10.0)
        # if neither of the above, then we can just add the shift
        else:
            self.params['eta0'] = self.params['eta0'] + eta0_shift

        # randomly pick fit_intercept
        self.params['fit_intercept'] = rng.choice([True, False])
        # randomly pick loss
        self.params['loss'] = rng.choice(['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive'])
        # randomly pick penalty
        self.params['penalty'] = rng.choice(['l2','l1','elasticnet',None])
        # randomly pick learning_rate
        self.params['learning_rate'] = rng.choice(['constant','optimal','invscaling','adaptive'])

        # new regressor configuration
        self.regressor = SGDRegressor(**self.params)

# SVR regression
class SVRNode(ScikitNode, RegressorMixin):
    def __init__(self,
                 rng_: rng_t,
                 params: Dict = {},
                 name: name_t = name_t('SVR')):
        super().__init__(name)
        rng = np.random.default_rng(rng_)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'kernel': rng.choice(['linear', 'poly', 'rbf', 'sigmoid']),
                           'degree': np.int8(rng.choice([1,2,3])),
                           'gamma': rng.choice(['scale', 'auto']),
                           'C': np.float32(rng.uniform(low=0.0001, high=10.00)),
                           'epsilon': np.float32(rng.uniform(low=0.0001, high=10.00)),
                           'tol': np.float32(rng.uniform(low=1e-5, high=0.5))}
        else:
            assert len(params) == 6
            assert 'kernel' in params
            assert 'degree' in params
            assert 'gamma' in params
            assert 'C' in params
            assert 'epsilon' in params
            assert 'tol' in params
            self.params = params

        self.regressor = SVR(**self.params)

    def fit(self, X, y):
        return self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)

    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)

    def mutate(self, rng_):
        rng = np.random.default_rng(rng_)

        # get a random number from a normal distribution
        C_shift = np.float32(rng.normal(loc=0.0, scale=1.0))

        # check if the C is going to be less than 0.0001
        if self.params['C'] + C_shift < np.float32(0.0001):
            self.params['C'] = np.float32(0.0001)
        # check if the C is going to be greater than 10
        elif self.params['C'] + C_shift > np.float32(10.0):
            self.params['C'] = np.float32(10.0)
        # if neither of the above, then we can just add the shift
        else:
            self.params['C'] = self.params['C'] + C_shift

        # get a random number from a normal distribution
        epsilon_shift = np.float32(rng.normal(loc=0.0, scale=1.0))

        # check if the epsilon is going to be less than 0.0001
        if self.params['epsilon'] + epsilon_shift < np.float32(0.0001):
            self.params['epsilon'] = np.float32(0.0001)
        # check if the epsilon is going to be greater than 10
        elif self.params['epsilon'] + epsilon_shift > np.float32(10.0):
            self.params['epsilon'] = np.float32(10.0)
        # if neither of the above, then we can just add the shift
        else:
            self.params['epsilon'] = self.params['epsilon'] + epsilon_shift

        # get a random number from a normal distribution
        tol_shift = np.float32(rng.normal(loc=0.0, scale=0.05))

        # check if the tol is going to be less than 1e-5
        if self.params['tol'] + tol_shift < np.float32(1e-5):
            self.params['tol'] = np.float32(1e-5)
        # check if the tol is going to be greater than 0.5
        elif self.params['tol'] + tol_shift > np.float32(0.5):
            self.params['tol'] = np.float32(0.5)
        # if neither of the above, then we can just add the shift
        else:
            self.params['tol'] = self.params['tol'] + tol_shift

        # randomly pick kernel
        self.params['kernel'] = rng.choice(['linear', 'poly', 'rbf', 'sigmoid'])
        # randomly pick degree
        self.params['degree'] = np.int8(rng.choice([1,2,3]))
        # randomly pick gamma
        self.params['gamma'] = rng.choice(['scale', 'auto'])

        # new regressor configuration
        self.regressor = SVR(**self.params)

# Decision tree regression
class DecisionTreeRegressorNode(ScikitNode, RegressorMixin):
    def __init__(self,
                rng_: rng_t,
                seed: int = -1,
                params: Dict = {},
                name: name_t = name_t('DecisionTreeRegressor')):
        super().__init__(name)
        rng = np.random.default_rng(rng_)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'criterion': rng.choice(['squared_error', 'friedman_mse', 'absolute_error']),
                           'splitter': rng.choice(['best', 'random']),
                           'max_features': rng.choice([None, 'sqrt', 'log2']),
                           'max_depth': np.int8(rng.integers(1,10)),
                           'min_samples_split': np.int8(rng.integers(2,20)),
                           'min_samples_leaf': np.int8(rng.integers(1,20)),
                           'random_state': seed}
        else:
            assert len(params) == 7
            assert 'criterion' in params
            assert 'splitter' in params
            assert 'max_features' in params
            assert 'max_depth' in params
            assert 'min_samples_split' in params
            assert 'min_samples_leaf' in params
            assert 'random_state' in params
            self.params = params

        self.regressor = DecisionTreeRegressor(**self.params)

    def fit(self, X, y):
        return self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)

    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)

    def mutate(self, rng):
        # shift for max_depth up or down
        max_depth_shift = np.int8(rng.choice([-2,-1,1,2]))

        # check if the max_depth is going to be less than 1
        if self.params['max_depth'] + max_depth_shift < np.int8(1):
            self.params['max_depth'] = np.int8(1)
        # check if the max_depth is going to be greater than 10
        elif self.params['max_depth'] + max_depth_shift > np.int8(10):
            self.params['max_depth'] = np.int8(10)
        # if neither of the above, then we can just add the shift
        else:
            self.params['max_depth'] = self.params['max_depth'] + max_depth_shift

        # shift for min_samples_split up or down
        min_samples_split_shift = np.int8(rng.choice([-2,-1,1,2]))

        # check if the min_samples_split is going to be less than 1
        if self.params['min_samples_split'] + min_samples_split_shift < np.int8(2):
            self.params['min_samples_split'] = np.int8(2)
        # check if the min_samples_split is going to be greater than 20
        elif self.params['min_samples_split'] + min_samples_split_shift > np.int8(20):
            self.params['min_samples_split'] = np.int8(20)
        # if neither of the above, then we can just add the shift
        else:
            self.params['min_samples_split'] = self.params['min_samples_split'] + min_samples_split_shift

        # shift for min_samples_leaf up or down
        min_samples_leaf_shift = np.int8(rng.choice([-2,-1,1,2]))

        # check if the min_samples_leaf is going to be less than 1
        if self.params['min_samples_leaf'] + min_samples_leaf_shift < np.int8(1):
            self.params['min_samples_leaf'] = np.int8(1)
        # check if the min_samples_leaf is going to be greater than 20
        elif self.params['min_samples_leaf'] + min_samples_leaf_shift > np.int8(20):
            self.params['min_samples_leaf'] = np.int8(20)
        # if neither of the above, then we can just add the shift
        else:
            self.params['min_samples_leaf'] = self.params['min_samples_leaf'] + min_samples_leaf_shift

        # randomly pick criterion
        self.params['criterion'] = rng.choice(['squared_error', 'friedman_mse', 'absolute_error'])
        # randomly pick splitter
        self.params['splitter'] = rng.choice(['best', 'random'])
        # randomly pick max_features
        self.params['max_features'] = rng.choice([None, 'sqrt', 'log2'])

        # new regressor configuration
        self.regressor = DecisionTreeRegressor(**self.params)

# Random forest regression
class RandomForestRegressorNode(ScikitNode, RegressorMixin):
    def __init__(self,
                rng_: rng_t,
                seed: int = -1,
                params: Dict = {},
                name: name_t = name_t('RandomForestRegressor')):
        super().__init__(name)
        rng = np.random.default_rng(rng_)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'n_estimators': np.int8(rng.integers(10,100)),
                           'criterion': rng.choice(['squared_error', 'friedman_mse', 'absolute_error']),
                           'max_depth': np.int8(rng.integers(1,10)),
                           'max_features': rng.choice([None, 'sqrt', 'log2']),
                           'min_samples_split': np.int8(rng.integers(2,20)),
                           'min_samples_leaf': np.int8(rng.integers(1,20)),
                           'random_state': seed}
        else:
            assert len(params) == 8
            assert 'n_estimators' in params
            assert 'criterion' in params
            assert 'splitter' in params
            assert 'max_depth' in params
            assert 'max_features' in params
            assert 'min_samples_split' in params
            assert 'min_samples_leaf' in params
            assert 'random_state' in params
            self.params = params

        self.regressor = RandomForestRegressor(**self.params)

    def fit(self, X, y):
        return self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)
    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)

    def mutate(self, rng_):
        rng = np.random.default_rng(rng_)

        # get a random number from a uniform distribution
        n_estimators_shift = np.int8(rng.integers(-10, 10))

        # check if the n_estimators is going to be less than 10
        if self.params['n_estimators'] + n_estimators_shift < np.int8(10):
            self.params['n_estimators'] = np.int8(10)
        # check if the n_estimators is going to be greater than 100
        elif self.params['n_estimators'] + n_estimators_shift > np.int8(100):
            self.params['n_estimators'] = np.int8(100)
        # if neither of the above, then we can just add the shift
        else:
            self.params['n_estimators'] = self.params['n_estimators'] + n_estimators_shift

        # shift for max_depth up or down
        max_depth_shift = np.int8(rng.choice([-2,-1,1,2]))

        # check if the max_depth is going to be less than 1
        if self.params['max_depth'] + max_depth_shift < np.int8(1):
            self.params['max_depth'] = np.int8(1)
        # check if the max_depth is going to be greater than 10
        elif self.params['max_depth'] + max_depth_shift > np.int8(10):
            self.params['max_depth'] = np.int8(10)
        # if neither of the above, then we can just add the shift
        else:
            self.params['max_depth'] = self.params['max_depth'] + max_depth_shift

        # shift for min_samples_split up or down
        min_samples_split_shift = np.int8(rng.choice([-2,-1,1,2]))

        # check if the min_samples_split is going to be less than 1
        if self.params['min_samples_split'] + min_samples_split_shift < np.int8(2):
            self.params['min_samples_split'] = np.int8(2)
        # check if the min_samples_split is going to be greater than 20
        elif self.params['min_samples_split'] + min_samples_split_shift > np.int8(20):
            self.params['min_samples_split'] = np.int8(20)
        # if neither of the above, then we can just add the shift
        else:
            self.params['min_samples_split'] = self.params['min_samples_split'] + min_samples_split_shift

        # shift for min_samples_leaf up or down
        min_samples_leaf_shift = np.int8(rng.choice([-2,-1,1,2]))

        # check if the min_samples_leaf is going to be less than 1
        if self.params['min_samples_leaf'] + min_samples_leaf_shift < np.int8(1):
            self.params['min_samples_leaf'] = np.int8(1)
        # check if the min_samples_leaf is going to be greater than 20
        elif self.params['min_samples_leaf'] + min_samples_leaf_shift > np.int8(20):
            self.params['min_samples_leaf'] = np.int8(20)
        # if neither of the above, then we can just add the shift
        else:
            self.params['min_samples_leaf'] = self.params['min_samples_leaf'] + min_samples_leaf_shift

        # randomly pick criterion
        self.params['criterion'] = rng.choice(['squared_error', 'friedman_mse', 'absolute_error'])
        # randomly pick max_features
        self.params['max_features'] = rng.choice([None, 'sqrt', 'log2'])

        # new regressor configuration
        self.regressor = RandomForestRegressor(**self.params)

# Gradient boosting regression
class GradientBoostingRegressorNode(ScikitNode, RegressorMixin):
    def __init__(self,
                rng_: rng_t,
                seed: int = -1,
                params: Dict = {},
                name: name_t = name_t('RandomForestRegressor')):
        super().__init__(name)
        rng = np.random.default_rng(rng_)

        # if params is an empty dictionary, then we will initialize the params
        if params == {}:
            self.params = {'loss': rng.choice(['squared_error', 'absolute_error', 'huber', 'quantile']),
                           'learning_rate': rng.uniform(low=1e-3, high=1.0),
                           'n_estimators': np.int8(rng.integers(10,100)),
                           'criterion': rng.choice(['squared_error', 'friedman_mse']),
                           'max_depth': rng.integers(1,10),
                           'min_samples_split': rng.integers(2,20),
                           'min_samples_leaf': rng.integers(1,20),
                           'random_state': seed}
        else:
            assert len(params) == 8
            assert 'loss' in params
            assert 'learning_rate' in params
            assert 'n_estimators' in params
            assert 'criterion' in params
            assert 'max_depth' in params
            assert 'min_samples_split' in params
            assert 'min_samples_leaf' in params
            assert 'random_state' in params
            self.params = params

        # initialize the regressor
        self.regressor = GradientBoostingRegressor(**self.params)

    def fit(self, X, y):
        return self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)

    def transform(self, X):
        # For consistency with the abstract class, we use the regressor's prediction as the transform output
        return self.predict(X)

    def mutate(self, rng_):
        rng = np.random.default_rng(rng_)

        # get random number from a normal distribution
        learning_rate_shift = rng.normal(loc=0.0, scale=0.5)

        # check if the learning_rate is going to be less than 1e-3
        if self.params['learning_rate'] + learning_rate_shift < 1e-3:
            self.params['learning_rate'] = 1e-3
        # check if the learning_rate is going to be greater than 10.0
        elif self.params['learning_rate'] + learning_rate_shift > 10.0:
            self.params['learning_rate'] = 10.0
        # if neither of the above, then we can just add the shift
        else:
            self.params['learning_rate'] = self.params['learning_rate'] + learning_rate_shift

        # shift for n_estimators up or down
        n_estimators_shift = np.int8(rng.choice([-10, 10]))

        # check if the n_estimators is going to be less than 10
        if self.params['n_estimators'] + n_estimators_shift < np.int8(10):
            self.params['n_estimators'] = np.int8(10)
        # check if the n_estimators is going to be greater than 100
        elif self.params['n_estimators'] + n_estimators_shift > np.int8(100):
            self.params['n_estimators'] = np.int8(100)
        # if neither of the above, then we can just add the shift
        else:
            self.params['n_estimators'] = self.params['n_estimators'] + n_estimators_shift

        # shift for max_depth up or down
        max_depth_shift = np.int8(rng.choice([-2,-1,1,2]))

        # check if the max_depth is going to be less than 1
        if self.params['max_depth'] + max_depth_shift < np.int8(1):
            self.params['max_depth'] = np.int8(1)
        # check if the max_depth is going to be greater than 10
        elif self.params['max_depth'] + max_depth_shift > np.int8(10):
            self.params['max_depth'] = np.int8(10)
        # if neither of the above, then we can just add the shift
        else:
            self.params['max_depth'] = self.params['max_depth'] + max_depth_shift

        # shift for min_samples_split up or down
        min_samples_split_shift = np.int8(rng.choice([-2,-1,1,2]))

        # check if the min_samples_split is going to be less than 1
        if self.params['min_samples_split'] + min_samples_split_shift < np.int8(2):
            self.params['min_samples_split'] = np.int8(2)
        # check if the min_samples_split is going to be greater than 20
        elif self.params['min_samples_split'] + min_samples_split_shift > np.int8(20):
            self.params['min_samples_split'] = np.int8(20)
        # if neither of the above, then we can just add the shift
        else:
            self.params['min_samples_split'] = self.params['min_samples_split'] + min_samples_split_shift

        # shift for min_samples_leaf up or down
        min_samples_leaf_shift = np.int8(rng.choice([-2,-1,1,2]))

        # check if the min_samples_leaf is going to be less than 1
        if self.params['min_samples_leaf'] + min_samples_leaf_shift < np.int8(1):
            self.params['min_samples_leaf'] = np.int8(1)
        # check if the min_samples_leaf is going to be greater than 20
        elif self.params['min_samples_leaf'] + min_samples_leaf_shift > np.int8(20):
            self.params['min_samples_leaf'] = np.int8(20)
        # if neither of the above, then we can just add the shift
        else:
            self.params['min_samples_leaf'] = self.params['min_samples_leaf'] + min_samples_leaf_shift

        # randomly pick loss
        self.params['loss'] = rng.choice(['squared_error', 'absolute_error', 'huber', 'quantile'])
        # randomly pick criterion
        self.params['criterion'] = rng.choice(['squared_error', 'friedman_mse'])

        # new regressor configuration
        self.regressor = GradientBoostingRegressor(**self.params)

##########################################################################################
############################ the ld classes ##############################################
##########################################################################################
@typechecked
class LDSelector(ScikitNode, TransformerMixin):
    def __init__(self,
                    rng_: rng_t,
                    seed: int = -1,
                    params: Dict = {},
                    name: name_t = name_t('LDSelector')
                    ):
            super().__init__(name)

            rng = np.random.default_rng(rng_)

            # if params is an empty dictionary, then we will initialize the params
            if params == {}:
                self.params = {'threshold': np.float32(rng.uniform(low=0.6, high=0.95)), 'genomic_distance': int(1000000)}
            else:
                # make sure params is correct
                assert 'threshold' in params
                assert len(params) == 2
                assert isinstance(params['threshold'], np.float32)
                assert isinstance(params['genomic_distance'], int)
                self.params = params

            self.threshold = self.params['threshold']
            self.genomic_distance = self.params['genomic_distance']
            self.seed = seed
            self.params = params
            #self.selector = LDSelector(x_original=x_original, header_snps_dict=header_snps_dict, filtered_feature_names=filtered_feature_names, rng_=rng, **self.params)
            self.selector = 'LDSelector'
            self.selected_features_ = None
            self.bool_mask = None

            # newly added attributes for ld selector feature selection
            # if only one returned, get snp_name (np.str_). Or else will be None
            self.name_of_selected_features = None

            # dictionary to store true or false to see if it was ld pruned or not
            # {'snp_name': True/False}
            self.snp_details_after_ld = None


    def fit(self, X_original, X_encoded, y, snp_r2_dict):
        """
        Fit the feature selector to the data.

        Parameters:
        - X (array-like): The input features (2D array of shape [n_samples, n_features]).
        - y : The target variable.

        Returns:
        - self: The fitted selector.
        """
        selected_snps = [] # List to store the selected SNPs
        ld_removed_snps = set() # Set to store the SNPs removed due to LD pruning
        ld_removed_details = {} # Dictionary to store the details of the SNPs removed due to LD pruning
        snp_details_after_ld = {} # Dictionary to store the details of the SNPs after LD pruning
        ld_threshold = self.threshold # Threshold for LD pruning
        max_distance = self.genomic_distance # Maximum genomic distance in between SNPs to be considered for LD pruning
        final_selected_snps = [] # List to store the final selected SNPs after LD pruning and conditional analysis

        if X_original.empty: # If the selector before the LD operator didn't select any features
            # set self.selected_features_ to None and return self
            self.selected_features_ = None
            return self

        # function to remove same groups being checked for LD and also to remove subsets of groups
        def remove_subsets(groups):
            """
            Removes subsets from a list of SNP groups.
            Args:
                groups: List of groups, where each group is a list of SNP names (or tuples of chromosome and position).
            Returns:
                List of unique groups with subsets removed.
            """
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

        # function to calculate the correlation coefficient between two SNPs
        def calculate_ld(X, snp1: np.str_, snp2: np.str_): # LD coefficient shoud be calculated using original data not reencoded data
            # Calculate correlation coefficient (Pearson's r)
            correlation = np.corrcoef(X[snp1], X[snp2])[0, 1]

            # Compute R² value
            r_squared = correlation ** 2
            # # print the r_squared value
            # print(f"R² value between {snp1} and {snp2} is {r_squared}", flush=True)

            return r_squared

        # get the column names of the original data which are in numpy array format
        column_names = X_original.columns
        chr,pos = [],[]

        for snp in column_names:
            chr.append(int(snp.split('.')[0]))
            pos.append(int(snp.split('.')[1]))

        # Create a DataFrame with SNP names, chromosomes, and positions
        genotype_df_columns = pd.DataFrame({'chrom': chr, 'pos': pos, 'snp': column_names})
        # sort the genotype_df_columns based on chromosome and position
        genotype_df_columns = genotype_df_columns.sort_values(['chrom', 'pos'])
        # get a list of sorted snps
        sorted_snps = genotype_df_columns['snp'].tolist()
        # this dataframe will have the original unencoded data in a sorted order
        genotype_df_original = X_original[sorted_snps]
        # this dataframe will have the encoded data in a sorted order
        genotype_df_encoded = X_encoded[sorted_snps]
        # all the chromosomes in the data
        chromosomes = genotype_df_columns['chrom'].unique()

        # calculate marginal R² values for each SNP
        marginal_r2 = {}
        # use the snp_r2_dict to get the marginal r2 values
        for snp in column_names:
            marginal_r2[snp] = snp_r2_dict[snp]

        # set the snp_details_after_ld to True for all the snps - True indicates LD pruned snps
        for snp in column_names:
            snp_details_after_ld[snp] = True

        for chrom in chromosomes:
            # list of snps to check for conditional analysis in the current chromosome
            snps_to_check_for_ca_in_chr = []
            # Get SNPs and their positions for the current chromosome
            chr_snps_df = genotype_df_columns[genotype_df_columns['chrom'] == chrom]
            chr_snps = chr_snps_df['snp'].tolist()
            # ----------------------
            # Group SNPs by distance
            # ----------------------
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
            # Remove duplicate groups and subsets
            groups = remove_subsets(groups)
            # Convert groups back to original SNP names
            groups = [[snp[0] for snp in group] for group in groups]
            # --------------------
            # Within-group LD prune
            # --------------------
            for group in groups:
                if len(group) == 1:
                    final_selected_snps.append(group[0])
                else:
                    group_df = genotype_df_original[group]
                    snp_list = group_df.columns.tolist()
                    # Mark SNPs in high LD
                    for i, snp1 in enumerate(snp_list):
                        if snp1 in ld_removed_snps:
                            continue
                        for j in range(i + 1, len(snp_list)):
                            snp2 = snp_list[j]
                            if snp2 in ld_removed_snps:
                                continue
                            ld_value = calculate_ld(genotype_df_original, snp1, snp2)
                            if ld_value > ld_threshold:
                                # Remove the lower marginal R² SNP
                                if marginal_r2[snp1] > marginal_r2[snp2]:
                                    ld_removed_snps.add(snp2)
                                    ld_removed_details[snp2] = (
                                        f"Removed due to high LD (R²={ld_value:.3f}) with {snp1}"
                                    )
                                else:
                                    ld_removed_snps.add(snp1)
                                    ld_removed_details[snp1] = (
                                        f"Removed due to high LD (R²={ld_value:.3f}) with {snp2}"
                                    )
                    # Collect non-removed SNPs
                    non_removed = [s for s in snp_list if s not in ld_removed_snps]
                    snps_to_check_for_ca_in_chr.extend(non_removed)

            # If no SNPs remain in this chromosome, skip
            snps_to_check_for_ca_in_chr = list(set(snps_to_check_for_ca_in_chr))
            # print("Number of SNPs to check for conditional analysis in chromosome : ", f'{chrom}', " is ", len(snps_to_check_for_ca_in_chr), flush=True)
            if len(snps_to_check_for_ca_in_chr) == 0:
                # print("Entering the if condition for 0 snps in chromosome to check for CA", flush=True)
                continue

            if len(snps_to_check_for_ca_in_chr) == 1:
                # print("Entering the if condition for 1 snp in chromosome to check for CA", flush=True)
                #final_chr_snps.extend(snps_to_check_for_ca_in_chr)
                final_selected_snps.extend(snps_to_check_for_ca_in_chr)
                continue
            # --------------------------------------
            # Iterative (stepwise) conditional analysis
            # --------------------------------------
            snps_remaining = snps_to_check_for_ca_in_chr[:]
            # print("Number of SNPs to check for conditional analysis in chromosome : ", f'{chrom}', " is ", len(snps_remaining), flush=True)
            final_chr_snps = []
            # We loop until we can't prune any more SNPs
            while True:
                if len(snps_remaining) < 2:
                    # Either 0 or 1 SNP left, just add them all and break
                    final_chr_snps.extend(snps_remaining)
                    break
                # Find the next peak SNP (highest marginal R² among snps_remaining)
                candidate_r2 = [(snp, marginal_r2[snp]) for snp in snps_remaining]
                peak_snp = max(candidate_r2, key=lambda x: x[1])[0]
                # Perform conditional analysis using ONLY the peak SNP as covariate
                X_peak = genotype_df_encoded[[peak_snp]]
                y = y.reshape(-1, 1)
                p_values = []
                tested_snps = []
                for snp in snps_remaining:
                    if snp == peak_snp:
                        continue
                    X_full = pd.concat([X_peak, genotype_df_encoded[[snp]]], axis=1)
                    model_full = LinearRegression().fit(X_full, y)
                    beta_snp = model_full.coef_[-1]
                    residuals = y - model_full.predict(X_full)
                    sigma_sq = np.sum(residuals**2) / (len(y) - X_full.shape[1])
                    X_snp_values = genotype_df_encoded[[snp]].values
                    std_error = np.sqrt(sigma_sq / np.sum((X_snp_values - X_snp_values.mean())**2))
                    wald_stat = beta_snp / std_error
                    p_val = 2 * (1 - stats.norm.cdf(abs(wald_stat)))
                    p_values.append(p_val)
                    tested_snps.append(snp)
                # Correct for multiple testing
                # print("Length of p_values: ", len(p_values), flush=True)
                alpha = 0.05
                # Ensure p_values is a 1-dimensional array
                if len(p_values) == 0:
                    # print("No p-values to process. Skipping.")
                    continue  # or handle this case differently if needed

                # Convert to 1D array
                p_values = np.array(p_values).flatten()

                # Handle single p-value explicitly
                if len(p_values) == 1:
                    # print("Only one p-value provided. Skipping multiple testing correction.")
                    # Decide on how to handle this case
                    if p_values[0] < alpha:
                        rejected = [True]
                    else:
                        rejected = [False]
                    pvals_corr = p_values  # No correction needed
                else:
                    # Perform multiple testing correction
                    rejected, pvals_corr, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
                # Decide which SNPs to remove
                to_remove = {s for s, r in zip(tested_snps, rejected) if not r}
                # If we didn't prune anything this round, we're done
                if not to_remove:
                    # Add the peak SNP to final list if not already present
                    if peak_snp not in final_chr_snps:
                        final_chr_snps.append(peak_snp)
                    break
                # Otherwise, remove the pruned SNPs
                snps_remaining = [s for s in snps_remaining if s not in to_remove]
                # Add the peak SNP to final list (it is an independent peak)
                if peak_snp not in final_chr_snps:
                    final_chr_snps.append(peak_snp)
                # Remove the peak SNP from further consideration so the next iteration
                # can find the next peak ignoring this one
                snps_remaining.remove(peak_snp)
                # Loop continues with the updated snps_remaining
            # Add the final chromosome SNPs to the overall final selection
            final_selected_snps.extend(list(set(final_chr_snps)))
        # Print the final list of selected SNPs
        # print("Final list of independent SNPs selected: ", list(set(final_selected_snps)) )

        assert len(final_selected_snps) > 0, "No SNPs were selected by the LDSelector"

        # setting the name of the selected feature to be used during evolution
        if len(final_selected_snps) == 1:
            self.name_of_selected_features = final_selected_snps[0]
            # print(f"Selected SNP name: {self.name_of_selected_features}", flush=True)

        # change the details in snp_details_after_ld to False for the selected snps
        for snp in final_selected_snps:
            snp_details_after_ld[snp] = False

        # print("Snp details after LD: ", snp_details_after_ld, flush=True)
        self.snp_details_after_ld = snp_details_after_ld

        # Create a boolean mask for the selected SNPs
        boolean_mask = np.isin(column_names, final_selected_snps)
        self.bool_mask = boolean_mask

        self.selected_features_ = np.array(final_selected_snps)
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
        if self.threshold + shift < np.float32(0.6):
            self.threshold = np.float32(0.6)
        # check if the threshold is going to be greater than 1
        elif self.threshold + shift > np.float32(0.95):
            self.threshold = np.float32(0.95)
        # if neither of the above, then we can just add the shift
        else:
            self.threshold = self.threshold + shift

        # increment genomin distance by 100000 with a minimum of 500000 and maximum of 1000000, in increments of 100000
        genomic_distance_shift = np.int32(rng.choice([-500000, 500000]))
        # check if the genomic_distance is going to be less than 500000
        if self.genomic_distance + genomic_distance_shift < 500000:
            self.genomic_distance = 500000
        # check if the genomic_distance is going to be greater than 1000000
        elif self.genomic_distance + genomic_distance_shift > 10000000:
            self.genomic_distance = 10000000
        # if neither of the above, then we can just add the shift
        else:
            self.genomic_distance = self.genomic_distance + genomic_distance_shift

        # initialize the selector with the new threshold
        self.params['threshold'] = np.float32(self.threshold)
        self.params['genomic_distance'] = int(self.genomic_distance)
        LDSelector(rng_= rng, params= self.params)

    def get_feature_count(self):
        """
            Get the number of features selected by the selector.

            Returns:
            - int: The number of features selected. If the selector has not been fitted yet,
           raises a RuntimeError.
    """
        if self.selected_features_ is None:
            raise RuntimeError("LDSelector has not been fitted yet.")
        # print"No of selected features: ", len(self.selected_features_), flush=True)
        # return length of selected features as no of true values in boolean mask
        return len(self.selected_features_)
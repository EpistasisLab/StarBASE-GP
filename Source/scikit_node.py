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
        self.regressor.fit(X, y)

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
        self.regressor.fit(X, y)

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
        self.regressor.fit(X, y)

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
        self.regressor.fit(X, y)

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
        self.regressor.fit(X, y)

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
        self.regressor.fit(X, y)

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
        self.params['criterion'] = rng.choice(['squared_error', 'friedman_mse', 'absolute_error'])

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
                self.params = {'threshold': np.float32(rng.uniform(low=0.1, high=1)), 'genomic_distance': 500000}
            else:
                # make sure params is correct
                assert 'threshold' in params
                assert len(params) == 1
                assert isinstance(params['threshold'], np.float32)
                self.params = params

            self.threshold = self.params['threshold']
            self.genomic_distance = self.params['genomic_distance']
            self.seed = seed
            self.params = params
            #self.selector = LDSelector(x_original=x_original, header_snps_dict=header_snps_dict, filtered_feature_names=filtered_feature_names, rng_=rng, **self.params)
            self.selector = 'LDSelector'
            self.selected_features_ = None
            self.bool_mask = None
    

    def fit(self, X_original, X_encoded, y, snp_r2_dict):
        """
        Fit the feature selector to the data.

        Parameters:
        - X (array-like): The input features (2D array of shape [n_samples, n_features]).
        - y : The target variable.

        Returns:
        - self: The fitted selector.
        """
        # print("LDSelector fit method called", flush=True)
        # print("Printing X original: ", X_original, flush=True)
        # print("Printing X encoded: ", X_encoded, flush=True)
        #X = np.asarray(self.x_original)  # Ensure input is a numpy array
        selected_snps = [] # List to store the selected SNPs
        ld_removed_snps = set() # Set to store the SNPs removed due to LD pruning
        ld_removed_details = {} # Dictionary to store the details of the SNPs removed due to LD pruning
        ld_threshold = self.threshold # Threshold for LD pruning
        max_distance = self.genomic_distance # Maximum genomic distance in between SNPs to be considered for LD pruning
        final_selected_snps = [] # List to store the final selected SNPs after LD pruning and conditional analysis

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
        def calculate_ld(X, snp1: np.str_, snp2: np.str_): # LD coefficient shoud be calculated using original data not reencoded
            # Extract genotype vectors for the two SNPs
            # snp1 = X[snp1]
            # snp2 = X[snp2]

            # Calculate correlation coefficient (Pearson's r)
            # correlation = np.corrcoef(snp1, snp2)[0, 1]
            correlation = np.corrcoef(X[snp1], X[snp2])[0, 1]

            # Compute R² value
            r_squared = correlation ** 2
            # # print the r_squared value
            # print(f"R² value between {snp1} and {snp2} is {r_squared}", flush=True)

            return r_squared
        
        # extract chromosome number and position from the column names
        # Assume SNP names are in the format 'X.yyyyy' where X is chromosome and yyyyy is position
        def extract_chr_pos(snp_name):
            chrom, pos = snp_name.split('.')
            return int(chrom), int(pos)  # Use float for positions to preserve precision

        # get the column names of the original data which are in numpy array format
        column_names = X_original.columns
   
        # Create a DataFrame with SNP names, chromosomes, and positions
        genotype_df_columns = pd.DataFrame({'snp': column_names})
        #print the genotype_df_columns
        # print("Printing the genotype_df_columns: ", genotype_df_columns, flush=True)
        #genotype_df_columns = pd.DataFrame({'snp': column_names}, index=range(len(column_names)))

        # Apply the function to extract chromosome and position
        genotype_df_columns[['chrom', 'pos']] = genotype_df_columns['snp'].apply(
            lambda x: pd.Series(extract_chr_pos(x))
        )
        #print("Printing the genotype_df_columns after extracting chromosome and position: ", genotype_df_columns, flush=True)

        # sort the SNPs by chromosome and position
        genotype_df_columns['chrom'] = pd.to_numeric(genotype_df_columns['chrom'], errors='coerce')
        genotype_df_columns['pos'] = pd.to_numeric(genotype_df_columns['pos'], errors='coerce')
        genotype_df_columns = genotype_df_columns.sort_values(['chrom', 'pos'])
        #print the genotype_df_columns after sorting
        #print("Printing the genotype_df_columns after sorting: ", genotype_df_columns, flush=True)

        sorted_snps = genotype_df_columns['snp'].tolist()
        #print("Printing the sorted snps: ", sorted_snps, flush=True)
        ################################################################################
        # extract the sorted_snps from the self.x_original
        # filter x_original to have the sorted_snps columns only
        genotype_df_original = X_original[sorted_snps]
        #print("Type of genotype_df_original: ", type(genotype_df_original), flush=True)
        # print the genotype_df_original
        #print("Printing the genotype unencoded data: ", genotype_df_original, flush=True)
        ################################################################################
        # encoded data
        genotype_df_encoded = X_encoded[sorted_snps]
        # print the genotype_df_encoded
        #print("Printing the genotype encoded data: ", genotype_df_encoded, flush=True)

        # all the chromosomes in the data
        chromosomes = genotype_df_columns['chrom'].unique()

        # calculate marginal R² values for each SNP
        marginal_r2 = {}
        # Fit a univariate linear regression model for each SNP
        for snp in column_names:
            # Extract the genotype vector for the SNP
            x = X_encoded[[snp]]
            # Fit a linear regression model
            model = LinearRegression()
            model.fit(x, y)

            # Calculate the R² value
            marginal_r2[snp] = model.score(x, y)

        for chrom in chromosomes:
            # Get SNPs and their positions for the current chromosome
            chr_snps_df = genotype_df_columns[genotype_df_columns['chrom'] == chrom]
            chr_snps = chr_snps_df['snp'].tolist()
            num_snps = len(chr_snps)

            # Group SNPs by maximum genomic distance
            groups = []
            current_group = [(chr_snps_df.iloc[0]['snp'], chr_snps_df.iloc[0]['pos'])]  # Store SNP and position as tuples

            for i in range(1, len(chr_snps)):
                current_snp = (chr_snps_df.iloc[i]['snp'], chr_snps_df.iloc[i]['pos'])
                previous_snp = (chr_snps_df.iloc[i - 1]['snp'], chr_snps_df.iloc[i - 1]['pos'])

                if int(abs(current_snp[1] - previous_snp[1])) <= max_distance:
                    current_group.append(current_snp)
                else:
                    groups.append(current_group)
                    current_group = [current_snp]

            groups.append(current_group)

            # Remove duplicate groups and subsets
            groups = remove_subsets(groups)

            # Convert groups back to original SNP names for further processing
            groups = [[snp[0] for snp in group] for group in groups]

            # # Print group statistics
            # print("A total of", len(groups), "unique groups were created for chromosome", chrom)
            # print("Group sizes:", [len(group) for group in groups])

            # Apply LD pruning within each group (only if group size > 1)
            for group in groups:
                #print("Processing group:", group)
                group_selected_snps = []  # List to store selected SNPs within the group, which will be later checked for conditional analysis
                if len(group) > 1:
                    # Convert group into a DataFrame
                    group_df = genotype_df_original[group]
                    snp_list = group_df.columns.tolist()

                    # LD pruning logic: compare every pair of SNPs
                    for i, snp1 in enumerate(snp_list):
                        if snp1 in ld_removed_snps:
                            continue  # Skip SNPs already removed

                        for j in range(i + 1, len(snp_list)):
                            snp2 = snp_list[j]

                            # Calculate LD between snp1 and snp2
                            ld_value = calculate_ld(genotype_df_original, snp1, snp2)
                            if ld_value > ld_threshold:
                                # Mark snp2 as removed due to high LD with snp1
                                # Compare marginal R² values of the two SNPs

                                if marginal_r2[snp1] > marginal_r2[snp2]:
                                    ld_removed_snps.add(snp2)
                                    ld_removed_details[snp2] = f"Removed due to high LD (R²={ld_value:.3f}) with {snp1}"
                                else:
                                    ld_removed_snps.add(snp1)
                                    ld_removed_details[snp1] = f"Removed due to high LD (R²={ld_value:.3f}) with {snp2}"

                    # Add non-removed SNPs from this group to group_selected_snps
                    group_selected_snps.extend([snp for snp in snp_list if snp not in ld_removed_snps])

                    # perform conditional anlysis if group_selected_snps is greater than 1
                    if len(group_selected_snps) > 1:
                        # Identify peak SNP with the highest marginal R² within this group
                        group_selected_df = pd.DataFrame({
                            'snp': group_df.columns,
                            'marginal_r2': [marginal_r2[snp] for snp in group_df.columns]
                        })
                        peak_snp = group_selected_df.loc[group_selected_df['marginal_r2'].idxmax(), 'snp']
                        #print(f"Peak SNP for the group: {peak_snp}, Marginal R²: {marginal_r2[peak_snp]:.4f}", flush=True)

                        # Perform conditional analysis using the peak SNP as covariate
                        X_peak = genotype_df_encoded[[peak_snp]]
                        y = y.reshape(-1, 1)
                        p_values = []
                        snp_list = []

                        # For each SNP, perform conditional analysis
                        for snp in group_selected_snps:
                            # print the snp
                            #print(f"Performing conditional analysis for SNP {snp}", flush=True)
                            if snp == peak_snp:
                                # print('Peak SNP should not be in the list of removed SNPs')
                                # print(snp, '---', peak_snp, flush=True)
                                # exit(0)
                                continue
                            # Full model with peak SNP and the current SNP
                            X_full = pd.concat([X_peak, genotype_df_encoded[[snp]]], axis=1)
                            model_full = LinearRegression().fit(X_full, y)
                            ssr_full = np.sum((y - model_full.predict(X_full)) ** 2)
                            df_full = len(y) - X_full.shape[1]
                            # Check for valid SSR and DF
                            # if np.isclose(ssr_full, 0) or df_full <= 0:
                            #     print(f"Skipping SNP {snp} due to invalid SSR or degrees of freedom.")
                            #     continue
                            # Reduced model with peak SNP only
                            model_reduced = LinearRegression().fit(X_peak, y)
                            ssr_reduced = np.sum((y - model_reduced.predict(X_peak)) ** 2)
                            # Check SSR values
                            # if ssr_reduced < ssr_full:
                            #     print(f"Skipping SNP {snp} due to invalid SSR values (reduced < full).")
                            #     continue
                            #df_reduced = len(y) - X_peak.shape[1]
                            # F-test to see if the current SNP adds significant information
                            num = ssr_reduced - ssr_full
                            denom = ssr_full / df_full
                            F_stat = num / denom
                            # if np.isnan(F_stat) or np.isinf(F_stat):
                            #     print(f"Skipping SNP {snp} due to invalid F-statistic.")
                            #     continue
                            # print("F statistics: ", F_stat)
                            # print("DF full: ", df_full)
                            # print('stats.f.cdf(F_stat, 1, df_full):',stats.f.cdf(F_stat, 1, df_full))
                            p_value = 1 - stats.f.cdf(F_stat, 1, df_full)
                            #print(f"Conditional analysis for SNP {snp}: F-statistic = {F_stat:.4f}, p-value = {p_value:.4f}", flush=True)
                            # Store the p-value and SNP for FDR correction
                            p_values.append(p_value)
                            snp_list.append(snp)

                        # Apply FDR correction to the p-values
                        alpha = 0.05  # Desired overall significance level
                        # print"Length of p_values to be FDR corrected: ", len(p_values), flush=True)
                        if not p_values:
                            # print"No p-values to correct. Skipping FDR correction.", flush=True)
                            continue
                        rejected, p_values_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

                        # Remove SNPs that did not pass the conditional analysis
                        conditional_removed_snps = set()
                        for snp, reject in zip(snp_list, rejected):
                            if not reject:
                                # SNP does not provide significant additional information
                                conditional_removed_snps.add(snp)

                        # Add SNPs that passed the conditional analysis
                        final_group_selected_snps = [snp for snp in group_selected_snps if snp not in conditional_removed_snps]
                        final_selected_snps.extend(final_group_selected_snps)

                        # # Print selected SNPs for this group after conditional analysis
                        # print(f"Selected SNPs for this group after conditional analysis: {final_group_selected_snps}")
                        # print(f"Total SNPs selected for this group: {len(final_group_selected_snps)}")

                    else:
                        # If group size is 1, no conditional analysis is needed; directly add the SNP
                        final_selected_snps.append(group_selected_snps[0])
                else:
                    # If group size is 1, no conditional analysis is needed; directly add the SNP
                    single_snp = group[0]
                    #print(f"Group contains only one SNP: {single_snp}. Skipping conditional analysis.")
                    final_selected_snps.append(single_snp)

        # Print the final list of selected SNPs
        # print"Final list of selected SNPs:")
        # printfinal_selected_snps, flush=True)
        # print"Length of final selected SNPs: ", len(final_selected_snps), flush=True)
        assert len(final_selected_snps) > 0, "No SNPs were selected by the LDSelector"

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
        shift = np.float32(rng.normal(loc=0.1, scale=1))

        # check if the threshold is going to be less than 0.1
        if self.threshold + shift < np.float32(0.1):
            self.threshold = np.float32(0.1)
        # check if the threshold is going to be greater than 1
        elif self.threshold + shift > np.float32(1):
            self.threshold = np.float32(1)
        # if neither of the above, then we can just add the shift
        else:
            self.threshold = self.threshold + shift

        self.params['threshold'] = self.threshold
        # initialize the selector with the new threshold
        LDSelector(x_original=self.x_original, header_snps_dict=self.header_snps_dict, filtered_feature_names=self.filtered_feature_names, rng_= rng)

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
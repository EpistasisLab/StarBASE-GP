#####################################################################################################
#
# This class provides a set of feature selectors that are used in the pipeline workflow.
# Derived class for feature selectors.
#
#####################################################################################################

from abc import ABC, abstractmethod
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, SelectFwe, SelectFromModel, f_regression
from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesRegressor
from typeguard import typechecked
from .types import (rng_t, snp_t, float32_t, int8_t)

# base class for all scikit-learn nodes
class SelectorNode(ABC):
    def __init__(self, selector):
        '''Initialize the selector node.
        Parameters:
        - selector: A scikit-learn selector instance.
        '''
        self.selector = selector
        # Note: self.name and self.params should be set by subclasses BEFORE calling super().__init__()

    def fit(self, X, y=None):
        self.selector.fit(X, y)

    def transform(self, X):
        return self.selector.transform(X)

    def get_params(self):
        """Return the parameters of the selector."""
        return getattr(self, 'params', {})  # Return params if it exists, otherwise empty dict

    @abstractmethod
    def mutate(self, rng: rng_t):
        pass

    def get_feature_names(self, feature_names):
        # ensure feature_names is a NumPy array
        feature_names = np.array(feature_names)

        # ensure the length of feature_names matches the number of features in the data
        support_mask = self.selector.get_support()
        if len(feature_names) != len(support_mask):
            raise ValueError("Length of feature_names does not match the number of features in the data.")

        # use the Boolean mask to filter feature names
        return feature_names[support_mask]

    def get_feature_count(self):
        return self.selector.get_support().sum()

##########################################################################################
########################## the feature selector classes ##################################
##########################################################################################

@typechecked
class VarianceThresholdNode(SelectorNode):
    def __init__(self, rng: rng_t):
        self.name = 'VarianceThreshold'
        # threshold is a float between 0.0001 and 0.05
        self.params = {'threshold': float32_t(rng.uniform(low=0.0001, high=0.05))}
        # pass super class the initialized selector
        super().__init__(VarianceThreshold(threshold=self.params['threshold']))

    def fit(self, X, y=None):
        self.selector.fit(X)

    def mutate(self, rng: rng_t):
        # get a random number from a normal distribution
        shift = float32_t(rng.normal(loc=0.0, scale=0.025))

        # check if the threshold is going to be negative
        if self.params['threshold'] + shift < float32_t(0.0):
            self.params['threshold'] = float32_t(0.0)
        # check if the threshold is going to be greater than 1
        elif self.params['threshold'] + shift > float32_t(1.0):
            self.params['threshold'] = float32_t(1.0)
        # if neither of the above, then we can just add the shift
        else:
            self.params['threshold'] = self.params['threshold'] + shift

        # new selector configuration
        self.selector = VarianceThreshold(threshold=self.params['threshold'])

@typechecked
class SelectPercentileNode(SelectorNode):
    def __init__(self, rng: rng_t):
        self.name = 'SelectPercentile'
        # percentile is an int between 50 and 100
        self.params = {'percentile': int8_t(rng.integers(low=50, high=100)), 'score_func': f_regression}
        # pass super class the initialized selector
        super().__init__(SelectPercentile(score_func=self.params['score_func'], percentile=self.params['percentile']))

    def mutate(self, rng: rng_t):
        # maginitude by which we are shfiting the percentile
        shift = int8_t(rng.integers(low=-5, high=5, endpoint=True))

        # check if the percentile is going to be less than 1
        if self.params['percentile'] + shift < int8_t(1):
            self.params['percentile'] = int8_t(1)
        # check if the percentile is going to be greater than 100
        elif self.params['percentile'] + shift > int8_t(100):
            self.params['percentile'] = int8_t(100)
        # if neither of the above, then we can just add the shift
        else:
            self.params['percentile'] = self.params['percentile'] + shift

        # new selector configuration
        self.selector = SelectPercentile(score_func=self.params['score_func'], percentile=self.params['percentile'])

@typechecked
class SelectFweNode(SelectorNode):
    def __init__(self, rng: rng_t):
        self.name = 'SelectFwe'
        # alpha is a float between 0.0001 and 0.05
        self.params = {'alpha': float32_t(rng.uniform(low=1e-4, high=0.05)), 'score_func': f_regression}
        # pass super class the initialized selector
        super().__init__(SelectFwe(score_func=self.params['score_func'], alpha=self.params['alpha']))

    def mutate(self, rng: rng_t):
        # get a random number from a normal distribution
        shift = float32_t(rng.normal(loc=0.0, scale=0.005))

        # check if the alpha is going to be negative
        if self.params['alpha'] + shift < float32_t(0.0001):
            self.params['alpha'] = float32_t(0.0001)
        # check if the alpha is going to be greater than .99
        elif self.params['alpha'] + shift > float32_t(0.99):
            self.params['alpha'] = float32_t(0.99)
        # if neither of the above, then we can just add the shift
        else:
            self.params['alpha'] = self.params['alpha'] + shift

        # new selector configuration
        self.selector = SelectFwe(score_func=self.params['score_func'], alpha=self.params['alpha'])

@typechecked
class SelectFromModelLasso(SelectorNode):
    def __init__(self, rng: rng_t, seed: int):
        self.name = 'SelectFromModelLasso'
        # threshold is either 'mean' or 'median', Lasso with random_state = seed
        self.params = {'estimator': Lasso(random_state=seed), 'threshold': rng.choice([snp_t('mean'), snp_t('median')])}
        # pass super class the initialized selector
        super().__init__(SelectFromModel(estimator = self.params['estimator'], threshold=self.params['threshold']))

    def mutate(self, rng: rng_t):
        # randomly select threshold
        self.params['threshold'] = rng.choice([snp_t('mean'), snp_t('median')])
        # new selector configuration
        self.selector = SelectFromModel(estimator = self.params['estimator'], threshold=self.params['threshold'])

@typechecked
class SelectFromModelTree(SelectorNode):
    # select from model using tree-based feature selection (model is ExtraTreesRegressor)
    def __init__(self, rng: rng_t, seed: int):
        self.name = 'SelectFromModelTree'
        # threshold is either 'mean' or 'median', ExtraTreesRegressor with random_state = seed
        self.params = {'estimator': ExtraTreesRegressor(random_state=seed), 'threshold': rng.choice([snp_t('mean'), snp_t('median')])}
        # pass super class the initialized selector
        super().__init__(SelectFromModel(estimator = self.params['estimator'], threshold=self.params['threshold']))

    def mutate(self, rng: rng_t):
        # randomly select threshold
        self.params['threshold'] = rng.choice([snp_t('mean'), snp_t('median')])
        # new selector configuration
        self.selector = SelectFromModel(estimator = self.params['estimator'], threshold=self.params['threshold'])

@typechecked
class FeatureEncodingFrequencySelector(SelectorNode):
    """
    Feature selector based on Encoding Frequency.
    Encoding frequency is the frequency of each unique element(0/1/2/3) present in a feature set.
    Features are selected on the basis of a threshold assigned for encoding frequency.
    If frequency of any unique element is less than or equal to threshold, the feature is removed.
    """

    def __init__(self, rng: rng_t):
        self.name = 'FeatureEncodingFrequencySelector'
        # threshold is a float between 0.01 and 0.3
        self.threshold = float32_t(rng.uniform(low=0.01, high=0.3)) # increments of 0.05
        self.params = {'threshold': self.threshold}  # Add params dict for consistency
        self.boolean_mask = None
        return

    def fit(self, X, y=None):
        """
        Fit the feature selector to the data.

        Parameters:
            X (array-like): The input features (2D array of shape [n_samples, n_features]).
            y (ignored): The target variable (not used in this selector).

        Returns:
            self: The fitted selector.
        """

        X = np.asarray(X)  # Ensure input is a numpy array
        n_samples, n_features = X.shape
        selected_features = []
        for i in range(n_features):
            _, counts = np.unique(X[:, i], return_counts=True)
            frequencies = counts / n_samples
            if np.all(frequencies >= self.threshold):
                selected_features.append(i)
        self.selected_features_ = np.array(selected_features)
        self.selected_features_ = np.array(self.selected_features_, dtype=int)
        # make a boolean mask of the selected features
        self.boolean_mask = np.zeros(X.shape[1], dtype=bool)
        self.boolean_mask[self.selected_features_] = True
        return

    def transform(self, X):
        """
        Transform the data to include only the selected features.

        Parameters:
            X (array-like): The input features (2D array of shape [n_samples, n_features]).
        Returns:
            X_transformed (array-like): The transformed array with only selected features.
        """

        if self.selected_features_ is None:
            raise RuntimeError("FeatureEncodingFrequencySelector has not been fitted yet.")
        X = np.asarray(X)  # Ensure input is a numpy array

        if X.shape[1] != len(self.boolean_mask):
            raise ValueError("Number of features in X does not match the number of features in the selector.")

        return X[:, self.boolean_mask]

    def mutate(self, rng: rng_t):
        # shift is a rng from normal distribution with a change in 2nd decimal place
        shift = float32_t(rng.normal(loc=0.01, scale=0.01))
        # check if the threshold is going to be less than 0.0
        if self.threshold + shift < float32_t(0.01):
            self.threshold = float32_t(0.01)
        # check if the threshold is going to be greater than 0.3
        elif self.threshold + shift > float32_t(0.3):
            self.threshold = float32_t(0.3)
        # if neither of the above, then we can just add the shift
        else:
            self.threshold = self.threshold + shift

        # Update params dict to stay in sync
        self.params['threshold'] = self.threshold

    def get_feature_count(self):
        """
        Get the number of features selected by the selector.

        Returns:
            int: The number of features selected. If the selector has not been fitted yet, raises a RuntimeError.
        """

        if self.selected_features_ is None:
            raise RuntimeError("FeatureEncodingFrequencySelector has not been fitted yet.")
        return len(self.selected_features_)

    def get_feature_names(self, feature_names):
        """
        Get the names of the features selected by the selector.

        Parameters:
            feature_names (array-like): The names of the features.

        Returns:
            array-like: The names of the selected features.
        """

        if self.selected_features_ is None:
            raise RuntimeError("FeatureEncodingFrequencySelector has not been fitted yet.")

        # Use the Boolean mask to filter feature names
        final_features = []
        for i in range(len(self.boolean_mask)):
            if self.boolean_mask[i]:
                final_features.append(feature_names[i])

        return final_features


##########################################################################################
########################## OLS Regressor for PFI #########################################
##########################################################################################

class OLSRegressor:
    """
    Simple wrapper for statsmodels OLS to work with sklearn's permutation_importance.
    This class provides a scikit-learn compatible interface for OLS regression.
    """
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        """Fit OLS model to training data."""
        import statsmodels.api as sm
        self.model = sm.OLS(y, X).fit()
        return self

    def predict(self, X):
        """Predict using the fitted OLS model."""
        return self.model.predict(X)
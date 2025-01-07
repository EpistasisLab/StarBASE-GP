import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Define constants
ENCODINGS = ['dominant', 'recessive', 'heterosis', 'underdominant', 
             'subadditive', 'superadditive', 'pager']

def _apply_basic_mapping(snp_values, mapping):
    """Vectorized mapping application"""
    result = np.zeros_like(snp_values, dtype=np.float32)
    for k, v in mapping.items():
        result[snp_values == k] = v
    return result

def _compute_pager_encoding(snp_values, y):
    """Compute PAGER encoding for a single column"""
    df = pd.DataFrame({'snp': snp_values, 'Phenotype': y})
    geno_agg = df.groupby('snp').agg(mean_phenotype=('Phenotype', 'mean')).reset_index()
    anchor_mean = geno_agg.loc[geno_agg['snp'].idxmin(), 'mean_phenotype']
    geno_agg['rel_dist'] = geno_agg['mean_phenotype'] - anchor_mean
    
    scaler = MinMaxScaler()
    geno_agg['normalized_rel_dist'] = scaler.fit_transform(
        geno_agg['rel_dist'].values.reshape(-1, 1)
    ).ravel()
    
    mapping = dict(zip(geno_agg['snp'], geno_agg['normalized_rel_dist']))
    result = np.array([mapping.get(val, 0.5) for val in snp_values], dtype=np.float32)
    return result

def map_snp_values(X, snp_pos, encoding_type='all', y=None):
    """
    Map SNP values based on the encoding type, with optimized 'all' option.
    
    Args:
        X (pd.DataFrame or np.ndarray): Input data
        snp_pos (int): Position of the SNP column
        encoding_type (str): Encoding type ('all' or specific encoding)
        y (np.ndarray, optional): Phenotype values for PAGER encoding
    
    Returns:
        np.ndarray: Mapped values, shape (n_samples, 1) or (n_samples, n_encodings) if encoding_type='all'
    """
    # Extract SNP values
    snp_values = X.iloc[:, snp_pos].values if isinstance(X, pd.DataFrame) else X[:, snp_pos]
    
    # Define basic mappings
    mappings = {
        'dominant': {0: 0, 0.5: 1, 1: 1},
        'recessive': {0: 0, 0.5: 0, 1: 1},
        'heterosis': {0: 0, 0.5: 1, 1: 0},
        'underdominant': {0: 0.5, 0.5: 0, 1: 1},
        'subadditive': {0: 0, 0.5: 0.25, 1: 1},
        'superadditive': {0: 0, 0.5: 0.75, 1: 1}
    }
    
    if encoding_type == 'all':
        # Initialize result matrix
        n_samples = len(snp_values)
        result = np.zeros((n_samples, len(ENCODINGS)), dtype=np.float32)
        
        # Apply basic mappings in parallel
        for i, enc in enumerate(ENCODINGS[:-1]):  # Exclude PAGER
            result[:, i] = _apply_basic_mapping(snp_values, mappings[enc])
        
        # Compute PAGER encoding if y is provided
        if y is not None:
            result[:, -1] = _compute_pager_encoding(snp_values, y)
        else:
            result[:, -1] = 0.5  # Default value if no phenotype data
            
        return result
    
    elif encoding_type == 'pager':
        if y is None:
            raise ValueError("Phenotype values (y) must be provided for PAGER encoding.")
        return _compute_pager_encoding(snp_values, y).reshape(-1, 1)
    
    elif encoding_type in mappings:
        return _apply_basic_mapping(snp_values, mappings[encoding_type]).reshape(-1, 1)
    
    else:
        raise ValueError(f"Unsupported encoding type. Must be one of {['all'] + ENCODINGS}")


# def compute_r2_optimized(X_train, X_test, y_train, y_test):
#     # Ensure inputs are numpy arrays
#     X_train = np.asarray(X_train)
#     X_test = np.asarray(X_test)
#     y_train = np.asarray(y_train)
#     y_test = np.asarray(y_test)

#     # Preallocate array for R² scores
#     r2_scores = np.zeros(X_train.shape[1])

#     # Loop over each column (feature)
#     for i in range(X_train.shape[1]):
#         # Extract individual feature column
#         X_train_col = X_train[:, i]
#         X_test_col = X_test[:, i]

#         # Add bias term (intercept)
#         X_train_bias = np.column_stack([X_train_col, np.ones(X_train_col.shape)])
#         X_test_bias = np.column_stack([X_test_col, np.ones(X_test_col.shape)])

#         # Compute weights using least squares
#         weights = np.linalg.lstsq(X_train_bias, y_train, rcond=None)[0]

#         # Predict test targets
#         y_pred = X_test_bias @ weights

#         # Compute R² for this feature
#         ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
#         ss_residual = np.sum((y_test - y_pred) ** 2)
#         r2_scores[i] = 1 - (ss_residual / ss_total)

#     return r2_scores


def compute_r2_optimized(X_train, X_test, y_train, y_test):
    """
    Vectorized computation of R² values for each column using train/test split.
    
    Args:
        X_train (np.ndarray): Training data, shape (n_train_samples, n_features)
        X_test (np.ndarray): Test data, shape (n_test_samples, n_features) 
        y_train (np.ndarray): Training target values, shape (n_train_samples,)
        y_test (np.ndarray): Test target values, shape (n_test_samples,)
    
    Returns:
        np.ndarray: R² values for each column on test data, shape (n_features,)
    """
    # Ensure inputs are numpy arrays
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Preallocate array for R² scores
    r2_scores = np.zeros(X_train.shape[1])

    # Compute R² for each feature column
    for i in range(X_train.shape[1]):
        # Reshape for single feature regression
        X_train_col = X_train[:, i].reshape(-1, 1)
        X_test_col = X_test[:, i].reshape(-1, 1)

        # Fit linear regression model
        model = LinearRegression()
        model.fit(X_train_col, y_train)

        # Predict on test data
        y_pred = model.predict(X_test_col)

        # Compute R² score
        r2_scores[i] = r2_score(y_test, y_pred)

    return r2_scores
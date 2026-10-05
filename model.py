"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy as np
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    X = np.asarray(X, dtype = np.float64)
    X_clean = X.copy()
    nb_ligne, nb_colonne = X_clean.shape
    # On calcule la moyenne tout en ignorant les valeurs manquantes
    moyenne = np.nanmean(X_clean, axis=0)
    moyenne = np.nan_to_num(moyenne, nan = 0.0)
    #On cherche tous les NaN on remplace par la moyenne
    X_clean = np.where(np.isnan(X_clean), moyenne,X_clean)

    return X_clean
    pass

# Step 2 - compute_iqr_bounds
import numpy as np
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    q1 = np.percentile(X, 25, axis = 0)
    q3 = np.percentile(X, 75, axis = 0)
    iqr = q3 - q1 
    lower = q1 - (k * iqr)
    upper = q3 + (k * iqr)
    return lower, upper

# Step 3 - clip_columns
import numpy as np
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    assert lower.shape == upper.shape == (X.shape[1],)
    return np.clip(X, lower, upper)

# Step 4 - make_ratio_feature
import numpy as np
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    numerator = np.asarray(numerator).flatten()
    denominator = np.asarray(denominator).flatten()
    r_i = numerator / (denominator + eps)
    return r_i

# Step 5 - append_column
import numpy as np
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    X = np.asarray(X)
    col = np.asarray(col)
    return np.column_stack(([X, col]))

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix
    labels = np.asarray(labels)
    uniques = np.unique(labels)
    return (labels[:, None] == uniques[None, :]).astype(float)

# Step 7 - fit_standardizer
import numpy as np
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    X_mean = np.mean(X, axis = 0)
    X_std = np.std(X, axis=0)
    X_std = np.where(X_std== 0, 1.0, X_std)
    return X_mean, X_std

# Step 8 - apply_standardizer
import numpy as np
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    std = np.where(std==0, 1e-8, std)
    return (X - mean) / std

# Step 9 - add_bias_column
import numpy as np
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    N = X.shape[0]
    b_col = np.ones((N, 1))
    X = np.hstack([b_col, X])
    return X

# Step 10 - make_shuffled_indices
import numpy as np

def make_shuffled_indices(n_samples, seed):
    # 1. On crée le générateur aléatoire avec la graine (seed) pour la reproductibilité
    rng = np.random.RandomState(seed)
    
    # 2. On génère directement la permutation des indices de 0 à n_samples - 1
    indices_melanges = rng.permutation(n_samples)
    
    # 3. On retourne les indices sous forme de liste (ou enlevez .tolist() si vous voulez garder un tableau NumPy)
    return indices_melanges

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    N = len(indices)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    n_test = N -(n_train + n_val)
    id_train = indices[0:n_train]
    id_val = indices[n_train:n_train + n_val]
    id_test = indices[n_train + n_val:]
    return id_train, id_val, id_test

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    X = X[indices]
    y = y[indices]
    return X, y

# Step 13 - ols_fit
import numpy as np
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    A = X.T @ X
    A1 = np.linalg.inv(A)
    b = X.T @ y 
    result = np.linalg.solve(A, b)
    return result

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    pass
    X = np.asarray(X, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64).flatten()
    return np.dot(X , theta)

# Step 15 - mean_absolute_error
import math
import numpy as np
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    erreur = y_pred - y_true
    MAE = np.mean(abs(erreur))
    return float(MAE)

# Step 16 - root_mean_squared_error
import math
import numpy as np
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    erreur_carre = (y_true - y_pred)**2
    MSE =np.mean(erreur_carre)
    RMSE = math.sqrt(MSE)
    return float(RMSE)

    pass

# Step 17 - r_squared
import math
import numpy as np
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    # Convert both inputs to arrays if neeeded and xork with shape(N,)
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    # Calcul du SSRES
    erreur_1 = (y_true - y_pred)**2
    SSRES = np.sum(erreur_1)
    # Calcul du SSTOT
    erreur_2 = (y_true - (y_true).mean())**2
    SSTOT = np.sum(erreur_2)

    if SSTOT == 0:
        return 0.0
    else:
        R2 = 1.0 - (SSRES/SSTOT)
    return R2

# Step 18 - residual_summary
import numpy as np
import math
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals..
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    r = y_true - y_pred
    mean = np.mean(r)
    mean = float(mean)
    std = np.std(r)
    std = float(std)
    median_ab = np.abs(r)
    median_abs = float(np.median(median_ab))
    return {'mean':mean, 'std':std, 'median_abs':median_abs}

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    x_imputed = impute_nan_with_mean(X)
    lower, upper = compute_iqr_bounds(x_imputed, k =iqr_k)
    
    return clip_columns(x_imputed, lower, upper)

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement


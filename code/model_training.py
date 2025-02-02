from sklearn.ensemble import RandomForestClassifier
from xgboost import DMatrix, train
from xgboost import callback as xgb_callback
import joblib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


#Train a Random Forest Classifier.
def train_random_forest(X_train, y_train):

    rf_model = RandomForestClassifier(
        n_estimators=500,  # Number of trees
        max_depth=20,  # Maximum depth of trees
        min_samples_split=10,  # Minimum samples to split a node
        min_samples_leaf=5,  # Minimum samples in a leaf node
        n_jobs=-1  # Parallel processing
    )
    rf_model.fit(X_train, y_train)
    return rf_model

# Train an XGBoost Classifier using the native XGBoost Learning API.
def train_xgboost(X_train, y_train, X_test, y_test):
    # Create DMatrix for train and test
    dtrain = DMatrix(X_train, label=y_train)
    dtest = DMatrix(X_test, label=y_test)

    # Define model parameters
    params = {
        'objective': 'binary:logistic',  # Binary classification
        'eval_metric': 'logloss',  # Log loss as evaluation metric
        'learning_rate': 0.05,  # Lower learning rate for more stable learning
        'max_depth': 6,
        'subsample': 0.8,  # Subsample training data to avoid overfitting
        'colsample_bytree': 0.8,  # Sample features to avoid overfitting
        'lambda': 1,  # L2 regularization
        'alpha': 1,  # L1 regularization
        'random_state': 42
    }

    # Defining early stopping callback
    early_stopping = xgb_callback.EarlyStopping(
        rounds=10,  # Stops after 10 rounds without improvement
        save_best=True
    )

    # Train the model
    xgb_model = train(
        params=params,
        dtrain=dtrain,
        num_boost_round=500,  # Increased boosting rounds to allow more learning
        evals=[(dtrain, 'train'), (dtest, 'test')],  # Evaluation sets for train and test
        callbacks=[early_stopping]  # Include early stopping callback
    )

    return xgb_model

  # Train an SVM model using LinearSVC, scale the data, evaluate performance.
def train_svm(X_train, y_train, X_test, y_test):

    # Use StandardScaler with with_mean=False for sparse matrices
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train SVM model
    model = LinearSVC(C=0.5, max_iter=8000)
    model.fit(X_train_scaled, y_train)
    return model, X_test_scaled



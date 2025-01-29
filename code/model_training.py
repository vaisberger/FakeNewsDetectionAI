from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import DMatrix, train
from xgboost import callback as xgb_callback


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest Classifier.
    """
    rf_model = RandomForestClassifier(
        n_estimators=500,  # Number of trees
        max_depth=20,  # Maximum depth of trees
        min_samples_split=10,  # Minimum samples to split a node
        min_samples_leaf=5,  # Minimum samples in a leaf node
        n_jobs=-1  # Parallel processing
    )
    rf_model.fit(X_train, y_train)
    return rf_model


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost Classifier using the native XGBoost Learning API.
    """
    # Create DMatrix for train and test
    dtrain = DMatrix(X_train, label=y_train)
    dtest = DMatrix(X_test, label=y_test)

    # Define model parameters
    params = {
        'objective': 'binary:logistic',  # Binary classification
        'eval_metric': 'logloss',  # Log loss as evaluation metric
        'learning_rate': 0.05,  # Lower learning rate for more stable learning
        'max_depth': 6,  # Adjust this based on experimentation
        'subsample': 0.8,  # Subsample training data to avoid overfitting
        'colsample_bytree': 0.8,  # Sample features to avoid overfitting
        'lambda': 1,  # L2 regularization
        'alpha': 1,  # L1 regularization
        'random_state': 42  # Ensure reproducibility
    }

    # Define early stopping callback
    early_stopping = xgb_callback.EarlyStopping(
        rounds=10,  # Stop after 10 rounds without improvement
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


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """
    Train a Gradient Boosting Classifier.
    """
    # Initialize the Gradient Boosting model
    gb_model = GradientBoostingClassifier(random_state=0)

    # Train the Gradient Boosting model
    gb_model.fit(X_train, y_train)

    # Predict using the trained Gradient Boosting model
    pred_gb = gb_model.predict(X_test)

    return gb_model, pred_gb






from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, class_weight=None):
    """
    Trains a Random Forest Classifier with optional hyperparameters.

    Parameters:
        X_train (DataFrame): Features for training the model.
        y_train (Series): Labels for training the model.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the trees.
        class_weight (dict or 'balanced', optional): Class weights for handling imbalanced data.

    Returns:
        model: Trained Random Forest model.
    """
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, learning_rate=0.1, scale_pos_weight=None):
    """
    Trains an XGBoost Classifier with optional hyperparameters.

    Parameters:
        X_train (DataFrame): Features for training the model.
        y_train (Series): Labels for training the model.
        n_estimators (int): Number of boosting rounds.
        max_depth (int): Maximum depth of trees.
        learning_rate (float): Boosting learning rate.
        scale_pos_weight (float, optional): Balances positive and negative weights for imbalanced data.

    Returns:
        model: Trained XGBoost model.
    """
    model = XGBClassifier(
        random_state=42,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        eval_metric = "logloss"
    )
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    print("This file defines functions for training Random Forest and XGBoost models with customizable parameters.")


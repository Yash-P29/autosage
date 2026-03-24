"""
AutoSage Configuration
Global constants, color palette, model registry, and hyperparameter grids.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ──────────────────────────────────────────────
# Theme & Color Palette
# ──────────────────────────────────────────────
COLORS = {
    "primary": "#6C63FF",
    "secondary": "#00D2FF",
    "accent": "#FF6584",
    "success": "#00E676",
    "warning": "#FFD600",
    "danger": "#FF5252",
    "bg_dark": "#0E1117",
    "bg_card": "#1A1E2E",
    "bg_card_hover": "#252A3A",
    "text_primary": "#FFFFFF",
    "text_secondary": "#A0AEC0",
    "gradient_start": "#6C63FF",
    "gradient_end": "#00D2FF",
}

# ──────────────────────────────────────────────
# Feature Selection Defaults
# ──────────────────────────────────────────────
VARIANCE_THRESHOLD = 0.01
K_BEST_DEFAULT = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
N_ITER_SEARCH = 20

# ──────────────────────────────────────────────
# Model Registry
# ──────────────────────────────────────────────
MODEL_REGISTRY = {
    "Logistic Regression": {
        "class": LogisticRegression,
        "default_params": {"max_iter": 1000, "random_state": RANDOM_STATE},
        "param_grid": {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "liblinear"],
        },
        "icon": "📈",
    },
    "Random Forest": {
        "class": RandomForestClassifier,
        "default_params": {"random_state": RANDOM_STATE},
        "param_grid": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [3, 5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "icon": "🌲",
    },
    "Support Vector Machine": {
        "class": SVC,
        "default_params": {"probability": True, "random_state": RANDOM_STATE},
        "param_grid": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
        "icon": "🧠",
    },
    "XGBoost": {
        "class": XGBClassifier,
        "default_params": {
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
            "random_state": RANDOM_STATE,
        },
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 1.0],
        },
        "icon": "🚀",
    },
    "K-Nearest Neighbors": {
        "class": KNeighborsClassifier,
        "default_params": {},
        "param_grid": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "icon": "📍",
    },
    "Decision Tree": {
        "class": DecisionTreeClassifier,
        "default_params": {"random_state": RANDOM_STATE},
        "param_grid": {
            "max_depth": [3, 5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        },
        "icon": "🌳",
    },
    "Gradient Boosting": {
        "class": GradientBoostingClassifier,
        "default_params": {"random_state": RANDOM_STATE},
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 1.0],
        },
        "icon": "📊",
    },
    "Extra Trees": {
        "class": ExtraTreesClassifier,
        "default_params": {"random_state": RANDOM_STATE},
        "param_grid": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [3, 5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
        },
        "icon": "🌴",
    },
}

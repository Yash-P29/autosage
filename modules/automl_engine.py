"""
AutoSage — AutoML Engine
Model training with RandomizedSearchCV hyperparameter tuning.
"""

import time
import warnings
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from config import MODEL_REGISTRY, CV_FOLDS, N_ITER_SEARCH, RANDOM_STATE

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings("ignore")


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_models: list,
    n_iter: int = N_ITER_SEARCH,
    cv_folds: int = CV_FOLDS,
    tuning_strategy: str = "random",
    progress_callback=None,
) -> dict:
    """
    Train and tune selected models.

    Args:
        X_train: Training features
        y_train: Training labels
        selected_models: List of model names from MODEL_REGISTRY
        n_iter: Number of RandomizedSearchCV/Optuna iterations
        cv_folds: Number of cross-validation folds
        tuning_strategy: 'random', 'grid', or 'optuna'
        progress_callback: Optional callable(model_name, step, total) for progress updates

    Returns:
        Dict of {model_name: {model, best_params, cv_score, train_time}}
    """
    results = {}
    total = len(selected_models)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    for idx, model_name in enumerate(selected_models):
        if model_name not in MODEL_REGISTRY:
            continue

        reg = MODEL_REGISTRY[model_name]
        if progress_callback:
            progress_callback(model_name, idx, total)

        start_time = time.time()

        try:
            # Instantiate with defaults
            base_model = reg["class"](**reg["default_params"])
            param_grid = reg["param_grid"]

            # Calculate actual n_iter (can't exceed total combinations)
            n_combinations = 1
            for v in param_grid.values():
                n_combinations *= len(v)
            actual_n_iter = min(n_iter, n_combinations)

            if actual_n_iter < 2:
                # Too few combinations — train directly
                base_model.fit(X_train, y_train)
                train_score = accuracy_score(y_train, base_model.predict(X_train))
                results[model_name] = {
                    "model": base_model,
                    "best_params": reg["default_params"],
                    "cv_score": train_score,
                    "train_time": round(time.time() - start_time, 2),
                }
            elif tuning_strategy == "optuna" and OPTUNA_AVAILABLE:
                def objective(trial):
                    params = {}
                    for k, v in param_grid.items():
                        # Handle varied types by using suggest_categorical for the grid values
                        params[k] = trial.suggest_categorical(k, v)
                    
                    model = reg["class"](**reg["default_params"])
                    model.set_params(**params)
                    score = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean()
                    return score

                study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
                # Add a timeout to prevent hanging
                study.optimize(objective, n_trials=actual_n_iter, timeout=60, n_jobs=1)
                
                best_model = reg["class"](**reg["default_params"])
                best_model.set_params(**study.best_params)
                best_model.fit(X_train, y_train)
                
                results[model_name] = {
                    "model": best_model,
                    "best_params": study.best_params,
                    "cv_score": round(study.best_value, 4),
                    "train_time": round(time.time() - start_time, 2),
                }
            elif tuning_strategy == "grid":
                search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=cv,
                    scoring="accuracy",
                    n_jobs=-1,
                    error_score="raise",
                )
                search.fit(X_train, y_train)
                results[model_name] = {
                    "model": search.best_estimator_,
                    "best_params": search.best_params_,
                    "cv_score": round(search.best_score_, 4),
                    "train_time": round(time.time() - start_time, 2),
                }
            else:
                search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_grid,
                    n_iter=actual_n_iter,
                    cv=cv,
                    scoring="accuracy",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    error_score="raise",
                )
                search.fit(X_train, y_train)

                results[model_name] = {
                    "model": search.best_estimator_,
                    "best_params": search.best_params_,
                    "cv_score": round(search.best_score_, 4),
                    "train_time": round(time.time() - start_time, 2),
                }

        except Exception as e:
            # If something fails, try training with defaults only
            try:
                fallback = reg["class"](**reg["default_params"])
                fallback.fit(X_train, y_train)
                results[model_name] = {
                    "model": fallback,
                    "best_params": reg["default_params"],
                    "cv_score": round(accuracy_score(y_train, fallback.predict(X_train)), 4),
                    "train_time": round(time.time() - start_time, 2),
                    "warning": f"Hyperparameter tuning failed ({str(e)[:80]}), used defaults.",
                }
            except Exception as e2:
                results[model_name] = {
                    "model": None,
                    "best_params": {},
                    "cv_score": 0.0,
                    "train_time": round(time.time() - start_time, 2),
                    "error": str(e2)[:120],
                }

    return results


def get_best_model(results: dict) -> tuple:
    """Return (model_name, result_dict) for the best performing model."""
    valid = {k: v for k, v in results.items() if v.get("model") is not None}
    if not valid:
        return None, None
    best_name = max(valid, key=lambda k: valid[k]["cv_score"])
    return best_name, valid[best_name]

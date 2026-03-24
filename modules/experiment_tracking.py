"""
AutoSage — Experiment Tracking Module
Lightweight SQLite/JSON wrapper to track experiments and datasets.
"""

import sqlite3
import pandas as pd
import json
import hashlib
from datetime import datetime

DB_PATH = "autosage_experiments.db"

def _init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            dataset_hash TEXT,
            target_col TEXT,
            model_name TEXT,
            cv_score REAL,
            test_accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            best_params TEXT
        )
    ''')
    conn.commit()
    conn.close()

def hash_dataset(df: pd.DataFrame) -> str:
    """Create a unique hash for the dataset to track internal versions."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def log_experiment(dataset_hash: str, target_col: str, model_name: str, 
                   cv_score: float, test_metrics: dict, best_params: dict):
    """Log a single experiment run."""
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO experiments (
            timestamp, dataset_hash, target_col, model_name, 
            cv_score, test_accuracy, precision, recall, f1_score, best_params
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        dataset_hash,
        target_col,
        model_name,
        cv_score,
        test_metrics.get("accuracy", 0.0),
        test_metrics.get("precision", 0.0),
        test_metrics.get("recall", 0.0),
        test_metrics.get("f1_score", 0.0),
        json.dumps({k: str(v) for k, v in best_params.items()})
    ))
    conn.commit()
    conn.close()

def load_experiments() -> pd.DataFrame:
    """Load all tracked experiments into a DataFrame."""
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM experiments ORDER BY timestamp DESC", conn)
    conn.close()
    return df

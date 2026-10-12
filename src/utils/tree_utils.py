### Libraries ###
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
try:
    from zss import simple_distance, Node
except ImportError:
    simple_distance, Node = None, None

# ==========================================
# 1. TREE CONVERTERS (PySORTD / Sklearn -> ZSS)
# ==========================================
def _format_leaf_label(label) -> str:
    """Integer class labels stay Class:k. Real-valued leaves stay numeric."""
    value = float(np.asarray(label).reshape(-1)[0])
    if np.isfinite(value) and abs(value - round(value)) < 1e-8:
        return f"Class:{int(round(value))}"
    return f"Leaf:{value:.6g}"


def pysortd_to_zss(node):
    """Converts a live PySORTD C++ node to a ZSS Node."""
    if node.is_leaf_node():
        return Node(_format_leaf_label(node.label))
    else:
        zss_node = Node(f"Feat:{int(node.feature)}")
        if hasattr(node, 'left_child'):
            zss_node.addkid(pysortd_to_zss(node.left_child))
        if hasattr(node, 'right_child'):
            zss_node.addkid(pysortd_to_zss(node.right_child))
        return zss_node

def sklearn_to_zss(tree, node_id=0):
    """Converts a Scikit-Learn Tree (Oracle) to a ZSS Node."""
    left = tree.children_left[node_id]
    right = tree.children_right[node_id]

    if left == -1:  # Leaf
        try:
            value = np.asarray(tree.value[node_id])
            if value.shape[-1] == 1:
                return Node(_format_leaf_label(value.reshape(-1)[0]))
            class_label = int(np.argmax(value))
        except (IndexError, TypeError, ValueError):
            class_label = 0
        return Node(f"Class:{class_label}")
    else: # Split
        feature_idx = tree.feature[node_id]
        zss_node = Node(f"Feat:{feature_idx}")
        zss_node.addkid(sklearn_to_zss(tree, left))
        zss_node.addkid(sklearn_to_zss(tree, right))
        return zss_node

def _extract_root(model):
    """Helper to find the root ZSS node regardless of Wrapper type."""
    # 1. Unwrap the wrapper if needed
    inner = model.model if hasattr(model, 'model') else model

    # 2. Check for PySORTD active tree (Standard Wrapper)
    if hasattr(inner, 'tree_') and hasattr(inner.tree_, 'is_leaf_node'):
        return pysortd_to_zss(inner.tree_)
    
    # 3. Check for Scikit-Learn Decision Tree (Standard Arrays)
    if hasattr(inner, 'tree_') and hasattr(inner.tree_, 'children_left'):
        return sklearn_to_zss(inner.tree_)
    
    # 4. Random Forests and Logistic Regression do NOT have a single root so return None
    return None

# ==========================================
# 2. METRIC CALCULATORS
# ==========================================

def calculate_oracle_agreement(current_model, oracle_model, df_test, task="classification"):
    """
    Classification: fraction of test rows where the two models predict the same class.
    Regression: correlation between the two models' predictions.
    """
    X_test = df_test.drop(columns="Y")
    pred_current = np.asarray(current_model.predict(X_test), dtype=float).reshape(-1)
    pred_oracle = np.asarray(oracle_model.predict(X_test), dtype=float).reshape(-1)
    if task == "regression":
        if np.std(pred_current) < 1e-12 or np.std(pred_oracle) < 1e-12:
            return 0.0
        return float(np.corrcoef(pred_current, pred_oracle)[0, 1])
    return float(np.mean(pred_current == pred_oracle))

def calculate_ted_score(model, oracle):
    """
    Calculates Tree Edit Distance (TED).
    Returns -1 if ZSS is missing or models are not single trees.
    """
    if simple_distance is None: 
        return -1.0

    try:
        oracle_root = _extract_root(oracle)
        model_root = _extract_root(model)

        if oracle_root is None or model_root is None:
            return -1.0

        return float(simple_distance(model_root, oracle_root))
    except Exception:
        return -1.0

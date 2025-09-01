# Hungarian matching for node correspondence (not fully implemented for brevity).
# In this code, we simplified by aligning nodes by index and using padding.
# If needed, one could implement graph matching here for better reconstruction accuracy.
import numpy as np
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

def match_prediction_to_target(pred_adj, target_adj, pred_node_feats, target_node_feats):
    """
    Align the predicted graph's nodes to target graph nodes using the Hungarian algorithm (on degree/type similarity).
    Returns permuted predicted adjacency and node features.
    """
    N = target_adj.shape[0]
    if linear_sum_assignment is None or pred_adj.shape[0] != N:
        # No matching if SciPy not available or sizes differ
        return pred_adj, pred_node_feats
    # Compute cost matrix based on node degree difference and feature difference
    cost = np.zeros((N, N))
    # Degrees
    pred_degrees = pred_adj.sum(axis=1)
    target_degrees = target_adj.sum(axis=1)
    for i in range(N):
        for j in range(N):
            # Type cost: 0 if same type, 1 if different (using argmax of one-hot)
            pred_type = np.argmax(pred_node_feats[i])
            target_type = np.argmax(target_node_feats[j])
            type_cost = 0 if pred_type == target_type else 1
            # Degree cost: absolute difference
            deg_cost = abs(pred_degrees[i] - target_degrees[j])
            cost[i, j] = type_cost + deg_cost
    row_ind, col_ind = linear_sum_assignment(cost)
    # Permute prediction according to assignment
    perm = np.argsort(col_ind)
    pred_adj_perm = pred_adj[row_ind][:, row_ind]  # align rows and cols
    pred_node_feats_perm = pred_node_feats[row_ind]
    return pred_adj_perm, pred_node_feats_perm

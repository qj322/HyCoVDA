from typing import Tuple, Dict, Iterable, Set
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, accuracy_score


def load_indices_txt(path: str) -> Dict[str, int]:
    mp: Dict[str, int] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t') if ('\t' in line) else line.split()
            if len(parts) < 2:
                continue
            name, idx = parts[0], int(parts[1])
            mp[name] = idx
    return mp


def _parse_two_ints(line: str) -> Tuple[int, int] | None:
    if not line:
        return None
    parts = line.split()
    if len(parts) < 2:
        return None
    try:
        d = int(parts[0]); v = int(parts[1])
        return d, v
    except Exception:
        return None


def load_pairs_txt(path: str) -> np.ndarray:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            p = _parse_two_ints(line.strip())
            if p is not None:
                pairs.append(p)
    return np.array(pairs, dtype=np.int64)



def build_bipartite_adj(num_drugs: int, num_viruses: int, pos_pairs: np.ndarray) -> sp.csr_matrix:
    if len(pos_pairs) == 0:
        return sp.csr_matrix((num_drugs, num_viruses), dtype=np.float32)
    rows = pos_pairs[:, 0]
    cols = pos_pairs[:, 1]
    data = np.ones(len(rows), dtype=np.float32)
    mat = coo_matrix((data, (rows, cols)), shape=(num_drugs, num_viruses), dtype=np.float32)
    return mat.tocsr()


def row_stochasticize_csr(adj: sp.csr_matrix) -> sp.csr_matrix:
    adj = adj.tocsr(copy=True)
    row_sum = np.array(adj.sum(axis=1)).flatten()
    row_sum[row_sum == 0.0] = 1.0
    for i in range(adj.shape[0]):
        s, e = adj.indptr[i], adj.indptr[i+1]
        if e > s:
            adj.data[s:e] /= row_sum[i]
    return adj


def build_incidence_from_bipartite(train_adj: sp.csr_matrix, drop_size1: bool = False) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    D, V = train_adj.shape
    csc = train_adj.tocsc()
    d_rows, d_cols = [], []
    e_id = 0
    for v in range(V):
        s, e = csc.indptr[v], csc.indptr[v+1]
        members = csc.indices[s:e]
        if drop_size1 and members.size < 2:
            continue
        for d in members:
            d_rows.append(d); d_cols.append(e_id)
        e_id += 1
    H_d = (coo_matrix((np.ones(len(d_rows), dtype=np.float32), (np.array(d_rows), np.array(d_cols))),
                      shape=(D, (max(d_cols)+1 if d_cols else 1)), dtype=np.float32).tocsr()
           if len(d_rows) else sp.csr_matrix((D, 1), dtype=np.float32))
    d_rows2, d_cols2 = [], []
    e2 = 0
    for d in range(D):
        s, e = train_adj.indptr[d], train_adj.indptr[d+1]
        members = train_adj.indices[s:e]
        if drop_size1 and members.size < 2:
            continue
        for v in members:
            d_rows2.append(v); d_cols2.append(e2)
        e2 += 1
    H_v = (coo_matrix((np.ones(len(d_rows2), dtype=np.float32), (np.array(d_rows2), np.array(d_cols2))),
                      shape=(V, (max(d_cols2)+1 if d_cols2 else 1)), dtype=np.float32).tocsr()
           if len(d_rows2) else sp.csr_matrix((V, 1), dtype=np.float32))
    return H_d, H_v


def to_torch_sparse_from_scipy(m: sp.coo_matrix, device: torch.device) -> torch.Tensor:
    m = m.tocoo()
    idx = np.vstack([m.row, m.col]).astype(np.int64)
    dat = m.data.astype(np.float32) if m.data.size > 0 else np.array([0.0], dtype=np.float32)
    idx_t = torch.from_numpy(idx).to(device)
    dat_t = torch.from_numpy(dat).to(device)
    return torch.sparse_coo_tensor(idx_t, dat_t, m.shape, device=device)



def build_pos_set(pairs: Iterable[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    return set((int(d), int(v)) for d, v in pairs)


def sample_fixed_negatives(num_drugs: int,
                           num_viruses: int,
                           pos_set: Set[Tuple[int, int]],
                           count: int,
                           seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    negs = []
    tried = 0
    count = int(max(1, count))
    while len(negs) < count and tried < count * 200:
        d = int(rng.randint(0, num_drugs)); v = int(rng.randint(0, num_viruses))
        if (d, v) not in pos_set:
            negs.append((d, v))
        tried += 1
    if len(negs) < count:
        all_pairs = [(d, v) for d in range(num_drugs) for v in range(num_viruses) if (d, v) not in pos_set]
        replace = len(all_pairs) < (count - len(negs))
        extra_idx = rng.choice(len(all_pairs), size=(count - len(negs)), replace=replace)
        negs.extend([all_pairs[i] for i in extra_idx])
    return np.array(negs, dtype=np.int64)



@torch.no_grad()
def eval_on_pairs(model,
                  pairs_pos: np.ndarray,
                  num_drugs: int,
                  num_viruses: int,
                  fixed_neg_pairs: np.ndarray,
                  device: torch.device = torch.device('cpu'),
                  batch: int = 4096) -> Tuple[float, float, float, float, float, float]:
    """
    Calculates AUC, AUPR, and threshold-based metrics (F1, Precision, Recall, Accuracy)
    at the optimal F1 threshold.
    """
    model.eval()
    all_pairs = np.vstack([pairs_pos, fixed_neg_pairs]) if len(fixed_neg_pairs) > 0 else pairs_pos
    labels = np.array([1] * len(pairs_pos) + [0] * len(fixed_neg_pairs), dtype=np.int64)

    scores = []
    for i in range(0, len(all_pairs), batch):
        chunk = all_pairs[i:i+batch]
        d = torch.tensor(chunk[:, 0], dtype=torch.long, device=device)
        v = torch.tensor(chunk[:, 1], dtype=torch.long, device=device)
        pred, _, _, _ = model(d, v)
        scores.append(pred.detach().cpu().numpy())
    scores = np.concatenate(scores, axis=0)

    auc  = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5
    aupr = average_precision_score(labels, scores)

    precision, recall, thresholds = precision_recall_curve(labels, scores)


    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)

 
    best_f1_idx = np.argmax(f1_scores)

    best_f1 = f1_scores[best_f1_idx]
    best_precision = precision[best_f1_idx]
    best_recall = recall[best_f1_idx]


    best_threshold = thresholds[best_f1_idx]

    y_pred_at_best_f1 = (scores >= best_threshold).astype(int)
    accuracy = accuracy_score(labels, y_pred_at_best_f1)

    return float(auc), float(aupr), float(best_f1), float(best_precision), float(best_recall), float(accuracy)






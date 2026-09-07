import os
import numpy as np
import torch

from dataset import (
    load_indices_txt, load_pairs_txt, build_bipartite_adj, 
    build_incidence_from_bipartite, to_torch_sparse_from_scipy, eval_on_pairs,
    build_pos_set, sample_fixed_negatives
)
from model import HyCoVDA


DATA_DIR          = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
PATH_DRUG_INDEX   = os.path.join(DATA_DIR, "drug_index.txt")
PATH_VIRUS_INDEX  = os.path.join(DATA_DIR, "disease_index.txt")
PATH_GLOBAL_POS   = os.path.join(DATA_DIR, "drug_disease.txt")
FOLD_TRAIN_FILES  = [os.path.join(DATA_DIR, f"train_{i}.txt") for i in range(5)]
FOLD_TEST_FILES   = [os.path.join(DATA_DIR, f"test_{i}.txt") for i in range(5)]
RESULTS_DIR       = os.path.join(os.path.dirname(__file__), "results_vda_folds")

SEED              = 2025
LATDIM            = 128
GAT_DROPOUT       = 0.1
GAT_ALPHA         = 0.2
TEMPERATURE       = 0.05
EDGE_DROP_RATE    = 0 
DEG_LOSS_WEIGHT   = 0.4
LAYERS            = 1
TEST_NEG_PER_POS  = 1    
VALIDATION_RATIO  = 0.1

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def split_train_validation(train_pos: np.ndarray, fold_id: int,
                           validation_ratio: float = VALIDATION_RATIO):
    """Mirror train.py's deterministic split so checkpoint graphs are identical."""
    if len(train_pos) < 2 or validation_ratio <= 0:
        return train_pos.copy(), np.empty((0, 2), dtype=np.int64)
    rng = np.random.RandomState(SEED + 5000 + fold_id)
    order = rng.permutation(len(train_pos))
    n_val = min(max(1, int(round(len(train_pos) * validation_ratio))), len(train_pos) - 1)
    return train_pos[order[n_val:]], train_pos[order[:n_val]]

def main():
    device = get_device()
    print(f"Device: {device}")


    drug_map  = load_indices_txt(PATH_DRUG_INDEX) if os.path.exists(PATH_DRUG_INDEX) else {}
    virus_map = load_indices_txt(PATH_VIRUS_INDEX) if os.path.exists(PATH_VIRUS_INDEX) else {}

    global_pos = load_pairs_txt(PATH_GLOBAL_POS) if os.path.exists(PATH_GLOBAL_POS) else np.empty((0,2), dtype=np.int64)
    global_pos_set = build_pos_set(global_pos)


    folds = []
    max_d = -1; max_v = -1
    for i in range(5):
        tr = load_pairs_txt(FOLD_TRAIN_FILES[i])
        te = load_pairs_txt(FOLD_TEST_FILES[i])
        folds.append((tr, te))
        if len(tr) > 0:
            max_d = max(max_d, int(tr[:,0].max())); max_v = max(max_v, int(tr[:,1].max()))
        if len(te) > 0:
            max_d = max(max_d, int(te[:,0].max())); max_v = max(max_v, int(te[:,1].max()))
    if len(global_pos) > 0:
        max_d = max(max_d, int(global_pos[:,0].max()))
        max_v = max(max_v, int(global_pos[:,1].max()))

    num_drugs   = max(max(drug_map.values()) + 1 if len(drug_map) > 0 else 0, max_d + 1)
    num_viruses = max(max(virus_map.values()) + 1 if len(virus_map) > 0 else 0, max_v + 1)
    print(f"#Drugs={num_drugs}, #Viruses={num_viruses}, #GlobalPos={len(global_pos)}")

    all_aucs, all_auprs = [], []
    all_f1s, all_precisions, all_recalls, all_accuracies = [], [], [], []

    for fold_id, (train_pos, test_pos) in enumerate(folds, start=0):
        print(f"\n===== Eval Fold {fold_id} =====  train={len(train_pos)}  test={len(test_pos)}")


        fit_pos, val_pos = split_train_validation(train_pos, fold_id)
        train_adj = build_bipartite_adj(num_drugs, num_viruses, fit_pos)
        adj_tensor = to_torch_sparse_from_scipy(train_adj.tocoo(), device)
        

        H_d_sp, H_v_sp = build_incidence_from_bipartite(train_adj, drop_size1=False)
        H_d_tensor = to_torch_sparse_from_scipy(H_d_sp.tocoo(), device)
        H_v_tensor = to_torch_sparse_from_scipy(H_v_sp.tocoo(), device)


        model = HyCoVDA(num_drugs, num_viruses,
                           adj_dv=adj_tensor, 
                           H_d=H_d_tensor, H_v=H_v_tensor,
                           latdim=LATDIM,
                           gat_dropout=GAT_DROPOUT, gat_alpha=GAT_ALPHA,
                           temperature=TEMPERATURE,
                           edge_drop_rate=EDGE_DROP_RATE, 
                           n_layers = LAYERS,
                           device=device).to(device)


        ckpt_path = os.path.join(RESULTS_DIR, f"best_model_fold{fold_id}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[Fold {fold_id}] WARNING: checkpoint not found -> {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval() # Set model to evaluation mode


        test_neg_fixed = sample_fixed_negatives(
            num_drugs, num_viruses, global_pos_set,
            len(test_pos) * TEST_NEG_PER_POS,
            seed=SEED + 1000 * fold_id
        )

       
        test_auc, test_aupr, test_f1, test_precision, test_recall, test_acc = eval_on_pairs(
            model, test_pos, num_drugs, num_viruses,
            fixed_neg_pairs=test_neg_fixed, device=device
        )

       
        rec_auc  = ckpt.get("val_auc", None)
        rec_aupr = ckpt.get("val_aupr", None)
        print(f"[Fold {fold_id}] Loaded ckpt epoch={ckpt.get('epoch','?')}")
        if rec_auc is not None and rec_aupr is not None:
            print(f"[Fold {fold_id}] ckpt selection metric: Val AUC={rec_auc:.4f}  AUPR={rec_aupr:.4f}")
        print(f"[Fold {fold_id}] re-evaluated: Test AUC={test_auc:.4f}  AUPR={test_aupr:.4f}  ({ckpt_path})")
        print(f"[Fold {fold_id}] (at best F1): Precision={test_precision:.4f}  Recall={test_recall:.4f}  Accuracy={test_acc:.4f}")

        all_aucs.append(test_auc)
        all_auprs.append(test_aupr)
        all_f1s.append(test_f1)
        all_precisions.append(test_precision)
        all_recalls.append(test_recall)
        all_accuracies.append(test_acc)

    if len(all_aucs) > 0:
        print("\n===== Evaluation Summary =====")
        for i, (a, p) in enumerate(zip(all_aucs, all_auprs), start=0):
            print(f"Fold {i}: AUC={a:.4f}  AUPR={p:.4f} F1={all_f1s[i]:.4f}  Acc={all_accuracies[i]:.4f}")
        print(f"Mean AUC={np.mean(all_aucs):.4f} ± {np.std(all_aucs):.4f}")
        print(f"Mean AUPR={np.mean(all_auprs):.4f} ± {np.std(all_auprs):.4f}")
        print(f"Mean F1={np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")
        print(f"Mean Precision={np.mean(all_precisions):.4f} ± {np.std(all_precisions):.4f}")
        print(f"Mean Recall={np.mean(all_recalls):.4f} ± {np.std(all_recalls):.4f}")
        print(f"Mean Accuracy={np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
    else:
        print("\nNo folds evaluated (missing checkpoints?).")

if __name__ == "__main__":
    main()

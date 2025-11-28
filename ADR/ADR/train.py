import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from dataset import (
    load_indices_txt, load_pairs_txt, build_bipartite_adj, row_stochasticize_csr,
    build_incidence_from_bipartite, to_torch_sparse_from_scipy, eval_on_pairs,
    build_pos_set, sample_fixed_negatives
)
from model import CoBiHADR  


PATH_DRUG_INDEX   = "/home/qhjiang/works/ADR/drug_index.txt" 
PATH_VIRUS_INDEX  = "/home/qhjiang/works/ADR/disease_index.txt"
PATH_GLOBAL_POS   = "/home/qhjiang/works/ADR/drug_disease.txt"
FOLD_TRAIN_FILES  = [f"/home/qhjiang/works/ADR/train_{i}.txt" for i in range(5)]
FOLD_TEST_FILES   = [f"/home/qhjiang/works/ADR/test_{i}.txt"  for i in range(5)]

RESULTS_DIR       = "./results_adr_folds" 
SEED              = 2025
EPOCHS            = 1500
BATCH_SIZE        = 512
LR                = 5e-4
LATDIM            = 128
GAT_DROPOUT       = 0.1
GAT_ALPHA         = 0.2
TEMPERATURE       = 0.05
EDGE_DROP_RATE    = 0 
CONTRAST_WEIGHT   = 0.3
DEG_LOSS_WEIGHT   = 0.4
LAYERS            = 1
NEG_PER_POS       = 1
TEST_NEG_PER_POS  = 1


def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return F.softplus(-(pos_scores - neg_scores)).mean()


def build_global_pos_map(pos_pairs: np.ndarray, num_drugs: int) -> list:
    mp = [set() for _ in range(num_drugs)]
    for d, v in pos_pairs:
        mp[int(d)].add(int(v))
    return mp


def sample_negatives_for_drug(d: int, num_viruses: int, forbid_set: set, k: int, rng: np.random.RandomState) -> list:
    negs = []
    tried = 0
    k = int(k)
    while len(negs) < k and tried < k * 50:
        v = int(rng.randint(0, num_viruses))
        if v not in forbid_set:
            negs.append(v)
        tried += 1
    if len(negs) < k:
        pool = [x for x in range(num_viruses) if x not in forbid_set]
        if len(pool) == 0:
            pool = list(range(num_viruses))
        extra = list(rng.choice(pool, size=k-len(negs), replace=len(pool)<(k-len(negs))))
        negs.extend(extra)
    return negs


def train_one_fold(model: CoBiHADR,
                   train_adj,
                   train_pos,
                   test_pos,
                   global_pos_pairs,
                   device: torch.device,
                   fold_id: int) -> str:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_csr = train_adj.tocsr()
    global_pos_map = build_global_pos_map(global_pos_pairs, model.D)
    rng = np.random.RandomState(SEED + fold_id)

    global_pos_set = build_pos_set(global_pos_pairs)
    test_neg_fixed = sample_fixed_negatives(model.D, model.V, global_pos_set,
                                            len(test_pos) * TEST_NEG_PER_POS,
                                            seed=SEED + 1000 * fold_id)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_test_auc = -1.0
    best_path = os.path.join(RESULTS_DIR, f"best_model_fold{fold_id}.pt")

    for ep in range(1, EPOCHS + 1):
        model.train()
        drug_ids = rng.permutation(model.D)
        steps = int(np.ceil(len(drug_ids) / BATCH_SIZE))
        total_loss = 0.0
        for s in range(steps):
            bs = drug_ids[s * BATCH_SIZE : (s + 1) * BATCH_SIZE]
            pos_d, pos_v, neg_d, neg_v = [], [], [], []
            for d in bs:
                row = train_csr.getrow(int(d))
                pos_vs = row.indices
                if pos_vs.size == 0: continue
                v_pos = int(rng.choice(pos_vs))
                forb = global_pos_map[int(d)]
                v_negs = sample_negatives_for_drug(int(d), model.V, forb, k=NEG_PER_POS, rng=rng)
                pos_d.append(int(d)); pos_v.append(v_pos)
                neg_d.extend([int(d)] * len(v_negs)); neg_v.extend(v_negs)

            if len(pos_d) == 0: continue

            pos_d_t, pos_v_t = torch.tensor(pos_d, dtype=torch.long, device=device), torch.tensor(pos_v, dtype=torch.long, device=device)
            neg_d_t, neg_v_t = torch.tensor(neg_d, dtype=torch.long, device=device), torch.tensor(neg_v, dtype=torch.long, device=device)

            s_pos, cl_pos, deg_d_pos, deg_v_pos = model(pos_d_t, pos_v_t)
            s_pos_rep = s_pos.repeat_interleave(NEG_PER_POS)
            s_neg, _, deg_d_neg, deg_v_neg  = model(neg_d_t, neg_v_t)

            loss_bpr = bpr_loss(s_pos_rep, s_neg)

            loss_cl = cl_pos

            all_drug_ids = torch.cat([pos_d_t, neg_d_t])
            all_virus_ids = torch.cat([pos_v_t, neg_v_t])
            all_deg_d_pred = torch.cat([deg_d_pos, deg_d_neg])
            all_deg_v_pred = torch.cat([deg_v_pos, deg_v_neg])
            tgt_d = model.deg_d_target.index_select(0, all_drug_ids)
            tgt_v = model.deg_v_target.index_select(0, all_virus_ids)
            deg_loss = F.mse_loss(all_deg_d_pred, tgt_d) + F.mse_loss(all_deg_v_pred, tgt_v)

            loss = loss_bpr + CONTRAST_WEIGHT * loss_cl + DEG_LOSS_WEIGHT * deg_loss

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += float(loss.item())


        test_auc, test_aupr, test_f1, test_precision, test_recall, test_acc = eval_on_pairs(
            model, test_pos, model.D, model.V,
            fixed_neg_pairs=test_neg_fixed, device=device
        )
        
        print(f"[Fold {fold_id}] Ep {ep:03d} loss={total_loss/max(1,steps):.4f}  Test AUC={test_auc:.4f} AUPR={test_aupr:.4f} F1={test_f1:.4f}")

        if test_auc > best_test_auc:
            best_test_auc = test_auc

            torch.save({
                'epoch': ep, 
                'model_state': model.state_dict(), 
                'test_auc': test_auc, 
                'test_aupr': test_aupr,
                'test_f1': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_acc': test_acc
            }, best_path)

    return best_path


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    drug_map   = load_indices_txt(PATH_DRUG_INDEX) if os.path.exists(PATH_DRUG_INDEX) else {}
    virus_map  = load_indices_txt(PATH_VIRUS_INDEX) if os.path.exists(PATH_VIRUS_INDEX) else {}
    global_pos = load_pairs_txt(PATH_GLOBAL_POS) if os.path.exists(PATH_GLOBAL_POS) else np.empty((0,2), dtype=np.int64)

    folds, max_d, max_v = [], -1, -1
    for i in range(5):
        tr = load_pairs_txt(FOLD_TRAIN_FILES[i]); te = load_pairs_txt(FOLD_TEST_FILES[i])
        folds.append((tr, te))
        if len(tr) > 0: max_d = max(max_d, int(tr[:,0].max())); max_v = max(max_v, int(tr[:,1].max()))
        if len(te) > 0: max_d = max(max_d, int(te[:,0].max())); max_v = max(max_v, int(te[:,1].max()))

    if len(global_pos) > 0: max_d = max(max_d, int(global_pos[:,0].max())); max_v = max(max_v, int(global_pos[:,1].max()))

    num_drugs   = max(max(drug_map.values()) + 1 if len(drug_map) > 0 else 0, max_d + 1)
    num_viruses = max(max(virus_map.values()) + 1 if len(virus_map) > 0 else 0, max_v + 1)

    print(f"#Drugs={num_drugs}, #Viruses={num_viruses}, #GlobalPos={len(global_pos)}")


    fold_aucs, fold_auprs, fold_f1s, fold_precisions, fold_recalls, fold_accs, ckpts = [], [], [], [], [], [], []
    
    for fold_id, (train_pos, test_pos) in enumerate(folds, start=0):
        print(f"\n===== Fold {fold_id} =====  train={len(train_pos)}  test={len(test_pos)}")
        train_adj = build_bipartite_adj(num_drugs, num_viruses, train_pos)
        H_d_sp, H_v_sp = build_incidence_from_bipartite(train_adj, drop_size1=False)

        # 注意：这里我们传递未经 row-stochasticize 的原始二部图给模型
        adj_tensor = to_torch_sparse_from_scipy(train_adj.tocoo(), device)
        H_d_tensor = to_torch_sparse_from_scipy(H_d_sp.tocoo(), device)
        H_v_tensor = to_torch_sparse_from_scipy(H_v_sp.tocoo(), device)

        model = CoBiHADR(num_drugs, num_viruses,
                           adj_dv=adj_tensor,
                           H_d=H_d_tensor, H_v=H_v_tensor,
                           latdim=LATDIM,
                           gat_dropout=GAT_DROPOUT, gat_alpha=GAT_ALPHA,
                           temperature=TEMPERATURE,
                           edge_drop_rate=EDGE_DROP_RATE,
                           n_layers = LAYERS,
                           device=device).to(device)

        best_path = train_one_fold(model, train_adj, train_pos, test_pos,
                                   global_pos_pairs=global_pos, device=device, fold_id=fold_id)

        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        global_pos_set = build_pos_set(global_pos)
        test_neg_fixed = sample_fixed_negatives(model.D, model.V, global_pos_set, len(test_pos) * TEST_NEG_PER_POS, seed=SEED + 1000 * fold_id)
        

        test_auc, test_aupr, test_f1, test_precision, test_recall, test_acc = eval_on_pairs(
            model, test_pos, model.D, model.V, 
            fixed_neg_pairs=test_neg_fixed, device=device
        )
        

        fold_aucs.append(test_auc)
        fold_auprs.append(test_aupr)
        fold_f1s.append(test_f1)
        fold_precisions.append(test_precision)
        fold_recalls.append(test_recall)
        fold_accs.append(test_acc)
        ckpts.append(best_path)
        

        print(f"[Fold {fold_id}] BEST Test AUC={test_auc:.4f}  AUPR={test_aupr:.4f}  F1={test_f1:.4f}  -> {best_path}")

    print("\n===== 5-Fold Summary =====")

    for i, (a, p, f1, c) in enumerate(zip(fold_aucs, fold_auprs, fold_f1s, ckpts), start=0):
        print(f"Fold {i}: AUC={a:.4f}  AUPR={p:.4f}  F1={f1:.4f}  ckpt={c}")
        

    print(f"Mean AUC={np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"Mean AUPR={np.mean(fold_auprs):.4f} ± {np.std(fold_auprs):.4f}")
    print(f"Mean F1={np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"Mean Precision={np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
    print(f"Mean Recall={np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
    print(f"Mean Accuracy={np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")


if __name__ == '__main__':
    main()
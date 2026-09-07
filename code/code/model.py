import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_edge_drop(adj: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0.0 or p >= 1.0:
        return adj
    nnz = adj._nnz()
    keep_prob = 1.0 - p
    num_to_keep = int(nnz * keep_prob)
    perm = torch.randperm(nnz, device=adj.device)
    keep_indices = perm[:num_to_keep]
    indices = adj._indices()[:, keep_indices]
    values = adj._values()[keep_indices]
    return torch.sparse_coo_tensor(indices, values, adj.shape).coalesce()


class BipartiteGraphAttentionLayer(nn.Module):
    def __init__(self, in_src: int, in_tgt: int, out: int, dropout: float, alpha: float, concat: bool = True):
        super().__init__()
        self.W_src = nn.Linear(in_src, out, bias=False)
        self.W_tgt = nn.Linear(in_tgt, out, bias=False)
        self.a = nn.Parameter(torch.empty(2 * out, 1))
        nn.init.xavier_uniform_(self.a, gain=1.414)
        self.leaky = nn.LeakyReLU(alpha)
        self.dropout = dropout
        self.concat = concat
        self.out = out

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h_i = self.W_src(src)
        h_j = self.W_tgt(tgt)
        s_id, t_id = adj._indices()
        e = self.leaky(torch.matmul(torch.cat([h_i[s_id], h_j[t_id]], dim=1), self.a)).squeeze(-1)
        e = e.clamp(min=-30., max=30.)
        exp_e = torch.exp(e)
        denom = torch.zeros(src.size(0), device=src.device)
        denom.scatter_add_(0, s_id, exp_e)
        alpha = exp_e / (denom[s_id] + 1e-8)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.zeros_like(h_i)
        out.scatter_add_(0, s_id.unsqueeze(-1).expand(-1, self.out), h_j[t_id] * alpha.unsqueeze(-1))
        return F.elu(out) if self.concat else out

class HypergraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        X = self.lin(X)
        He = torch.sparse.sum(H, dim=0).to_dense().clamp(min=1.0)
        Hv = torch.sparse.sum(H, dim=1).to_dense().clamp(min=1.0)
        Dv_inv_sqrt = (1.0 / Hv).sqrt().unsqueeze(-1)
        De_inv = 1.0 / He
        X1 = X * Dv_inv_sqrt
        H_T = torch.transpose(H, 0, 1).coalesce()
        tmp = torch.sparse.mm(H_T, X1)
        tmp = tmp * De_inv.unsqueeze(-1)
        out = torch.sparse.mm(H, tmp)
        out = out * Dv_inv_sqrt
        return F.elu(out)


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, T: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = (z1 @ z2.T) / T
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)

class DegreeHead(nn.Module):
    def __init__(self, dim: int, hidden: int = None):
        super().__init__()
        h = hidden if hidden is not None else max(16, dim // 2)
        self.fc1 = nn.Linear(dim, h)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(h, 1)
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x).squeeze(-1)
        return x


class HyCoVDA(nn.Module):
    def __init__(self,
                 num_drugs: int,
                 num_viruses: int,
                 adj_dv: torch.Tensor,
                 H_d: torch.Tensor,
                 H_v: torch.Tensor,
                 latdim: int = 128,
                 gat_dropout: float = 0.1,
                 gat_alpha: float = 0.2,
                 temperature: float = 0.05,
                 edge_drop_rate: float = 0,
                 n_layers: int = 1,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        self.D = num_drugs
        self.V = num_viruses
        self.lat = latdim
        self.tau = temperature
        self.edge_drop_rate = edge_drop_rate
        self.device = device

        self.drug_embed = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.D, self.lat, device=device)))
        self.virus_embed = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.V, self.lat, device=device)))

        self.adj = adj_dv.coalesce()
        self.adj_T = torch.transpose(self.adj, 0, 1).coalesce()
        self.H_d = H_d.coalesce()
        self.H_v = H_v.coalesce()

        self.gat_d_layers = nn.ModuleList()
        self.gat_v_layers = nn.ModuleList()
        self.hgc_d_layers = nn.ModuleList()
        self.hgc_v_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.gat_d_layers.append(
                BipartiteGraphAttentionLayer(latdim, latdim, latdim, gat_dropout, gat_alpha)
            )
            self.gat_v_layers.append(
                BipartiteGraphAttentionLayer(latdim, latdim, latdim, gat_dropout, gat_alpha)
            )
            self.hgc_d_layers.append(HypergraphConv(latdim, latdim))
            self.hgc_v_layers.append(HypergraphConv(latdim, latdim))


        with torch.no_grad():
            deg_d = torch.sparse.sum(self.adj, dim=1).to_dense()
            deg_v = torch.sparse.sum(self.adj, dim=0).to_dense()
            deg_d_n = torch.log1p(deg_d)
            deg_v_n = torch.log1p(deg_v)
            md, mv = deg_d_n.max().clamp(min=1.0), deg_v_n.max().clamp(min=1.0)
            deg_d_n = (deg_d_n / md).to(device)
            deg_v_n = (deg_v_n / mv).to(device)
        self.register_buffer("deg_d_target", deg_d_n)
        self.register_buffer("deg_v_target", deg_v_n)


        self.deg_head_d = DegreeHead(latdim)
        self.deg_head_v = DegreeHead(latdim)
        with torch.no_grad():
            mean_d = self.deg_d_target.mean().clamp(1e-6, 1-1e-6)
            mean_v = self.deg_v_target.mean().clamp(1e-6, 1-1e-6)
            def inv_sigmoid(p): return torch.log(p/(1-p))
            self.deg_head_d.fc2.bias.fill_(inv_sigmoid(mean_d))
            self.deg_head_v.fc2.bias.fill_(inv_sigmoid(mean_v))


        self.mlp = nn.Sequential(nn.Linear(latdim * 2, latdim), nn.ReLU(), nn.Linear(latdim, 1))
        self.gate = nn.Sequential(nn.Linear(latdim, 1), nn.Sigmoid())
        

    def forward(self, drug_ids: torch.Tensor, virus_ids: torch.Tensor):
        d0 = self.drug_embed
        v0 = self.virus_embed
        
        if self.training and self.edge_drop_rate > 0:
            adj_aug = sparse_edge_drop(self.adj, self.edge_drop_rate)
            adj_aug_T = torch.sparse_coo_tensor(
                adj_aug.indices().flip([0]), 
                adj_aug.values(), 
                self.adj_T.shape
            ).coalesce()
        else:
            adj_aug, adj_aug_T = self.adj, self.adj_T

        d, v = d0, v0
        for i in range(len(self.gat_d_layers)):
            d = self.gat_d_layers[i](d, v, adj_aug)
            v = self.gat_v_layers[i](v, d, adj_aug_T)
        sg_d, sg_v = d, v

        hd, hv = d0, v0
        for i in range(len(self.hgc_d_layers)):
            hd = self.hgc_d_layers[i](hd, self.H_d)
            hv = self.hgc_v_layers[i](hv, self.H_v)

        final_d = sg_d + hd 
        final_v = sg_v + hv 

        sel_d = final_d.index_select(0, drug_ids)
        sel_v = final_v.index_select(0, virus_ids)

        # contrastive loss
        contrast_loss = info_nce_loss(sg_d, hd, T=self.tau) + info_nce_loss(sg_v, hv, T=self.tau)

        # degree prediction
        pred_deg_d = torch.sigmoid(self.deg_head_d(sel_d))
        pred_deg_v = torch.sigmoid(self.deg_head_v(sel_v))

        # main prediction
        gmf_vec = sel_d * sel_v
        gmf_pred = gmf_vec.sum(dim=-1)
        mlp_pred = self.mlp(torch.cat([sel_d, sel_v], dim=-1)).squeeze(-1)
        gate = self.gate(gmf_vec).squeeze(-1)
        pred = gate * gmf_pred + (1 - gate) * mlp_pred

        return pred, contrast_loss, pred_deg_d, pred_deg_v

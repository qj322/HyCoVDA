# Welcome to HyCoVDA: Self-Supervised Hypergraph Contrastive Learning for Virus–Drug Association
Viral infections remain a major threat to global public health. Although antiviral drugs can significantly reduce disease burden, traditional drug development approaches that depend on biological experiments are often time consuming, costly, and inefficient. Computational methods provide a viable alternative to accelerate the discovery of potential antiviral agents by prioritizing candidates for further validation. We hypothesize that integrating pairwise interaction information with higher-order drug–virus relationships can improve the prediction of potential virus–drug associations, particularly under sparse association conditions. To test this hypothesis, we propose HyCoVDA, a novel computational framework for predicting virus–drug associations. HyCoVDA learns node representations from two complementary structural views tailored to VDA challenges: a graph attention network (GAT) captures sparse direct interactions, while drug- and virus-specific hypergraphs represent group-level patterns derived from shared drug–virus associations. To address the sparsity of known associations, we integrate a self-supervised contrastive loss that aligns representations from the two views, encouraging cross-view consistency. In addition, we introduce a degree regression objective to retain node-degree information in the learned representations. Experimental results demonstrate that HyCoVDA achieves AUC and AUPR scores of 0.892 and 0.889, respectively, outperforming the evaluated benchmark methods. A case study focused on SARS-CoV-2 found literature reports of possible associations for 17 of the top 20 predictions. Overall, these results suggest that HyCoVDA may help prioritize antiviral drug candidates for further experimental validation. 

<img title="" src="./model.png" alt="Alternative text" width="800">

## 🔧 Installation instructions

1. **Clone the repository**
```bash
git clone https://github.com/qj322/HyCoVDA.git
cd HyCoVDA
```
2. **Set up the Python environment**
```bash
conda create -n hycovda python=3.10
conda activate hycovda
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install numpy scipy scikit-learn -y
```
## Model Training

Train the model from scratch:

```bash
python train.py
```
The training script will automatically save the model with the best validation to `best_model.pth`.

## Model Evaluation

Evaluate the trained model:

```bash
python eval.py
```
The script reports the following metrics:

* AUC
* AUPR
* Accuracy
* Precision
* Recall
* F1-score

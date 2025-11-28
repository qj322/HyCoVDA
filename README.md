# Welcome to MGCL‑CAP: Masked Graph Contrastive Learning with Gated Cross‑Attention for Chemical Allergenicity Prediction
Chemical allergens are common in consumer and industrial products and can trigger hypersensitivity with important public-health and regulatory implications. Traditional experimental screening is time-consuming and labor-intensive, slowing discovery and risk assessment. Existing computational approaches often rely on hand-crafted fingerprints and shallow classifiers, which do not adequately capture molecular topology or cross-modal dependencies, limiting generalization and interpretability. We propose MGCL-CAP, a deep learning framework for chemical allergenicity prediction that enhances molecular representation through masked graph contrastive learning and gated cross-attention fusion. MGCL-CAP performs random subgraph masking within a shared GIN encoder to learn structure-invariant graph embeddings that remain robust to missing or spurious substructures. These embeddings are then integrated with one-dimensional molecular fingerprints via multi-head gated cross-attention to align modalities and emphasize salient chemical cues. Experimental results show that MGCL-CAP outperforms state-of-the-art allergenicity predictors and remains stable across reasonable hyperparameter ranges. Interpretability analyses highlight substructures consistent with known sensitization mechanisms, providing mechanistic insight. Overall, MGCL-CAP offers a reliable tool for computational assessment of chemical allergenicity, enabling efficient candidate prioritization and supporting safer formulation design while reducing experimental burden.

![The workflow of this study](https://github.com/GGCL7/MGCL-CAP/blob/main/workflow.png)


## 🔧 Installation instructions

1. **Clone the repository**
```bash
git clone https://github.com/GGCL7/MGCL-CAP.git
cd MGCL-CAP
```
2. **Set up the Python environment**
```bash
conda create -n mgclcap python=3.10
conda activate mgclcap
pip install -r requirements.txt
```
## Model Training

Train the model from scratch:

```bash
python main.py
```
The training script will automatically save the model with the best validation **MCC** to `best_model.pth`.

## Model Evaluation

Evaluate the trained model:

```bash
python evaluation.py
```
The script reports the following metrics:

* Accuracy
* Sensitivity
* Specificity
* Matthews Correlation Coefficient (MCC)
* Area Under the Curve (AUC)


## 🛠️ Using MGCL-CAP for chemical allergenicity prediction

## Single molecule prediction

We provide a simple interface to predict allergenicity for a single SMILES string. For example, to predict the allergenicity of "C(CNC(NCCCC)=S)CC" with a pre-trained model:

```bash
python predict.py --smiles "C(CNC(NCCCC)=S)CC" --model-path best_model.pth
```

## Output example:

```bash
SMILES   : C(CNC(NCCCC)=S)CC
Prob(+)  : 0.873241
Label    : 1  (threshold=0.5)

```

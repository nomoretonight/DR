import os
import json
import torch
import random
import numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass

def save_checkpoint(state, is_best, work_dir, filename='checkpoint.pth'):
    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(work_dir, 'best_model.pth')
        torch.save(state, best_path)

def compute_metrics(y_true_ref, y_pred_ref_logits, y_true_multi, y_pred_multi_logits):
    # binary referable AUC
    metrics = {}
    try:
        y_prob_ref = torch.sigmoid(torch.tensor(y_pred_ref_logits)).numpy()
    except:
        import numpy as np
        y_prob_ref = 1/(1+np.exp(-np.array(y_pred_ref_logits)))
    if len(np.unique(y_true_ref)) > 1:
        metrics['referable_auc'] = roc_auc_score(y_true_ref, y_prob_ref)
    else:
        metrics['referable_auc'] = float('nan')
    # multiclass predicted labels
    import numpy as np
    probs_multi = torch.softmax(torch.tensor(y_pred_multi_logits), dim=1).numpy()
    y_pred_multi = np.argmax(probs_multi, axis=1)
    try:
        metrics['kappa'] = cohen_kappa_score(y_true_multi, y_pred_multi, weights='quadratic')
    except:
        metrics['kappa'] = float('nan')
    # sensitivity/specificity for referable at threshold 0.5
    y_pred_ref = (y_prob_ref >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_ref, y_pred_ref, labels=[0,1]).ravel()
    metrics['sensitivity'] = tp / (tp + fn + 1e-8)
    metrics['specificity'] = tn / (tn + fp + 1e-8)
    return metrics

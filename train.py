import os
import argparse
import yaml
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import GroupKFold, train_test_split
from tqdm import tqdm

from dataset import FundusDataset
from model import DRMultiHeadModel
from utils import set_seed, save_checkpoint, compute_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to config.yaml')
    return parser.parse_args()

def prepare_loaders(cfg):
    df = pd.read_csv(cfg['data']['csv_path'])
    # Filter unknowns & drop missing paths
    df = df.dropna(subset=['image_path']).reset_index(drop=True)
    # patient-level split
    if 'patient_id' not in df.columns:
        df['patient_id'] = df.index
    train_ids, val_ids = train_test_split(df['patient_id'].unique(), test_size=0.2, random_state=cfg['training']['seed'])
    train_df = df[df['patient_id'].isin(train_ids)].reset_index(drop=True)
    val_df = df[df['patient_id'].isin(val_ids)].reset_index(drop=True)
    train_ds = FundusDataset(train_df, cfg['data']['image_root'], img_size=cfg['data']['img_size'], mode='train', filter_gradable=True)
    val_ds = FundusDataset(val_df, cfg['data']['image_root'], img_size=cfg['data']['img_size'], mode='val', filter_gradable=True)
    train_loader = DataLoader(train_ds, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'], pin_memory=True)
    return train_loader, val_loader

def validate(model, loader, device):
    model.eval()
    all_ref_logits = []
    all_mult_logits = []
    all_ref_true = []
    all_multi_true = []
    with torch.no_grad():
        for imgs, labels, referable, _ in tqdm(loader, desc='Val'):
            imgs = imgs.to(device)
            out_multi, out_ref = model(imgs)
            all_mult_logits.append(out_multi.detach().cpu())
            all_ref_logits.append(out_ref.detach().cpu())
            all_ref_true.extend(referable.numpy().tolist())
            all_multi_true.extend(labels.numpy().tolist())
    import torch
    all_ref_logits = torch.cat(all_ref_logits, dim=0)
    all_mult_logits = torch.cat(all_mult_logits, dim=0)
    metrics = compute_metrics(all_ref_true, all_ref_logits, all_multi_true, all_mult_logits)
    return metrics

def train(cfg):
    set_seed(cfg['training']['seed'])
    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = prepare_loaders(cfg)
    model = DRMultiHeadModel(backbone_name=cfg['training']['backbone'], pretrained=cfg['training']['pretrained'])
    model = model.to(device)
    # losses
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    # optimizer
    if cfg['training']['optimizer'].lower() == 'adamw':
        optimizer = AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    else:
        optimizer = SGD(model.parameters(), lr=cfg['training']['lr'], momentum=0.9, weight_decay=cfg['training']['weight_decay'])
    # scheduler
    if cfg['training']['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'])
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['training'].get('step_size',10), gamma=0.1)

    scaler = GradScaler(enabled=cfg['training'].get('amp', True))
    best_metric = -999
    no_better_epochs = 0
    os.makedirs(cfg['output']['work_dir'], exist_ok=True)
    for epoch in range(1, cfg['training']['epochs'] + 1):
        model.train()
        loop = tqdm(train_loader, desc=f'Train Epoch {epoch}')
        running_loss = 0.0
        for imgs, labels, referable, _ in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)
            referable = referable.float().to(device)
            optimizer.zero_grad()
            with autocast(enabled=cfg['training'].get('amp', True)):
                out_multi, out_ref = model(imgs)
                loss_multi = ce_loss(out_multi, labels)
                loss_ref = bce_loss(out_ref, referable)
                loss = loss_multi + 0.5 * loss_ref
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            loop.set_postfix(loss=running_loss/ (loop.n + 1))
        scheduler.step()
        # validate
        metrics = validate(model, val_loader, device)
        metric_key = cfg['training'].get('objective', 'referable_auc')
        current_metric = metrics.get(metric_key, metrics.get('referable_auc', 0.0))
        print(f"Epoch {epoch} metrics: {metrics}")
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric
            no_better_epochs = 0
        else:
            no_better_epochs += 1
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'cfg': cfg,
            'metrics': metrics,
        }, is_best, cfg['output']['work_dir'], filename=f'epoch_{epoch}.pth')
        if no_better_epochs >= cfg['output'].get('early_stopping_patience', 8):
            print("Early stopping triggered.")
            break

if __name__ == '__main__':
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    train(cfg)

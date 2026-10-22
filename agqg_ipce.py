import os
import random
import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import tqdm
import pandas as pd
import scipy.stats

# Project-specific module imports
import clip_atten
from clip_atten import clip
from MNL_Loss import loss_m3
import loss
from utils import (
    set_dataset_aigc_1k, 
    set_dataset_aigc_3k, 
    set_dataset_aigc_2023,
    _preprocess2, 
    _preprocess3, 
    convert_models_to_fp32, 
    get_logger, 
    log_and_print
)
# Assuming the main model is placed in modules/ipce.py
from modules.ipce import AGQG_IPCE 

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for AGQG + IPCE")
    
    # Paths and saving configurations
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/ipce_agqg', help='Directory to save checkpoints and logs')
    parser.add_argument('--csv_name', type=str, default='ipce_agqg_results.csv', help='Name of the CSV file to save predictions')
    parser.add_argument('--data_root', type=str, default='./Database', help='Root directory for datasets')
    
    # Dataset and task configurations
    parser.add_argument('--dataset', type=str, default='AGIQA-3K', choices=['AGIQA-1K', 'AGIQA-3K', 'AIGCIQA2023'], help='Dataset to use')
    parser.add_argument('--score_type', type=str, default='quality', choices=['quality', 'consistency', 'align', 'authenticity'], help='Assessment dimension')
    parser.add_argument('--session', type=int, default=0, help='Training session/index for cross-validation splits')
    
    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--train_patch', type=int, default=8, help='Number of training patches per image')
    parser.add_argument('--test_patch', type=int, default=15, help='Number of test patches per image')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    
    # Model configurations
    parser.add_argument('--D_grm', type=float, default=2.0, help='Scaling constant D in GRM')
    
    return parser.parse_args()

def get_unified_prompt(prompts, score_type):
    """
    Return the unified prompt based on the assessment dimension 
    (corresponding to the Optional Prompt Module in the paper).
    """
    unified_prompts = []
    for p in prompts:
        if score_type in ['quality', 'mos']:
            unified_prompts.append('A photo of good quality and clear details')
        elif score_type in ['consistency', 'align']:
            # For alignment/consistency assessment, use the original generation prompt by default
            unified_prompts.append(p) 
        elif score_type == 'authenticity':
            unified_prompts.append('A photo with genuine scene content and no synthetic artifacts.')
        else:
            raise ValueError(f"Unsupported score_type: {score_type}")
    return unified_prompts

def get_dataloaders(args):
    """Dynamically load the corresponding dataset based on arguments."""
    preprocess_train = _preprocess3()
    preprocess_val = _preprocess2()
    
    dataset_name = args.dataset
    split_idx = str(args.session)
    
    if dataset_name == 'AGIQA-1K':
        train_csv = os.path.join(args.data_root, 'AGIQA-1K', split_idx, 'train.csv')
        val_csv = os.path.join(args.data_root, 'AGIQA-1K', split_idx, 'val.csv')
        img_root = os.path.join(args.data_root, 'AGIQA-1K', 'images')
        loader_fn = set_dataset_aigc_1k
    elif dataset_name == 'AGIQA-3K':
        train_csv = os.path.join(args.data_root, 'AGIQA-3K', split_idx, 'train.csv')
        val_csv = os.path.join(args.data_root, 'AGIQA-3K', split_idx, 'val.csv')
        img_root = os.path.join(args.data_root, 'AGIQA-3K', 'images')
        loader_fn = set_dataset_aigc_3k
    elif dataset_name == 'AIGCIQA2023':
        train_csv = os.path.join(args.data_root, 'AIGCIQA2023', split_idx, 'train.csv')
        val_csv = os.path.join(args.data_root, 'AIGCIQA2023', split_idx, 'val.csv')
        img_root = os.path.join(args.data_root, 'AIGCIQA2023', 'images')
        loader_fn = set_dataset_aigc_2023
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    train_loader = loader_fn(train_csv, args.batch_size, img_root, args.num_workers, preprocess_train, args.train_patch, is_train=True)
    val_loader = loader_fn(val_csv, args.batch_size, img_root, args.num_workers, preprocess_val, args.test_patch, is_train=False)
    
    return train_loader, val_loader

def save_predictions(filepath, image_names, gmos, preds, logits, abilities, betas):
    """Save predictions and intermediate variables to a CSV file."""
    df = pd.DataFrame({
        'Image_Name': image_names,
        'GMOS': gmos,
        'Prediction': preds,
        'Logit_1': [l[0] for l in logits],
        'Logit_2': [l[1] for l in logits],
        'Logit_3': [l[2] for l in logits],
        'Logit_4': [l[3] for l in logits],
        'Logit_5': [l[4] for l in logits],
        'Ability': abilities,
        'Beta_1': [b[0] for b in betas],
        'Beta_2': [b[1] for b in betas],
        'Beta_3': [b[2] for b in betas],
        'Beta_4': [b[3] for b in betas],
    })
    df.to_csv(filepath, index=False)

def train_epoch(model, loader, optimizer, scheduler, epoch, device, args, logger):
    model.train()
    running_loss = 0.0
    
    # Loss functions
    mse_criterion = loss_m3
    plcc_criterion = loss.plccloss()
    
    loop = tqdm.tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for step, sample in enumerate(loop):
        images = sample['I'].to(device)
        prompts = sample['prompt']
        gmos = sample[args.score_type].to(device)
        image_names = sample['image_name']
        
        # Get unified prompts and tokenize them
        unified_prompts = get_unified_prompt(prompts, args.score_type)
        text_tokens = torch.cat([clip.tokenize(c, truncate=True) for c in unified_prompts]).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (using raw text input to support dynamic prompts)
        logits_quality, ability, beta = model(images, unified_prompts, require_beta=True)
        
        # Compute the weighted quality score (1-5)
        weights = torch.arange(1, 6, device=device, dtype=logits_quality.dtype)
        quality_preds = torch.matmul(logits_quality, weights)
        
        # Scale the predictions to match the MOS range of specific datasets (e.g., mapping 1-5 to 0-5)
        # Note: Comment out this line if the dataset MOS is already in the 1-5 range
        quality_preds = ((quality_preds - 1) / 4) * 5 
        
        # Compute losses
        mse_loss = mse_criterion(quality_preds, gmos.detach()).mean()
        plcc_loss = plcc_criterion(quality_preds.unsqueeze(1).float(), gmos.detach().unsqueeze(1).float())
        
        total_loss = mse_loss + plcc_loss
        total_loss.backward()
        
        # Optimizer step (handling FP16/FP32 conversion for the CLIP model)
        if device.type == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model.get_clip())
            optimizer.step()
            clip_atten.model.convert_weights(model.get_clip())
            
        running_loss += total_loss.item()
        loop.set_postfix(loss=running_loss / (step + 1))
        
    scheduler.step()
    avg_loss = running_loss / len(loader)
    log_and_print(logger, f'Epoch {epoch} Train Loss: {avg_loss:.4f}')

def evaluate(model, loader, epoch, device, args, logger, save_csv=False, csv_path=None):
    model.eval()
    
    q_mos, q_hat = [], []
    batch_logits, batch_ability, batch_beta, image_names = [], [], [], []
    
    loop = tqdm.tqdm(loader, desc=f'Epoch {epoch} [Eval]')
    with torch.no_grad():
        for sample in loop:
            images = sample['I'].to(device)
            prompts = sample['prompt']
            gmos = sample[args.score_type]
            image_names.extend(sample['image_name'])
            
            unified_prompts = get_unified_prompt(prompts, args.score_type)
            
            logits_quality, ability, beta = model(images, unified_prompts, require_beta=True)
            
            weights = torch.arange(1, 6, device=device, dtype=logits_quality.dtype)
            quality_preds = torch.matmul(logits_quality, weights)
            quality_preds = ((quality_preds - 1) / 4) * 5
            
            q_mos.extend(gmos.cpu().tolist())
            q_hat.extend(quality_preds.cpu().tolist())
            batch_logits.extend(logits_quality.cpu().tolist())
            batch_ability.extend(ability.cpu().tolist())
            batch_beta.extend(beta.cpu().tolist())
            
    srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
    plcc = scipy.stats.pearsonr(x=q_mos, y=q_hat)[0]
    avg_score = (srcc + plcc) / 2
    
    log_and_print(logger, f'Epoch {epoch} Eval -> SRCC: {srcc:.4f} | PLCC: {plcc:.4f} | Avg: {avg_score:.4f}')
    
    if save_csv and csv_path:
        save_predictions(csv_path, image_names, q_mos, q_hat, batch_logits, batch_ability, batch_beta)
        
    return avg_score, srcc, plcc

def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Device and directories
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize logger
    logger = get_logger(os.path.join(args.checkpoint_dir, 'train.log'), 'log')
    log_and_print(logger, f"Arguments: {vars(args)}")
    log_and_print(logger, f"Using device: {device}")
    
    # Load data
    train_loader, val_loader = get_dataloaders(args)
    
    # Initialize model
    # Note: Please ensure the AGQG_IPCE class exists in modules.ipce
    model = AGQG_IPCE(device=device, D_grm=args.D_grm)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    best_result = {'avg': 0.0, 'srcc': 0.0, 'plcc': 0.0}
    best_epoch = 0
    
    # Training loop
    for epoch in range(args.epochs):
        train_epoch(model, train_loader, optimizer, scheduler, epoch, device, args, logger)
        
        # Evaluation
        avg_score, srcc, plcc = evaluate(
            model, val_loader, epoch, device, args, logger, 
            save_csv=(epoch == args.epochs - 1), # Save detailed CSV only at the last epoch
            csv_path=os.path.join(args.checkpoint_dir, args.csv_name)
        )
        
        # Save the best model
        if avg_score > best_result['avg']:
            best_result = {'avg': avg_score, 'srcc': srcc, 'plcc': plcc}
            best_epoch = epoch
            ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, ckpt_path)
            log_and_print(logger, f"*** New Best Model saved at Epoch {epoch} ***")
            
    log_and_print(logger, f"Training Finished. Best Epoch: {best_epoch} | SRCC: {best_result['srcc']:.4f} | PLCC: {best_result['plcc']:.4f}")

if __name__ == '__main__':
    main()
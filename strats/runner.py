import pandas as pd
import numpy as np
import datetime
import os
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve, brier_score_loss


def pc2012_pretrain_model(model, train_loader, valid_loader, epochs, optimizer, device,
                    patience=7, model_save_path='best_pretrained_dmask_model.pt'):
    """
    Self-supervised learning function of model
    
    Parameters
    ----------
    model : nn.Module
        STraTS model to train
    train_loader : DataLoader 
        (hadm_id, times, varis, values, saps, sofa, death, mask_t, padding_mask) for training data
    valid_loader : DataLoader
        (hadm_id, times, varis, values, saps, sofa, death, mask_t, padding_mask) for validation data   
    epochs : int
        Number of epochs to pretrain the model
    optimizer : PyTorch optimizer
        Optimizer for learning rate scheduling
    device : torch.device
        CUDA or CPU, device to run model
    patience : int
        Number of epochs to train with no improvements in validation loss
    model_save_path : str
        Path to save the model, should be '~.pt'     
    
    Returns
    -------
    None, but prints epoch-wise loss and loss plot
    """
    writer = SummaryWriter()
    model.to(device)
    best_valid_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"[Pretrain] Epoch [{epoch+1}/{epochs}] - Training", leave=False):
            # 예상되는 batch 구조: (times, varis, values, saps, sofa, death, padding_mask)
            # 하지만 pretrain에서는 saps/sofa/death가 필요하지 않으므로 '_'로 받거나 무시.
            hadm_id, times, varis, values, _, _, _, mask_t, padding_mask = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            outputs = model(times, varis, values, 
                            padding_mask=padding_mask, 
                            pretrain=True,
                            pretrain_mask = mask_t)  # <-- pretrain=True
            loss = outputs['loss']

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"[Pretrain] Epoch [{epoch+1}/{epochs}] - Validation", leave=False):
                hadm_id, times, varis, values, _, _, _, mask_t, padding_mask = [x.to(device) for x in batch]
                outputs = model(times, varis, values, 
                                padding_mask=padding_mask, 
                                pretrain=True,
                                pretrain_mask = mask_t)
                valid_loss += outputs['loss'].item()
        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)

        print(f"[Pretrain] Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Valid Loss = {valid_loss:.4f}")

        # Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered (pretrain).")
            break

    writer.close()
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, valid_losses, label='Valid Loss')
    plt.title('Pretraining Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def pc2012_train_model(model : nn.Module, train_loader : DataLoader, valid_loader : DataLoader, epochs : int, 
                optimizer : torch.optim.Optimizer, device : torch.device,
                 patience=7, freeze=False, model_save_path='best_downstream_dmask_model.pt', use_loss=['death']):
    """
    Supervised learning function of model
    
    Parameters
    ----------
    model : nn.Module
        STraTS model to train
    train_loader : DataLoader 
        (hadm_id, times, varis, values, saps, sofa, death, mask_t, padding_mask) for training data
    valid_loader : DataLoader
        (hadm_id, times, varis, values, saps, sofa, death, mask_t, padding_mask) for validation data   
    epochs : int
        Number of epochs to train the model
    optimizer : PyTorch optimizer
        Optimizer for learning rate scheduling
    device : torch.device
        CUDA or CPU, device to run model
    patience : int
        Number of epochs to train with no improvements in validation loss
    freeze : bool
        Whether to update the embedding part of the model or not
    model_save_path : str
        Path to save the model, should be '~.pt' 
    use_loss : list(str)
        Which loss to use, in form ['death', 'saps']
    
    
    Returns
    -------
    loss_df : pd.DataFrame
        Contains ['Train loss', 'Train SAPS', 'Train SOFA', 'Train DEATH',
               'Valid loss', 'Valid SAPS', 'Valid SOFA', 'Valid DEATH',
               'Valid AUROC', 'Valid AUPRC']
    """
    if 'death' not in use_loss and 'saps' not in use_loss and 'sofa' not in use_loss:
        print(f'Invalid input for "use_loss". Using default "death" loss')
        use_loss = ['death']
    writer = SummaryWriter()
    model.to(device)
    best_valid_loss = float('inf')
    early_stop_counter = 0

    train_losses = []
    train_saps_losses = []
    train_sofa_losses = []
    train_death_losses = []

    valid_losses = []
    valid_saps_losses = []
    valid_sofa_losses = []
    valid_death_losses = []

    # AUROC / AUPRC 기록용
    valid_aurocs = []
    valid_auprcs = []

    for epoch in range(epochs):
        # ===================================================
        # (1) Training phase
        # ===================================================
        model.train()
        train_loss = 0
        train_saps_loss = 0
        train_sofa_loss = 0
        train_death_loss = 0

        for batch in tqdm(train_loader, desc=f"[Downstream] Epoch [{epoch+1}/{epochs}] - Training", leave=False):
            hadm_id, times, varis, values, saps, sofa, death, _, padding_mask = [x.to(device) for x in batch]
            
            # outcomes: shape (batch_size, 4) => [hadm_id, saps_label, sofa_label, death_label]
            outcomes = torch.stack([hadm_id, saps, sofa, death], dim=1)

            optimizer.zero_grad()
            # pretrain=False, 전달된 outcomes로 loss 계산
            outputs = model(times, varis, values, 
                            padding_mask=padding_mask, 
                            pretrain=False,
                            freeze_pretrained = freeze, 
                            outcomes=outcomes)

            loss = 0
            
            loss_saps  = outputs['loss_saps']
            loss_sofa  = outputs['loss_sofa']
            loss_death = outputs['loss_death']

            if 'death' in use_loss:
                loss += loss_death
            if 'sofa' in use_loss:
                loss += loss_sofa
            if 'saps' in use_loss:
                loss += loss_saps

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_saps_loss += loss_saps.item()
            train_sofa_loss += loss_sofa.item()
            train_death_loss += loss_death.item()

        # epoch 단위 평균
        train_loss       /= len(train_loader)
        train_saps_loss  /= len(train_loader)
        train_sofa_loss  /= len(train_loader)
        train_death_loss /= len(train_loader)

        train_losses.append(train_loss)
        train_saps_losses.append(train_saps_loss)
        train_sofa_losses.append(train_sofa_loss)
        train_death_losses.append(train_death_loss)

        # ===================================================
        # (2) Validation phase
        # ===================================================
        valid_loss = 0
        valid_saps_loss = 0
        valid_sofa_loss = 0
        valid_death_loss = 0

        # -----------------------------
        # (a) AUROC/AUPRC 계산 위해 전체 배치 예측/라벨 누적
        # -----------------------------
        death_preds_all = []
        death_labels_all = []

        model.eval()
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"[Downstream] Epoch [{epoch+1}/{epochs}] - Validation", leave=False):
                hadm_id, times, varis, values, saps, sofa, death, _, padding_mask = [x.to(device) for x in batch]
                outcomes = torch.stack([hadm_id, saps, sofa, death], dim=1)

                outputs = model(times, varis, values, 
                                padding_mask=padding_mask, 
                                pretrain=False,
                                freeze_pretrained=freeze, 
                                outcomes=outcomes)

                loss       = 0
                loss_saps  = outputs['loss_saps']
                loss_sofa  = outputs['loss_sofa']
                loss_death = outputs['loss_death']

                if 'death' in use_loss:
                    loss += loss_death
                if 'sofa' in use_loss:
                    loss += loss_sofa
                if 'saps' in use_loss:
                    loss += loss_saps

                valid_loss       += loss.item()
                valid_saps_loss  += loss_saps.item()
                valid_sofa_loss  += loss_sofa.item()
                valid_death_loss += loss_death.item()

                # -----------------------
                # (b) death pred & label
                # -----------------------
                # pred_death: (batch_size, 1) => sigmoid output
                # death_label: (batch_size,)
                pred_death  = outputs['pred_death'].squeeze(-1)  # shape: [bsz]
                death_label = outputs['death']  # shape: [bsz]

                # cpu로 옮겨 numpy 변환
                pred_death_np  = pred_death.detach().cpu().numpy()
                death_label_np = death_label.detach().cpu().numpy()

                death_preds_all.append(pred_death_np)
                death_labels_all.append(death_label_np)

        # (c) epoch 단위 평균 Loss
        valid_loss       /= len(valid_loader)
        valid_saps_loss  /= len(valid_loader)
        valid_sofa_loss  /= len(valid_loader)
        valid_death_loss /= len(valid_loader)

        valid_losses.append(valid_loss)
        valid_saps_losses.append(valid_saps_loss)
        valid_sofa_losses.append(valid_sofa_loss)
        valid_death_losses.append(valid_death_loss)

        # -----------------------
        # (d) AUROC / AUPRC 계산
        # -----------------------
        # stack 전체 death 예측/라벨 => shape (N,)
        death_preds_all = np.concatenate(death_preds_all, axis=0)
        death_labels_all = np.concatenate(death_labels_all, axis=0)

        auroc = roc_auc_score(death_labels_all, death_preds_all)
        auprc = average_precision_score(death_labels_all, death_preds_all)

        valid_aurocs.append(auroc)
        valid_auprcs.append(auprc)

        print(f"[Epoch {epoch+1}] Train Loss={train_loss:.4f}, Valid Loss={valid_loss:.4f}, "
              f"Valid AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

        # (e) Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered (downstream).")
            break

    writer.close()

    # (3) 에폭별 결과 시각화/데이터프레임 생성
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, valid_losses, label='Valid Loss')
    plt.title('Downstream Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 추가: AUROC, AUPRC 시각화(옵션)
    plt.figure()
    plt.plot(epochs_range, valid_aurocs, label='Valid AUROC')
    plt.plot(epochs_range, valid_auprcs, label='Valid AUPRC')
    plt.title('Death Prediction Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.show()

    # (4) DataFrame
    # zip(*...)로 행=epoch, 열=각 metric
    data = list(zip(train_losses, train_saps_losses, train_sofa_losses, train_death_losses,
                    valid_losses, valid_saps_losses, valid_sofa_losses, valid_death_losses,
                    valid_aurocs, valid_auprcs))
    columns = ['Train loss', 'Train SAPS', 'Train SOFA', 'Train DEATH',
               'Valid loss', 'Valid SAPS', 'Valid SOFA', 'Valid DEATH',
               'Valid AUROC', 'Valid AUPRC']
    loss_df = pd.DataFrame(data=data, columns=columns)

    return loss_df

def pc2012_evaluate_model(model, test_loader, device):
    """
    Test set에 대해 모델 추론 후, 다양한 분류 지표를 계산하는 함수.
    
    모델은 pred_death(= 사망 확률, 0~1)를 반환한다고 가정.
    실제 라벨(0/1)은 death로 가정.
    
    Parameters
    ----------
    model : nn.Module
        학습이 완료된 PyTorch 모델
    test_loader : DataLoader
        테스트 데이터셋 로더 (배치별로 hadm_id, times, varis, values, saps, sofa, death, padding_mask 등 반환)
    device : torch.device
        CUDA 또는 CPU
    
    Returns
    -------
    metrics : dict
        {
          'AUROC': float,
          'AUPRC': float,
          'Precision': float,
          'Recall': float,
          'F1': float,
          'Accuracy': float
        }
    """

    model.eval()  # 추론 모드
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # 배치 구조 예: hadm_id, times, varis, values, saps, sofa, death, padding_mask
            hadm_id, times, varis, values, saps, sofa, death, _, padding_mask = [x.to(device) for x in batch]

            # 모델에 전달할 outcomes (hadm_id, saps, sofa, death) - 필요 시 정의
            outcomes = torch.stack([hadm_id, saps, sofa, death], dim=1)

            # 추론
            outputs = model(times, varis, values, padding_mask=padding_mask, 
                            pretrain=False, outcomes=outcomes)

            # 모델이 반환하는 사망 확률 (배치 x 1)
            pred_death = outputs['pred_death'].squeeze(-1)  # (batch_size,)
            # 실제 라벨
            label_death = outputs['death']                  # (batch_size,)

            # CPU로 옮겨서 numpy로 변환
            all_preds.append(pred_death.detach().cpu().numpy())
            all_labels.append(label_death.detach().cpu().numpy())

    # 전체 배치를 합쳐서 (N,) 형태가 됨
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # AUROC (roc_auc_score), AUPRC (average_precision_score)는 확률값 사용
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)

    # Precision, Recall, F1, Accuracy 등은 threshold=0.5로 이진화하여 계산
    pred_binary = (all_preds >= 0.5).astype(int)

    prec = precision_score(all_labels, pred_binary)
    rec  = recall_score(all_labels, pred_binary)
    f1   = f1_score(all_labels, pred_binary)
    acc  = accuracy_score(all_labels, pred_binary)

    metrics = {
        'AUROC': auroc,
        'AUPRC': auprc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'Accuracy': acc
    }

    return metrics


def pc2012_validate_model(model, valid_loader, device):
    model.eval()
    results = []


    with torch.no_grad():
        for batch in valid_loader:
            hadm_id, times, varis, values, _, _, _, mask_t, padding_mask = [x.to(device) for x in batch]

            # Forward pass
            outputs = model(times, varis, values, padding_mask, pretrain=True, pretrain_mask = mask_t)

            # Gather outputs
            for i in range(times.size(0)):  # Loop through the batch
                result = pd.DataFrame({
                    'hadm_id': hadm_id[i].cpu().numpy().tolist(),
                    'Time': times[i].cpu().numpy().tolist(),
                    'Variable': varis[i].cpu().numpy().tolist(),
                    'True Value': values[i].cpu().numpy().tolist(),
                    'Predicted Value': outputs['forecast'][i].cpu().numpy().tolist(),
                    'Mask': outputs['mask'][i].cpu().numpy().tolist()
                })
                results.append(result)

    # Convert results to a DataFrame
    df = pd.concat(results)
    return df

def pretrain_model(model, train_loader, valid_loader, epochs, optimizer, device,
                    patience=7, model_save_path='best_pretrained_dmask_model.pt'):
    """
    Self-supervised learning function of model
    
    Parameters
    ----------
    model : nn.Module
        STraTS model to train
    train_loader : DataLoader 
        (hadm_id, times, varis, values, saps, sofa, death, mask_t, padding_mask) for training data
    valid_loader : DataLoader
        (hadm_id, times, varis, values, saps, sofa, death, mask_t, padding_mask) for validation data   
    epochs : int
        Number of epochs to pretrain the model
    optimizer : PyTorch optimizer
        Optimizer for learning rate scheduling
    device : torch.device
        CUDA or CPU, device to run model
    patience : int
        Number of epochs to train with no improvements in validation loss
    model_save_path : str
        Path to save the model, should be '~.pt'     
    
    Returns
    -------
    None, but prints epoch-wise loss and loss plot
    """
    writer = SummaryWriter()
    model.to(device)
    best_valid_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0

        with tqdm(train_loader, desc=f"[Pretrain] Epoch [{epoch+1}/{epochs}] - Training", leave=False) as pbar:
            for batch in pbar:
                # 예상되는 batch 구조: (times, varis, values, query_time, death, padding_mask)
                # 하지만 pretrain에서는 saps/sofa/death가 필요하지 않으므로 '_'로 받거나 무시.
                hadm_id, times, varis, values, query_time, _, mask_t, padding_mask = [x.to(device) for x in batch]
                
                optimizer.zero_grad()
                outputs = model(times, varis, values, 
                                padding_mask=padding_mask, 
                                pretrain=True,
                                pretrain_mask = mask_t)  # <-- pretrain=True
                loss = outputs['loss']

                loss.backward()
                optimizer.step()
                train_loss += loss.item()        

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            with tqdm(valid_loader, desc=f"[Pretrain] Epoch [{epoch+1}/{epochs}] - Validation", leave=False) as pbar:
                for batch in pbar:
                    hadm_id, times, varis, values, query_time, _, mask_t, padding_mask = [x.to(device) for x in batch]
                    outputs = model(times, varis, values, 
                                    padding_mask=padding_mask, 
                                    pretrain=True,
                                    pretrain_mask = mask_t)
                    valid_loss += outputs['loss'].item()
        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)

        print(f"[Pretrain] Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Valid Loss = {valid_loss:.4f}")

        # Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered (pretrain).")
            break

    writer.close()
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, valid_losses, label='Valid Loss')
    plt.title('Pretraining Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def train_model(model : nn.Module, train_loader : DataLoader, valid_loader : DataLoader, epochs : int, 
                optimizer : torch.optim.Optimizer, device : torch.device,
                 patience=7, freeze=False, model_save_path='best_downstream_dmask_model.pt'):
    """
    Supervised learning function of model
    
    Parameters
    ----------
    model : nn.Module
        STraTS model to train
    train_loader : DataLoader 
        (hadm_id, times, varis, values, saps, sofa, death, mask_t, padding_mask) for training data
    valid_loader : DataLoader
        (hadm_id, times, varis, values, saps, sofa, death, mask_t, padding_mask) for validation data   
    epochs : int
        Number of epochs to train the model
    optimizer : PyTorch optimizer
        Optimizer for learning rate scheduling
    device : torch.device
        CUDA or CPU, device to run model
    patience : int
        Number of epochs to train with no improvements in validation loss
    freeze : bool
        Whether to update the embedding part of the model or not
    model_save_path : str
        Path to save the model, should be '~.pt' 

    
    
    Returns
    -------
    loss_df : pd.DataFrame
        Contains ['Train loss', 'Train SAPS', 'Train SOFA', 'Train DEATH',
               'Valid loss', 'Valid SAPS', 'Valid SOFA', 'Valid DEATH',
               'Valid AUROC', 'Valid AUPRC']
    """

    writer = SummaryWriter()
    model.to(device)
    best_valid_loss = float('inf')
    early_stop_counter = 0

    train_losses = []
    valid_losses = []

    # AUROC / AUPRC 기록용
    valid_aurocs = []
    valid_auprcs = []

    for epoch in range(epochs):
        # ===================================================
        # (1) Training phase
        # ===================================================
        model.train()
        train_loss = 0
        with tqdm(train_loader, desc=f"[Downstream] Epoch [{epoch+1}/{epochs}] - Training", leave=False) as pbar:
            for batch in pbar:
                hadm_id, times, varis, values, query_time, death, _, padding_mask = [x.to(device) for x in batch]
                
                # outcomes: shape (batch_size, 3) => [hadm_id, query_time, death_label]
                outcomes = torch.stack([hadm_id, query_time, death], dim=1)

                optimizer.zero_grad()
                # pretrain=False, 전달된 outcomes로 loss 계산
                outputs = model(times, varis, values, 
                                padding_mask=padding_mask, 
                                pretrain=False,
                                freeze_pretrained = freeze, 
                                outcomes=outcomes)

                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        # epoch 단위 평균
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        # ===================================================
        # (2) Validation phase
        # ===================================================
        valid_loss = 0

        # -----------------------------
        # (a) AUROC/AUPRC 계산 위해 전체 배치 예측/라벨 누적
        # -----------------------------
        death_preds_all = []
        death_labels_all = []

        model.eval()
        with torch.no_grad():
            with tqdm(valid_loader, desc=f"[Downstream] Epoch [{epoch+1}/{epochs}] - Validation", leave=False) as pbar:
                for batch in pbar:
                    hadm_id, times, varis, values, query_time, death, _, padding_mask = [x.to(device) for x in batch]
                    outcomes = torch.stack([hadm_id, query_time, death], dim=1)

                    outputs = model(times, varis, values, 
                                    padding_mask=padding_mask, 
                                    pretrain=False,
                                    freeze_pretrained=freeze, 
                                    outcomes=outcomes)
                    loss = outputs['loss']
                    valid_loss += loss.item()

                    # -----------------------
                    # (b) death pred & label
                    # -----------------------
                    # pred_death: (batch_size, 1) => sigmoid output
                    # death_label: (batch_size,)
                    pred_death  = outputs['pred_death'].squeeze(-1)  # shape: [bsz]
                    death_label = outputs['death']  # shape: [bsz]

                    # cpu로 옮겨 numpy 변환
                    pred_death_np  = pred_death.detach().cpu().numpy()
                    death_label_np = death_label.detach().cpu().numpy()

                    death_preds_all.append(pred_death_np)
                    death_labels_all.append(death_label_np)

        # (c) epoch 단위 평균 Loss
        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)

        # -----------------------
        # (d) AUROC / AUPRC 계산
        # -----------------------
        # stack 전체 death 예측/라벨 => shape (N,)
        death_preds_all = np.concatenate(death_preds_all, axis=0)
        death_labels_all = np.concatenate(death_labels_all, axis=0)

        auroc = roc_auc_score(death_labels_all, death_preds_all)
        auprc = average_precision_score(death_labels_all, death_preds_all)

        valid_aurocs.append(auroc)
        valid_auprcs.append(auprc)

        print(f"[Epoch {epoch+1}] Train Loss={train_loss:.4f}, Valid Loss={valid_loss:.4f}, "
              f"Valid AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

        # (e) Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered (downstream).")
            break

    writer.close()

    # (3) 에폭별 결과 시각화/데이터프레임 생성
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, valid_losses, label='Valid Loss')
    plt.title('Downstream Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 추가: AUROC, AUPRC 시각화(옵션)
    plt.figure()
    plt.plot(epochs_range, valid_aurocs, label='Valid AUROC')
    plt.plot(epochs_range, valid_auprcs, label='Valid AUPRC')
    plt.title('Death Prediction Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.show()

    # (4) DataFrame
    # zip(*...)로 행=epoch, 열=각 metric
    data = list(zip(train_losses, valid_losses, valid_aurocs, valid_auprcs))
    columns = ['Train loss', 'Valid loss', 'Valid AUROC', 'Valid AUPRC']
    loss_df = pd.DataFrame(data=data, columns=columns)

    return loss_df

def evaluate_model(model, test_loader, device):
    """
    Test set에 대해 모델 추론 후, 다양한 분류 지표를 계산 + 
    각 (hadm_id, query_time) 별로 embedding을 저장하여 반환.
    """
    model.eval()  # 추론 모드
    all_preds = []
    all_labels = []

    # embedding 저장을 위한 리스트
    all_hadm_ids = []
    all_query_times = []
    all_embeddings = []

    with torch.no_grad():
        with tqdm(test_loader, desc=f"[Evaluating on dataset]", leave=False) as pbar:
            for batch in pbar:
                # 배치 구조 예:
                # hadm_id, times, varis, values, query_time, death, _, padding_mask
                hadm_id, times, varis, values, query_time, death, _, padding_mask = [x.to(device) for x in batch]

                # 모델에 전달할 outcomes (hadm_id, query_time, death)
                outcomes = torch.stack([hadm_id, query_time, death], dim=1)

                # 추론
                outputs = model(
                    times, varis, values, 
                    padding_mask=padding_mask, 
                    pretrain=False, 
                    outcomes=outcomes
                )

                # 사망 확률 (batch_size,)
                pred_death = outputs['pred_death'].squeeze(-1) 
                label_death = outputs['death']               # (batch_size,)
                
                # 임베딩 (batch_size, embed_dim)
                emb = outputs['emb']  # 최종 임베딩

                # CPU로 옮겨서 numpy로 변환
                pred_cpu = pred_death.detach().cpu().numpy()
                label_cpu = label_death.detach().cpu().numpy()
                emb_cpu   = emb.detach().cpu().numpy()
                hadm_cpu  = hadm_id.detach().cpu().numpy()
                qtime_cpu = query_time.detach().cpu().numpy()

                # 지표 계산용
                all_preds.append(pred_cpu)
                all_labels.append(label_cpu)

                # 임베딩 저장
                # 배치 내 각 row별 (hadm_id, query_time, embedding)을 기록
                for i in range(len(pred_cpu)):
                    all_hadm_ids.append(hadm_cpu[i])
                    all_query_times.append(qtime_cpu[i])
                    all_embeddings.append(emb_cpu[i])   # shape=(embed_dim,)

    # (A) 분류 지표 계산
    all_preds = np.concatenate(all_preds)  # shape=(N,)
    all_labels = np.concatenate(all_labels)  # shape=(N,)

    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)

    pred_binary = (all_preds >= 0.5).astype(int)
    prec = precision_score(all_labels, pred_binary)
    rec  = recall_score(all_labels, pred_binary)
    f1   = f1_score(all_labels, pred_binary)
    acc  = accuracy_score(all_labels, pred_binary)

    # 추가: Brier Score 계산
    brier = brier_score_loss(all_labels, all_preds)
    
    # 추가: 특정 recall 수준 (여기서는 0.8)에서의 precision 계산
    target_recall = 0.8
    precisions, recalls, _ = precision_recall_curve(all_labels, all_preds)
    # recall이 target_recall 이상인 구간에서의 최대 precision 선택
    if np.any(recalls >= target_recall):
        precision_at_target = np.max(precisions[recalls >= target_recall])
    else:
        precision_at_target = np.nan

    metrics = {
        'AUROC': auroc,
        'AUPRC': auprc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'Accuracy': acc,
        'Brier': brier,
        'Precision@Recall_0.8': precision_at_target
    }

    # (B) 임베딩 결과
    # all_hadm_ids, all_query_times, all_embeddings
    # 여기서는 dict나 별도 구조로 반환할 수 있음
    embeddings_data = {
        'hadm_id': np.array(all_hadm_ids),
        'query_time': np.array(all_query_times),
        'embedding': np.array(all_embeddings),  # shape=(N, embed_dim)
    }
    
    # (C) Precision-Recall Curve 그리기 (Test Data)
    precisions, recalls, _ = precision_recall_curve(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.lineplot(x=recalls, y=precisions, marker=',')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim(0, 1)  # x축 범위를 0.0 ~ 1.0로 고정
    plt.ylim(0, 1)  # y축 범위를 0.0 ~ 1.0로 고정
    plt.grid(True)
    plt.show()

    return metrics, embeddings_data


def validate_model(model, valid_loader, device):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in valid_loader:
            hadm_id, times, varis, values, query_time, death, mask_t, padding_mask = [x.to(device) for x in batch]

            # Forward pass
            outputs = model(times, varis, values, padding_mask, pretrain=True, pretrain_mask = mask_t)

            # Gather outputs
            for i in range(times.size(0)):  # Loop through the batch
                result = pd.DataFrame({
                    'hadm_id': hadm_id[i].cpu().numpy().tolist(),
                    'Time': times[i].cpu().numpy().tolist(),
                    'Variable': varis[i].cpu().numpy().tolist(),
                    'True Value': values[i].cpu().numpy().tolist(),
                    'Predicted Value': outputs['forecast'][i].cpu().numpy().tolist(),
                    'Mask': outputs['mask'][i].cpu().numpy().tolist()
                })
                results.append(result)

    # Convert results to a DataFrame
    df = pd.concat(results)
    return df
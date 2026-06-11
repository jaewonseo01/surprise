import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import wandb
import pandas as pd
import numpy as np
import datetime
import gc
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from create_data import create_inputs_from_data
from explain_attrs_rd import explain_attributions_common
from utils_rd import getStats, getStats_static, tensorize_normalize
from models_rd import Raindrop_Mod, STraTSModel, STraTSModelGR, STraTSGRVar, SurpriseSTraTSGRVar, STraTS, SurpriseSTraTS, SurpriseSTraTSLn, SurpSTraTSLnWa, SurpriseSTraTSLnVT_SeparateValue,SepSTraTS
from run_models import prepare_data, train_model, get_embs, train_model_STraTS, get_embs_STraTS, train_model_STraTSGR, get_embs_STraTSGR, get_embs_STraTSGR_varwise, get_masks_Surprise  # prepare_data, train_model 정의된 모듈

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pacmap import PaCMAP

def plot_pacmap(data, cols, label, name, 
                n_components=2,
                n_neighbors=None,
                MN_ratio=0.5,
                FP_ratio=2.0,
                random_state=42,
                figsize=(8,8),
                cmap='tab10',
                save_dir='./'):
    """
    data      : pandas.DataFrame
    cols      : list of column names to embed
    label     : column name in `data` to color the points by
    name      : base name for the saved figure (without extension)
    n_neighbors, MN_ratio, FP_ratio : PaCMAP hyperparameters (or None for defaults)
    """
    # 1) Prepare data matrix & labels
    X = data[cols].values
    y_raw = data[label].values

    # 2) Factorize if categorical
    if data[label].dtype == 'object' or data[label].dtype.name == 'category':
        y, categories = pd.factorize(y_raw)
    else:
        y = y_raw
        categories = None

    # print(categories)

    # 3) Run PaCMAP
    embedder = PaCMAP(n_components=n_components,
                      n_neighbors=n_neighbors,
                      MN_ratio=MN_ratio,
                      FP_ratio=FP_ratio,
                      random_state=random_state)
    X2 = embedder.fit_transform(X)

    # 4) Plot
    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        X2[:,0], X2[:,1],
        c=y,
        cmap=cmap,
        alpha=0.4,
        s=10,
        edgecolor='w',
        linewidth=0.4    
    )
    plt.title(f"PaCMAP: {name}", fontsize=16)
    plt.xlabel("PaCMAP 1")
    plt.ylabel("PaCMAP 2")

    # 5) Legend
    cmap_obj = scatter.cmap
    norm     = scatter.norm

    if categories is not None:
        handles = [
            Patch(color=cmap_obj(norm(code)), label=str(cat))
            for code, cat in enumerate(categories)
        ]
        plt.legend(handles=handles, title=label,
                   bbox_to_anchor=(1.05,1), loc='upper left')
    else:
        handles, _   = scatter.legend_elements(prop="colors", alpha=0.7)
        unique_vals  = sorted(set(y))
        plt.legend(handles, unique_vals, title=label,
                   bbox_to_anchor=(1.05,1), loc='upper left')

    plt.tight_layout()

    # 6) Save
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[✓] PaCMAP plot saved to {out_path}")

def check_for_inputs(data_dir = './processed_data', overwrite=False, max_len=1000):
    if overwrite==False:
        print("🪄 Overwriting is False: Checking for data existence")
        try: # Check mimic
            PTdict_list = np.load(f'{data_dir}/PTdict_list_mimic.npy', allow_pickle=True)
            print("✅ MIMIC input exists")
        except:
            print("❗ No MIMIC input, creating one")
            create_inputs_from_data('mimic', max_len=max_len)
        try: # Check eICU
            PTdict_list = np.load(f'{data_dir}/PTdict_list_eicu.npy', allow_pickle=True)
            print("✅ eICU input exists")
        except:
            print("❗ No eICU input, creating one")
            create_inputs_from_data('eicu', max_len=max_len)
    else:
        print("🪄 Overwriting is true: Creating data")
        create_inputs_from_data('mimic', max_len=max_len)
        create_inputs_from_data('eicu', max_len=max_len)
    print("✅ Data is ready.")

def run_experiment(source, task, model_name, batch_size=32, d_ob=4, weighted_loss=False, note="nah", use_dann=True, pretrain=True, exp_seed=42):
    """
    task : One of ['aki', 'cf', 'los', 'mor', 'readm']
    source : One of ['mimic', 'eicu']
    batch_size : Over 64 probably slows down
    d_ob : Embedding dimension of each sensor
    """
    print(f"[{model_name}]🪄 Starting experiment for: {task} on {source}")
    print(f"    Source : {source}")
    print(f"    Task : {task}")
    torch.manual_seed(exp_seed)
    np.random.seed(exp_seed)
    torch.cuda.manual_seed(exp_seed)
    torch.cuda.manual_seed_all(exp_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(exp_seed)
    # Index of corresponding outcome column
    dict_idx = {'aki' : 1,
                'cf' : 2,
                'los' : 3,
                'mor' : 4,
                'readm' : 5}
    idx = dict_idx[task]
    
    # Target data
    target = 'mimic' if source =='eicu' else 'eicu'

    # Initiate wandb
    wandb.init(
        project="replearner",
        config={
            "model": model_name,
            "data_name": source,
            "outcome_idx": idx, # 1:AKI,2:CF,3:LoS,4:Mortality
            "task": task,
            "architecture": "Raindrop",
            "batch_size": batch_size,
            "lr": 5e-4,
            "num_epochs": 30,
            "patience": 7,
            'weight_loss': weighted_loss,
            'note': note,
            'seed': exp_seed,
        }
    )
    wandb.run.name = f'{note}_source_{source}_{task}_{datetime.datetime.today().day}'    
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    # Prepare source and target data (MIMIC / eICU)
    data, data_target, class_weights = prepare_data(
        processed_data_dir="./processed_data",
        source=source,
        target=target,
        outcome_idx=idx,
        split_seed=42,
        train_frac=0.6,
        val_frac=0.2,
    )
    print(class_weights)
    print("✅ Source / Target data is ready.")

    # Define shapes and model
    shape = data["shape"]
    F = shape["F"]
    D = shape["D"]
    T = shape["T"]
    n_classes = 2
    if model_name == 'raindrop':
        model = Raindrop_Mod(
            d_inp=F, # dimension of input : 35 vars
            d_model=F * d_ob, # Input dimension * Embedding dimensions
            nhead=4, # Attention heads
            nhid=2 * F * d_ob, # Double the initial input
            nlayers=2,
            dropout=0.2,
            max_len=T, # Pre-defined in data creating stage
            d_static=D, # Number of static vars : 3 (this is embedded to F dimensions)
            MAX=100,
            aggreg="mean",
            n_classes=n_classes,
            global_structure=torch.ones(F, F)
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        if weighted_loss:
            class_weights = torch.FloatTensor(class_weights).to(device)
            criterion = nn.CrossEntropyLoss(class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        # 4) 학습 및 평가
        best_model = train_model(
            run_name=f'{config.data_name}_{config.task}_weighted_loss',
            data=data,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            patience=config.patience,
            device=device,
            wandb=wandb
        )
        

        # 5) 결과 저장
        try:
            get_embs(data, best_model, device, config.batch_size, name=f'{model_name}_{source}_{config.task}_source', data_type='train')
            get_embs(data, best_model, device, config.batch_size, name=f'{model_name}_{source}_{config.task}_source', data_type='val')
            get_embs(data, best_model, device, config.batch_size, name=f'{model_name}_{source}_{config.task}_source', data_type='test')

            get_embs(data_target, best_model, device, config.batch_size, name=f'{model_name}_{target}_{config.task}_target', data_type='train', metric=True)
            get_embs(data_target, best_model, device, config.batch_size, name=f'{model_name}_{target}_{config.task}_target', data_type='val', metric=True)
            get_embs(data_target, best_model, device, config.batch_size, name=f'{model_name}_{target}_{config.task}_target', data_type='test', metric=True)

            print("✅ Training complete. 모델과 예측 결과를 저장했습니다.")
        except:
            print("❗ Training complete, but failed to get embeddings. Maybe OOM? 모델을 저장했습니다. ")
        wandb.finish()

    elif model_name == "strats":
        combined_data = {
            'source': data,
            'target': data_target
        }
        print(F)

        model = STraTS(                
            num_features=F,
            embed_dim=32,
            static_dim=D, 
            num_heads=4, 
            num_blocks=2, 
            ff_dim=64,
            dropout=0.2, 
            time_activation='relu', 
            value_activation='relu', 
            final_emb_type='fusion', 
            fusion_emb_weight=0.5,
            final_emb_weight=0.5).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=config.lr)

        criterion = nn.BCEWithLogitsLoss()

        # 4) 학습 및 평가
        best_model = train_model_STraTSGR(
            run_name=f'{config.data_name}_{config.task}_{note}',
            source=source,
            data=data,
            model=model,
            use_pretrain=pretrain,
            use_dann=False,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            patience=config.patience,
            device=device,
            wandb=wandb,
            seed=exp_seed,
            varwise=False,
            model_name='STraTS'
        )
        

        # 5) 결과 저장
        try:
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='train', metric=True, wandb=wandb)
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='val', metric=True, wandb=wandb)
            df_source, df_target = get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='test', metric=True, wandb=wandb)

            print("✅ Training complete. 모델과 예측 결과를 저장했습니다.")
            
        except:
            print("❗ Training complete, but failed to get embeddings. Maybe OOM? 모델을 저장했습니다. ")
        

        # Visualize results
        try:
            cols_dict = {
                'all' : [f"emb_{i}" for i in range(0, 64)],
                'vars' : [f"emb_{i}" for i in range(0, 32)],
                'static' : [f"emb_{i}" for i in range(32, 64)],
            }
            vis_source = 'MIMIC_' if source=='mimic' else 'eICU_'
            vis_target = 'eICU_' if source=='mimic' else 'MIMIC_'
            df_source['label'] = vis_source + df_source['label'].astype(str)
            df_target['label'] = vis_target + df_target['label'].astype(str)
            df = pd.concat([df_source, df_target])
            df = df.sort_values(by='label')
            for k in cols_dict.keys():
                plot_pacmap(
                data=df,
                cols=cols_dict[k],
                label="label",
                name=f"{vis_source}{note}_{task}_{k}",
                save_dir="./pacmap_figs",
                cmap='coolwarm'
                )
            print("✅ PaCMAP. 모델과 예측 결과를 저장했습니다.")
        except:
            print("❗ Visualization failed.")
        wandb.finish()


    elif model_name == "stratsgr":
        combined_data = {
            'source': data,
            'target': data_target
        }
        
        var_groups = {'vital': [i for i in range(0,6)],
                      'lab': [i for i in range(6,19)],
                      'treat': [i for i in range(19,35)]}        
        model = STraTSModelGR(                
            num_features=F,
            embed_dim=8,
            static_dim=D, 
            num_heads=4, 
            num_blocks=2, 
            ff_dim=64,
            dropout=0.2, 
            time_activation='relu', 
            value_activation='tanh', 
            final_emb_type='fusion', 
            fusion_emb_weight=0.5,
            final_emb_weight=0.5,
            domain_lambda=0.5).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=config.lr)

        ratio = class_weights[1] / class_weights[0]
        print(ratio)
        if weighted_loss:
            pos_weight = torch.tensor(ratio, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # 4) 학습 및 평가
        best_model = train_model_STraTSGR(
            run_name=f'{config.data_name}_{config.task}_{note}',
            source=source,
            data=combined_data,
            model=model,
            use_pretrain=pretrain,
            use_dann=use_dann,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            patience=config.patience,
            device=device,
            wandb=wandb,
            seed=exp_seed,
        )
        

        # 5) 결과 저장
        try:
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{config.task}', data_type='train', metric=True, wandb=wandb)
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{config.task}', data_type='val', metric=True, wandb=wandb)
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{config.task}', data_type='test', metric=True, wandb=wandb)
            get_embs_STraTSGR_varwise(combined_data, best_model, var_groups, device, config.batch_size, name=f'{note}_{config.task}', data_type='train')
            get_embs_STraTSGR_varwise(combined_data, best_model, var_groups, device, config.batch_size, name=f'{note}_{config.task}', data_type='val')
            get_embs_STraTSGR_varwise(combined_data, best_model, var_groups, device, config.batch_size, name=f'{note}_{config.task}', data_type='test')

            print("✅ Training complete. 모델과 예측 결과를 저장했습니다.")
        except:
            print("❗ Training complete, but failed to get embeddings. Maybe OOM? 모델을 저장했습니다. ")
        wandb.finish()

    elif model_name == "stratsvar":
        combined_data = {
            'source': data,
            'target': data_target
        }
        var_groups = {'vital': [i for i in range(0,6)],
                      'lab': [i for i in range(6,19)],
                      'treat': [i for i in range(19,35)]}
        model = STraTSGRVar(                
            num_features=F,
            embed_dim=32,
            var_groups=var_groups,
            static_dim=D, 
            num_heads=4, 
            num_blocks=2, 
            ff_dim=64,
            dropout=0.2, 
            time_activation='relu', 
            value_activation='tanh', 
            final_emb_type='fusion', 
            fusion_emb_weight=0.5,
            final_emb_weight=0.5,
            domain_lambda=0.5).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        ratio = class_weights[1] / class_weights[0]
        print(ratio)
        if weighted_loss:
            pos_weight = torch.tensor(ratio, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # 4) 학습 및 평가
        best_model = train_model_STraTSGR(
            run_name=f'{config.data_name}_{config.task}_{note}',
            source=source,
            data=data,
            model=model,
            use_pretrain=pretrain,
            use_dann=use_dann,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            patience=config.patience,
            device=device,
            wandb=wandb,
            seed=exp_seed,
            varwise=True
        )
        

        # 5) 결과 저장
        try:
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{config.task}', data_type='train', metric=True, wandb=wandb)
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{config.task}', data_type='val', metric=True, wandb=wandb)
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{config.task}', data_type='test', metric=True, wandb=wandb)

            print("✅ Training complete. 모델과 예측 결과를 저장했습니다.")
        except:
            print("❗ Training complete, but failed to get embeddings. Maybe OOM? 모델을 저장했습니다. ")
        wandb.finish()

    elif model_name == "surpstratsvar":
        combined_data = {
            'source': data, # MIMIC
            'target': data_target # eICU
        }
        var_groups = {'vital': [i for i in range(0,6)],
                      'lab': [i for i in range(6,19)],
                      'treat': [i for i in range(19,35)]}
        model = SurpriseSTraTSGRVar(                
            num_features=F,
            embed_dim=32,
            var_groups=var_groups,
            static_dim=D, 
            num_heads=4, 
            num_blocks=2, 
            ff_dim=64,
            dropout=0.2, 
            time_activation='relu', 
            value_activation='tanh', 
            final_emb_type='fusion', 
            fusion_emb_weight=0.5,
            final_emb_weight=0.5,
            domain_lambda=0.5).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        ratio = class_weights[1] / class_weights[0]
        print(ratio)
        if weighted_loss:
            pos_weight = torch.tensor(ratio, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # 4) 학습 및 평가
        best_model = train_model_STraTSGR(
            run_name=f'{config.data_name}_{config.task}_{note}',
            source=source,
            data=data,
            model=model,
            use_pretrain=pretrain,
            use_dann=use_dann,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            patience=config.patience,
            device=device,
            wandb=wandb,
            seed=exp_seed,
            varwise=True,
            model_name='SurpSTraTSVar'
        )
        torch.cuda.empty_cache()

        # 5) 결과 저장
        try:
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='train', metric=True, wandb=wandb)
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='val', metric=True, wandb=wandb)
            df_source, df_target = get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='test', metric=True, wandb=wandb)

            print("✅ Training complete. 모델과 예측 결과를 저장했습니다.")
            
        except:
            print("❗ Training complete, but failed to get embeddings. Maybe OOM? 모델을 저장했습니다. ")
        
        # Get surprise masks
        try:
            get_masks_Surprise(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}')
            print("✅ Mask retrieved succesfully. yay!")
        except:
            print("❗ Mask retrieval failed.")

        # Visualize results
        try:
            cols_dict = {
                'all' : [f"emb_{i}" for i in range(0, 128)],
                'vital' : [f"emb_{i}" for i in range(0, 32)],
                'lab' : [f"emb_{i}" for i in range(32, 64)],
                'treat' : [f"emb_{i}" for i in range(64, 96)],
                'static' : [f"emb_{i}" for i in range(96, 128)],
            }
            vis_source = 'MIMIC_' if source=='mimic' else 'eICU_'
            vis_target = 'eICU_' if source=='mimic' else 'MIMIC_'
            df_source['label'] = vis_source + df_source['label'].astype(str)
            df_target['label'] = vis_target + df_target['label'].astype(str)
            df = pd.concat([df_source, df_target])
            df = df.sort_values(by='label')
            for k in cols_dict.keys():
                plot_pacmap(
                data=df,
                cols=cols_dict[k],
                label="label",
                name=f"{vis_source}{note}_{task}_{k}",
                save_dir="./pacmap_figs",
                cmap='coolwarm'
                )
            print("✅ PaCMAP. 모델과 예측 결과를 저장했습니다.")
        except:
            print("❗ Visualization failed.")
        wandb.finish()

    elif model_name == "surpstrats":
        combined_data = {
            'source': data, # MIMIC
            'target': data_target # eICU
        }
        var_groups = {'vital': [i for i in range(0,6)],
                      'lab': [i for i in range(6,19)],
                      'treat': [i for i in range(19,35)]}
        model = SurpriseSTraTS(                
            num_features=F,
            embed_dim=32,
            static_dim=D, 
            num_heads=4, 
            num_blocks=2, 
            ff_dim=64,
            dropout=0.2, 
            time_activation='relu', 
            value_activation='tanh', 
            final_emb_type='fusion', 
            fusion_emb_weight=0.5,
            final_emb_weight=0.5,
            domain_lambda=0.5,
            sim_threshold=0.97).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        ratio = class_weights[1] / class_weights[0]
        print(ratio)
        if weighted_loss:
            pos_weight = torch.tensor(ratio, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # 4) 학습 및 평가
        best_model = train_model_STraTSGR(
            run_name=f'{config.data_name}_{config.task}_{note}',
            source=source,
            data=data,
            model=model,
            use_pretrain=pretrain,
            use_dann=use_dann,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            patience=config.patience,
            device=device,
            wandb=wandb,
            seed=exp_seed,
            varwise=True,
            model_name='SurpSTraTS'
        )
        torch.cuda.empty_cache()

        # 5) 결과 저장
        try:
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='train', metric=True, wandb=wandb)
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='val', metric=True, wandb=wandb)
            df_source, df_target = get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='test', metric=True, wandb=wandb)

            print("✅ Training complete. 모델과 예측 결과를 저장했습니다.")
            
        except:
            print("❗ Training complete, but failed to get embeddings. Maybe OOM? 모델을 저장했습니다. ")
        
        # Get surprise masks
        try:
            get_masks_Surprise(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}')
            print("✅ Mask retrieved succesfully. yay!")
        except:
            print("❗ Mask retrieval failed.")

        # Visualize results
        try:
            cols_dict = {
                'all' : [f"emb_{i}" for i in range(0, 64)],
                'vars' : [f"emb_{i}" for i in range(0, 32)],
                'static' : [f"emb_{i}" for i in range(32, 64)],
            }
            vis_source = 'MIMIC_' if source=='mimic' else 'eICU_'
            vis_target = 'eICU_' if source=='mimic' else 'MIMIC_'
            df_source['label'] = vis_source + df_source['label'].astype(str)
            df_target['label'] = vis_target + df_target['label'].astype(str)
            df = pd.concat([df_source, df_target])
            df = df.sort_values(by='label')
            for k in cols_dict.keys():
                plot_pacmap(
                data=df,
                cols=cols_dict[k],
                label="label",
                name=f"{vis_source}{note}_{task}_{k}",
                save_dir="./pacmap_figs",
                cmap='coolwarm'
                )
            print("✅ PaCMAP. 모델과 예측 결과를 저장했습니다.")
        except:
            print("❗ Visualization failed.")
        wandb.finish()

    elif model_name == "surpstratsln":
        combined_data = {
            'source': data, # MIMIC
            'target': data_target # eICU
        }
        var_groups = {'vital': [i for i in range(0,6)],
                      'lab': [i for i in range(6,19)],
                      'treat': [i for i in range(19,35)]}
        model = SurpriseSTraTSLn(                
            num_features=F,
            embed_dim=32,
            static_dim=D, 
            num_heads=4, 
            num_blocks=2, 
            ff_dim=64,
            dropout=0.2, 
            time_activation='relu', 
            value_activation='relu', 
            final_emb_type='fusion', 
            fusion_emb_weight=0.5,
            final_emb_weight=0.5,
            domain_lambda=0.5,
            sim_threshold=0.97,
            direction='future').to(device)

        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        ratio = class_weights[1] / class_weights[0]
        print(ratio)
        if weighted_loss:
            pos_weight = torch.tensor(ratio, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # 4) 학습 및 평가
        best_model = train_model_STraTSGR(
            run_name=f'{config.data_name}_{config.task}_{note}',
            source=source,
            data=data,
            model=model,
            use_pretrain=pretrain,
            use_dann=use_dann,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            patience=config.patience,
            device=device,
            wandb=wandb,
            seed=exp_seed,
            varwise=True,
            model_name='SurpSTraTSLn'
        )
        torch.cuda.empty_cache()

        # 5) 결과 저장
        try:
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='train', metric=True, wandb=wandb)
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='val', metric=True, wandb=wandb)
            df_source, df_target = get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='test', metric=True, wandb=wandb)

            print("✅ Training complete. 모델과 예측 결과를 저장했습니다.")
            
        except:
            print("❗ Training complete, but failed to get embeddings. Maybe OOM? 모델을 저장했습니다. ")
        
        # Get surprise masks
        try:
            get_masks_Surprise(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}')
            print("✅ Mask retrieved succesfully. yay!")
        except:
            print("❗ Mask retrieval failed.")

        # Visualize results
        try:
            cols_dict = {
                'all' : [f"emb_{i}" for i in range(0, 64)],
                'vars' : [f"emb_{i}" for i in range(0, 32)],
                'static' : [f"emb_{i}" for i in range(32, 64)],
            }
            vis_source = 'MIMIC_' if source=='mimic' else 'eICU_'
            vis_target = 'eICU_' if source=='mimic' else 'MIMIC_'
            df_source['label'] = vis_source + df_source['label'].astype(str)
            df_target['label'] = vis_target + df_target['label'].astype(str)
            df = pd.concat([df_source, df_target])
            df = df.sort_values(by='label')
            for k in cols_dict.keys():
                plot_pacmap(
                data=df,
                cols=cols_dict[k],
                label="label",
                name=f"{vis_source}{note}_{task}_{k}",
                save_dir="./pacmap_figs",
                cmap='coolwarm'
                )
            print("✅ PaCMAP. 모델과 예측 결과를 저장했습니다.")
        except:
            print("❗ Visualization failed.")
        wandb.finish()
    
    elif model_name == "surpstratslnpast":
        combined_data = {
            'source': data, # MIMIC
            'target': data_target # eICU
        }
        var_groups = {'vital': [i for i in range(0,6)],
                      'lab': [i for i in range(6,19)],
                      'treat': [i for i in range(19,35)]}
        model = SurpriseSTraTSLn(                
            num_features=F,
            embed_dim=32,
            static_dim=D, 
            num_heads=4, 
            num_blocks=2, 
            ff_dim=64,
            dropout=0.2, 
            time_activation='relu', 
            value_activation='tanh', 
            final_emb_type='fusion', 
            fusion_emb_weight=0.5,
            final_emb_weight=0.5,
            domain_lambda=0.5,
            sim_threshold=0.97,
            direction='past').to(device)

        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        ratio = class_weights[1] / class_weights[0]
        print(ratio)
        if weighted_loss:
            pos_weight = torch.tensor(ratio, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # 4) 학습 및 평가
        best_model = train_model_STraTSGR(
            run_name=f'{config.data_name}_{config.task}_{note}',
            source=source,
            data=data,
            model=model,
            use_pretrain=pretrain,
            use_dann=use_dann,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            patience=config.patience,
            device=device,
            wandb=wandb,
            seed=exp_seed,
            varwise=True,
            model_name='SurpSTraTSLnPast')
        torch.cuda.empty_cache()

        # 5) 결과 저장
        try:
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='train', metric=True, wandb=wandb)
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='val', metric=True, wandb=wandb)
            df_source, df_target = get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='test', metric=True, wandb=wandb)

            print("✅ Training complete. 모델과 예측 결과를 저장했습니다.")
            
        except:
            print("❗ Training complete, but failed to get embeddings. Maybe OOM? 모델을 저장했습니다. ")
        
        # Get surprise masks
        try:
            get_masks_Surprise(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}')
            print("✅ Mask retrieved succesfully. yay!")
        except:
            print("❗ Mask retrieval failed.")

        # Visualize results
        try:
            cols_dict = {
                'all' : [f"emb_{i}" for i in range(0, 64)],
                'vars' : [f"emb_{i}" for i in range(0, 32)],
                'static' : [f"emb_{i}" for i in range(32, 64)],
            }
            vis_source = 'MIMIC_' if source=='mimic' else 'eICU_'
            vis_target = 'eICU_' if source=='mimic' else 'MIMIC_'
            df_source['label'] = vis_source + df_source['label'].astype(str)
            df_target['label'] = vis_target + df_target['label'].astype(str)
            df = pd.concat([df_source, df_target])
            df = df.sort_values(by='label')
            for k in cols_dict.keys():
                plot_pacmap(
                data=df,
                cols=cols_dict[k],
                label="label",
                name=f"{vis_source}{note}_{task}_{k}",
                save_dir="./pacmap_figs",
                cmap='coolwarm'
                )
            print("✅ PaCMAP. 모델과 예측 결과를 저장했습니다.")
        except:
            print("❗ Visualization failed.")
        wandb.finish()
    elif model_name == "surpstratslnwa":
        combined_data = {
            'source': data, # MIMIC
            'target': data_target # eICU
        }
        var_groups = {'vital': [i for i in range(0,6)],
                      'lab': [i for i in range(6,19)],
                      'treat': [i for i in range(19,35)]}
        model = SurpSTraTSLnWa(                
            num_features=F,
            embed_dim=32,
            static_dim=D, 
            num_heads=4, 
            num_blocks=2, 
            ff_dim=64,
            dropout=0.2, 
            time_activation='relu', 
            value_activation='tanh', 
            final_emb_type='fusion', 
            fusion_emb_weight=0.5,
            final_emb_weight=0.5,
            domain_lambda=0.5,
            sim_threshold=0.97,
            direction='future').to(device)

        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        ratio = class_weights[1] / class_weights[0]
        print(ratio)
        if weighted_loss:
            pos_weight = torch.tensor(ratio, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # 4) 학습 및 평가
        best_model = train_model_STraTSGR(
            run_name=f'{config.data_name}_{config.task}_{note}',
            source=source,
            data=data,
            model=model,
            use_pretrain=pretrain,
            use_dann=use_dann,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            patience=config.patience,
            device=device,
            wandb=wandb,
            seed=exp_seed,
            varwise=True,
            model_name='SurpSTraTSLnWa'
        )
        with torch.no_grad():
            w = torch.nn.functional.softmax(best_model.triplet_weight_logits, dim=-1)  # [F+1,3]
            print(w)
        torch.cuda.empty_cache()

        # 5) 결과 저장
        try:
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='train', metric=True, wandb=wandb)
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='val', metric=True, wandb=wandb)
            df_source, df_target = get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='test', metric=True, wandb=wandb)

            print("✅ Training complete. 모델과 예측 결과를 저장했습니다.")
            
        except:
            print("❗ Training complete, but failed to get embeddings. Maybe OOM? 모델을 저장했습니다. ")
        
        # Get surprise masks
        try:
            get_masks_Surprise(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}')
            print("✅ Mask retrieved succesfully. yay!")
        except:
            print("❗ Mask retrieval failed.")

        # Visualize results
        try:
            cols_dict = {
                'all' : [f"emb_{i}" for i in range(0, 64)],
                'vars' : [f"emb_{i}" for i in range(0, 32)],
                'static' : [f"emb_{i}" for i in range(32, 64)],
            }
            vis_source = 'MIMIC_' if source=='mimic' else 'eICU_'
            vis_target = 'eICU_' if source=='mimic' else 'MIMIC_'
            df_source['label'] = vis_source + df_source['label'].astype(str)
            df_target['label'] = vis_target + df_target['label'].astype(str)
            df = pd.concat([df_source, df_target])
            df = df.sort_values(by='label')
            for k in cols_dict.keys():
                plot_pacmap(
                data=df,
                cols=cols_dict[k],
                label="label",
                name=f"{vis_source}{note}_{task}_{k}",
                save_dir="./pacmap_figs",
                cmap='coolwarm'
                )
            print("✅ PaCMAP. 모델과 예측 결과를 저장했습니다.")
        except:
            print("❗ Visualization failed.")
        wandb.finish()

    elif model_name == "separatesurp":
        combined_data = {
            'source': data, # MIMIC
            'target': data_target # eICU
        }
        var_groups = {'vital': [i for i in range(0,6)],
                      'lab': [i for i in range(6,19)],
                      'treat': [i for i in range(19,35)]}
        model = SurpriseSTraTSLnVT_SeparateValue(                
            num_features=F,
            embed_dim=32,
            static_dim=D, 
            num_heads=4, 
            num_blocks=2, 
            ff_dim=64,
            dropout=0.2, 
            time_activation='relu', 
            value_activation='tanh', 
            final_emb_type='fusion', 
            fusion_emb_weight=0.5,
            final_emb_weight=0.5,
            sim_threshold=1.9,
            direction='future',
            gated_vars=None).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=config.lr)

        criterion = nn.BCEWithLogitsLoss()

        # 4) 학습 및 평가
        best_model = train_model_STraTSGR(
            run_name=f'{config.data_name}_{config.task}_{note}',
            source=source,
            data=data,
            model=model,
            use_pretrain=pretrain,
            use_dann=False,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            patience=config.patience,
            device=device,
            wandb=wandb,
            seed=exp_seed,
            varwise=True,
            model_name=f'separateSurpFuture'
        )

        torch.cuda.empty_cache()

        # 5) 결과 저장
        try:
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='train', metric=True, wandb=wandb)
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='val', metric=True, wandb=wandb)
            df_source, df_target = get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='test', metric=True, wandb=wandb)

            print("✅ Training complete. 모델과 예측 결과를 저장했습니다.")
            
        except:
            print("❗ Training complete, but failed to get embeddings. Maybe OOM? 모델을 저장했습니다. ")
        
        # Get surprise masks
        try:
            get_masks_Surprise(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}')
            print("✅ Mask retrieved succesfully. yay!")
        except:
            print("❗ Mask retrieval failed.")

        # Visualize results
        try:
            cols_dict = {
                'all' : [f"emb_{i}" for i in range(0, 64)],
                'vars' : [f"emb_{i}" for i in range(0, 32)],
                'static' : [f"emb_{i}" for i in range(32, 64)],
            }
            vis_source = 'MIMIC_' if source=='mimic' else 'eICU_'
            vis_target = 'eICU_' if source=='mimic' else 'MIMIC_'
            df_source['label'] = vis_source + df_source['label'].astype(str)
            df_target['label'] = vis_target + df_target['label'].astype(str)
            df = pd.concat([df_source, df_target])
            df = df.sort_values(by='label')
            for k in cols_dict.keys():
                plot_pacmap(
                data=df,
                cols=cols_dict[k],
                label="label",
                name=f"{vis_source}{note}_{task}_{k}",
                save_dir="./pacmap_figs",
                cmap='coolwarm'
                )
            print("✅ PaCMAP. 모델과 예측 결과를 저장했습니다.")
        except:
            print("❗ Visualization failed.")
        wandb.finish()

    elif model_name == "separate":
        combined_data = {
            'source': data,
            'target': data_target
        }
        print(F)

        model = SepSTraTS(                
            num_features=F,
            embed_dim=32,
            static_dim=D, 
            num_heads=4, 
            num_blocks=2, 
            ff_dim=64,
            dropout=0.2, 
            time_activation='relu', 
            value_activation='tanh', 
            final_emb_type='fusion', 
            fusion_emb_weight=0.5,
            final_emb_weight=0.5).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=config.lr)

        criterion = nn.BCEWithLogitsLoss()

        # 4) 학습 및 평가
        best_model = train_model_STraTSGR(
            run_name=f'{config.data_name}_{config.task}_{note}',
            source=source,
            data=data,
            model=model,
            use_pretrain=pretrain,
            use_dann=False,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            patience=config.patience,
            device=device,
            wandb=wandb,
            seed=exp_seed,
            varwise=False,
            model_name='SepSTraTS'
        )
        

        # 5) 결과 저장
        try:
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='train', metric=True, wandb=wandb)
            get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='val', metric=True, wandb=wandb)
            df_source, df_target = get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{source}_{config.task}', data_type='test', metric=True, wandb=wandb)

            print("✅ Training complete. 모델과 예측 결과를 저장했습니다.")
            
        except:
            print("❗ Training complete, but failed to get embeddings. Maybe OOM? 모델을 저장했습니다. ")
        

        # Visualize results
        try:
            cols_dict = {
                'all' : [f"emb_{i}" for i in range(0, 64)],
                'vars' : [f"emb_{i}" for i in range(0, 32)],
                'static' : [f"emb_{i}" for i in range(32, 64)],
            }
            vis_source = 'MIMIC_' if source=='mimic' else 'eICU_'
            vis_target = 'eICU_' if source=='mimic' else 'MIMIC_'
            df_source['label'] = vis_source + df_source['label'].astype(str)
            df_target['label'] = vis_target + df_target['label'].astype(str)
            df = pd.concat([df_source, df_target])
            df = df.sort_values(by='label')
            for k in cols_dict.keys():
                plot_pacmap(
                data=df,
                cols=cols_dict[k],
                label="label",
                name=f"{vis_source}{note}_{task}_{k}",
                save_dir="./pacmap_figs",
                cmap='coolwarm'
                )
            print("✅ PaCMAP. 모델과 예측 결과를 저장했습니다.")
        except:
            print("❗ Visualization failed.")
        wandb.finish()
def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ("yes","true","t","y","1"):   return True
    if v in ("no","false","f","n","0"):   return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--model", default="stratsgr")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--weighted_loss", type=str2bool, default=False)
    p.add_argument("--note", default="STraTS_dann")
    p.add_argument("--pretrain", type=str2bool, default=True), 
    p.add_argument("--use_dann", type=str2bool, default=True)
    p.add_argument("--input_len", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # 하나의 run만 실행
    check_for_inputs(max_len=args.input_len)
    run_experiment(args.source, args.task, args.model,
                   args.batch_size, weighted_loss=args.weighted_loss,
                   note=args.note, use_dann=args.use_dann, pretrain=args.pretrain,
                   exp_seed=args.seed)

    # out.to_feather(f"./rd_results/predictions_target_{data_type}_{name}.feather")
    try:
        df = explain_attributions_common(f'{args.note}_{args.source}_{args.task}', 0.25)
    except:
        print("❗ Attribution failed.")

    # 명시적 정리(선택)
    torch.cuda.empty_cache()
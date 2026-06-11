import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import wandb
import pandas as pd
import numpy as np
import os
import random

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from utils_rd import getStats, getStats_static, tensorize_normalize, get_data_split, tensorize_normalize_other
from models_rd import Raindrop_Mod, SurpriseSTraTSGRVar, STraTS, SurpriseSTraTS, SurpriseSTraTSLn, SurpSTraTSLnWa, SurpriseSTraTSLnVT_SeparateValue
from run_models import train_model, get_embs, train_model_STraTSGR, get_embs_STraTSGR, get_masks_Surprise, average_precision_calibrated

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pacmap import PaCMAP
plt.switch_backend('agg')
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


def run_experiment(dataset, model_name, batch_size=32, d_ob=4, note="nah", sim_threshold=0.95, pretrain=True, exp_seed=42):
    """
    source : One of ['p12', 'p19']
    batch_size : Over 64 probably slows down
    d_ob : Embedding dimension of each sensor
    """
    print(f"[{model_name}]🪄 Starting experiment for {dataset}")
    print(f"    Source : {dataset}")
    torch.manual_seed(exp_seed)
    np.random.seed(exp_seed)
    torch.cuda.manual_seed(exp_seed)
    torch.cuda.manual_seed_all(exp_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(exp_seed)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    # device = torch.device("cpu")

    base_path = f"./processed_data/{dataset}data"
    # Prepare data
    if dataset == 'P12':
        d_static = 9
        d_inp = 36
        static_info = 1
        max_len = 215
        vital = [8, 14, 21, 22, 30]
        lab = list(set(np.arange(d_inp)) - set(vital))
        n_classes = 2
        var_groups = {'vital': vital,
                      'lab': lab}
        
    elif dataset == 'P19':
        d_static = 6
        d_inp = 34
        static_info = 1
        max_len = 60
        n_classes = 2
        vital = [0, 1, 3, 4, 5]
        lab = list(set(np.arange(d_inp)) - set(vital))
        var_groups = {'vital': vital,
                'lab': lab}

    elif dataset == 'PAM':
        d_static = 0
        d_inp = 17
        static_info = None
        max_len = 600
        n_classes = 8
        vars = [i for i in range(0,d_inp)]        
        var_groups = {'vars': vars}



    print("✅ Source / Target data is ready.")

    # Define shapes and model
    aggreg = 'mean'

    MAX = 100

    n_runs = 1
    n_splits = 5
    subset = False

    acc_arr = np.zeros((n_splits, n_runs))
    auprc_arr = np.zeros((n_splits, n_runs))
    auroc_arr = np.zeros((n_splits, n_runs))
    precision_arr = np.zeros((n_splits, n_runs))
    recall_arr = np.zeros((n_splits, n_runs))
    F1_arr = np.zeros((n_splits, n_runs))

    rows = []
    for k in range(n_splits):

        split_idx = k + 1
        print(f'Split id: {split_idx}')

        # Initiate wandb
        wandb.init(
            project="Surprise",
            config={
                "model": model_name,
                "data_name": dataset,
                "batch_size": batch_size,
                "lr": 5e-4,
                "num_epochs": 30,
                "patience": 7,
                'note': note,
                'seed': exp_seed,
            }
        )
        wandb.run.name = f'{note}_{split_idx}'    
        config = wandb.config
        if dataset == 'P12':
            if subset == True:
                split_path = '/splits/phy12_split_subset' + str(split_idx) + '.npy'
            else:
                split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
        elif dataset == 'P19':
            split_path = '/splits/phy19_split' + str(split_idx) + '_new.npy'
        elif dataset == 'eICU':
            split_path = '/splits/eICU_split' + str(split_idx) + '.npy'
        elif dataset == 'PAM':
            split_path = '/splits/PAM_split_' + str(split_idx) + '.npy'

        # prepare the data:
        Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, split_type='random', reverse=False,
                                                                  baseline=False, dataset=dataset,
                                                                  predictive_label='mortality')
        print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))

        if dataset == 'P12' or dataset == 'P19':
            T, F = Ptrain[0]['arr'].shape
            D = len(Ptrain[0]['extended_static'])

            Ptrain_tensor = np.zeros((len(Ptrain), T, F))
            Ptrain_static_tensor = np.zeros((len(Ptrain), D))

            for i in range(len(Ptrain)):
                Ptrain_tensor[i] = Ptrain[i]['arr']
                Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

            mf, stdf = getStats(Ptrain_tensor)
            ms, ss = getStats_static(Ptrain_static_tensor, dataset=dataset)

            Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize(Ptrain, ytrain, mf,
                                                                                                         stdf, ms, ss)
            Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize(Pval, yval, mf, stdf, ms, ss)
            Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize(Ptest, ytest, mf, stdf, ms, ss)
            train_ids = [p['id'] for p in Ptrain]
            val_ids = [p['id'] for p in Pval]
            test_ids = [p['id'] for p in Ptest]
        elif dataset == 'PAM':
            T, F = Ptrain[0].shape
            D = 1

            Ptrain_tensor = Ptrain
            Ptrain_static_tensor = np.zeros((len(Ptrain), D))

            mf, stdf = getStats(Ptrain)
            Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize_other(Ptrain, ytrain, mf, stdf)
            Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize_other(Pval, yval, mf, stdf)
            Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize_other(Ptest, ytest, mf, stdf)
            train_ids = [p['id'] for p in Ptrain]
            val_ids = [p['id'] for p in Pval]
            test_ids = [p['id'] for p in Ptest]

        Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2)
        Pval_tensor = Pval_tensor.permute(1, 0, 2)
        Ptest_tensor = Ptest_tensor.permute(1, 0, 2)

        Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
        Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
        Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

        data_dict = {
        'Ptrain': Ptrain_tensor,
        'Pval': Pval_tensor,
        'Ptest': Ptest_tensor,
        'Ptrain_time':Ptrain_time_tensor,
        'Pval_time':Pval_time_tensor,
        'Ptest_time':Ptest_time_tensor,
        'Ptrain_static':Ptrain_static_tensor,
        'Pval_static':Pval_static_tensor,
        'Ptest_static':Ptest_static_tensor,
        'ytrain': ytrain_tensor,
        'yval': yval_tensor,
        'ytest': ytest_tensor,
        'shape': {'T':T,'F':F,'D':D},
        'train_ids':train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids,
        }
    
        if model_name == 'raindrop':
            model = Raindrop_Mod(
                d_inp=F, # dimension of input
                d_model=F * d_ob, # Input dimension * Embedding dimensions
                nhead=4, # Attention heads
                nhid=2 * F * d_ob, # Double the initial input
                nlayers=2,
                dropout=0.2,
                max_len=T, # Pre-defined in data creating stage
                d_static=D, # Number of static vars (this is embedded to F dimensions)
                MAX=100,
                aggreg="mean",
                n_classes=n_classes,
                global_structure=torch.ones(F, F)
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=config.lr)

            criterion = nn.CrossEntropyLoss()

            # 4) 학습 및 평가
            best_model = train_model(
                run_name=f'{dataset}_{split_idx}',
                data=data_dict,
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
                get_embs(dataset, best_model, device, config.batch_size, name=f'{model_name}_{split_idx}', data_type='train')
                get_embs(dataset, best_model, device, config.batch_size, name=f'{model_name}_{split_idx}', data_type='val')
                get_embs(dataset, best_model, device, config.batch_size, name=f'{model_name}_{split_idx}', data_type='test')


                print("✅ Training complete. 모델과 예측 결과를 저장했습니다.")
            except:
                print("❗ Training complete, but failed to get embeddings. Maybe OOM? 모델을 저장했습니다. ")
            wandb.finish()
        elif model_name == "strats":
            combined_data = {
                'source': data_dict, # Both the same data
                'target': data_dict # Remnant of when experimenting with two datasets
            }
            print(F)
            print(len(var_groups))

            model = STraTS(                
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
                run_name=f'{dataset}_{split_idx}_{note}',
                source=dataset,
                data=data_dict,
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
                model_name=f'STraTS{sim_threshold}_{split_idx}'
            )
            

            # 5) 결과 저장
            try:
                get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}', data_type='train', metric=True, wandb=wandb)
                get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}', data_type='val', metric=True, wandb=wandb)
                df_source, df_target = get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}', data_type='test', metric=True, wandb=wandb)

                print("✅ Training complete. 모델과 예측 결과를 저장했습니다.")
                
            except:
                print("❗ Training complete, but failed to get embeddings. Maybe OOM? 모델을 저장했습니다. ")
            
           
            wandb.finish()
        elif model_name == "surpstrats":
            combined_data = {
                'source': data_dict, # Both the same data
                'target': data_dict # Remnant of when experimenting with two datasets
            }
            print(F)
            print(len(var_groups))

            model = SurpriseSTraTS(                
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
                sim_threshold=sim_threshold).to(device)

            optimizer = optim.AdamW(model.parameters(), lr=config.lr)

            criterion = nn.BCEWithLogitsLoss()

            # 4) 학습 및 평가
            best_model = train_model_STraTSGR(
                run_name=f'{dataset}_{split_idx}_{note}',
                source=dataset,
                data=data_dict,
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
                model_name=f'SurpSTraTS{sim_threshold}_{split_idx}'
            )
            

            # 5) 결과 저장
            try:
                get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}', data_type='train', metric=True, wandb=wandb)
                get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}', data_type='val', metric=True, wandb=wandb)
                df_source, df_target = get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}', data_type='test', metric=True, wandb=wandb)

                print("✅ Training complete. 모델과 예측 결과를 저장했습니다.")
                
            except:
                print("❗ Training complete, but failed to get embeddings. Maybe OOM? 모델을 저장했습니다. ")
            try:
                get_masks_Surprise(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}')
                print("✅ Mask retrieved succesfully. yay!")
            except:
                print("❗ Mask retrieval failed.")
            
           
            wandb.finish()

        elif model_name == "surpstratsln":
            combined_data = {
                'source': data_dict, # Both the same data
                'target': data_dict # Remnant of when experimenting with two datasets
            }
            print(F)
            print(len(var_groups))

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
                sim_threshold=sim_threshold).to(device)

            optimizer = optim.AdamW(model.parameters(), lr=config.lr)

            criterion = nn.BCEWithLogitsLoss()

            # 4) 학습 및 평가
            best_model = train_model_STraTSGR(
                run_name=f'{dataset}_{split_idx}_{note}',
                source=dataset,
                data=data_dict,
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
                model_name=f'SurpSTraTSLn{sim_threshold}_{split_idx}'
            )
            

            # 5) 결과 저장
            try:
                get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}', data_type='train', metric=True, wandb=wandb)
                get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}', data_type='val', metric=True, wandb=wandb)
                df_source, df_target = get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}', data_type='test', metric=True, wandb=wandb)

                print("✅ Training complete. 모델과 예측 결과를 저장했습니다.")
                
            except:
                print("❗ Training complete, but failed to get embeddings. Maybe OOM? 모델을 저장했습니다. ")
            try:
                get_masks_Surprise(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}')
                print("✅ Mask retrieved succesfully. yay!")
            except:
                print("❗ Mask retrieval failed.")
            
           
            wandb.finish()

        elif model_name == "surpstratsvar":
            combined_data = {
                'source': data_dict, # Both the same data
                'target': data_dict # Remnant of when experimenting with two datasets
            }
            print(F)
            print(len(var_groups))

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
                value_activation='relu', 
                final_emb_type='fusion', 
                fusion_emb_weight=0.5,
                final_emb_weight=0.5,
                sim_threshold=sim_threshold).to(device)

            optimizer = optim.AdamW(model.parameters(), lr=config.lr)

            criterion = nn.BCEWithLogitsLoss()

            # 4) 학습 및 평가
            best_model = train_model_STraTSGR(
                run_name=f'{dataset}_{split_idx}_{note}',
                source=dataset,
                data=data_dict,
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
                model_name=f'SurpSTraTSVar{sim_threshold}_{split_idx}'
            )
            

            # 5) 결과 저장
            try:
                get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}', data_type='train', metric=True, wandb=wandb)
                get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}', data_type='val', metric=True, wandb=wandb)
                df_source, df_target = get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}', data_type='test', metric=True, wandb=wandb)

                print("✅ Training complete. 모델과 예측 결과를 저장했습니다.")
                
            except:
                print("❗ Training complete, but failed to get embeddings. Maybe OOM? 모델을 저장했습니다. ")
            
            # Get surprise masks
            try:
                get_masks_Surprise(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}')
                print("✅ Mask retrieved succesfully. yay!")
            except:
                print("❗ Mask retrieval failed.")

            # # Visualize results
            # try:
            #     counts = 0
            #     cols_dict ={}
            #     for i in var_groups.keys():                
            #         cols_dict[i] = [f"emb_{j}" for j in range(counts*32, (counts+1)*32)]
            #         counts += 1
            #     cols_dict['static'] = [f"emb_{j}" for j in range(counts*32, (counts+1)*32)]
            #     cols_dict['all'] = [f"emb_{j}" for j in range(0, (counts+1)*32)]

            #     df_source = df_source.sort_values(by='label')
            #     for k in cols_dict.keys():
            #         plot_pacmap(
            #         data=df_source,
            #         cols=cols_dict[k],
            #         label="label",
            #         name=f"{dataset}_{split_idx}_{note}",
            #         save_dir="./pacmap_figs",
            #         cmap='coolwarm'
            #         )
            #     print("✅ PaCMAP. 모델과 예측 결과를 저장했습니다.")
            # except:
            #     print("❗ Visualization failed.")
            wandb.finish()
        elif model_name == "surpstratslnfuture":
            combined_data = {
                'source': data_dict, # Both the same data
                'target': data_dict # Remnant of when experimenting with two datasets
            }
            print(F)
            print(len(var_groups))

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
                sim_threshold=sim_threshold,
                direction='future').to(device)

            optimizer = optim.AdamW(model.parameters(), lr=config.lr)

            criterion = nn.BCEWithLogitsLoss()

            # 4) 학습 및 평가
            best_model = train_model_STraTSGR(
                run_name=f'{dataset}_{split_idx}_{note}',
                source=dataset,
                data=data_dict,
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
                model_name=f'SurpSTraTSLnFuture{sim_threshold}_{split_idx}'
            )

            torch.cuda.empty_cache()

            # 5) 결과 저장
            try:
                get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}', data_type='train', metric=True, wandb=wandb)
                get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}', data_type='val', metric=True, wandb=wandb)
                df_source, df_target = get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}', data_type='test', metric=True, wandb=wandb)

                print("✅ Training complete. 모델과 예측 결과를 저장했습니다.")
                
            except:
                print("❗ Training complete, but failed to get embeddings. Maybe OOM? 모델을 저장했습니다. ")
            try:
                get_masks_Surprise(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}')
                print("✅ Mask retrieved succesfully. yay!")
            except:
                print("❗ Mask retrieval failed.")
            
           
            wandb.finish()
            
        elif model_name == "separatesurp":
            combined_data = {
                'source': data_dict, # Both the same data
                'target': data_dict # Remnant of when experimenting with two datasets
            }
            print(F)
            print(len(var_groups))

            model = SurpriseSTraTSLnVT_SeparateValue(                
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
                sim_threshold=sim_threshold,
                direction='future').to(device)

            optimizer = optim.AdamW(model.parameters(), lr=config.lr)

            criterion = nn.BCEWithLogitsLoss()

            # 4) 학습 및 평가
            best_model = train_model_STraTSGR(
                run_name=f'{dataset}_{split_idx}_{note}',
                source=dataset,
                data=data_dict,
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
                model_name=f'separateSurpFuture{sim_threshold}_{split_idx}'
            )

            torch.cuda.empty_cache()

            # 5) 결과 저장
            try:
                get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}', data_type='train', metric=True, wandb=wandb)
                get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}', data_type='val', metric=True, wandb=wandb)
                df_source, df_target = get_embs_STraTSGR(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}', data_type='test', metric=True, wandb=wandb)

                print("✅ Training complete. 모델과 예측 결과를 저장했습니다.")
                
            except:
                print("❗ Training complete, but failed to get embeddings. Maybe OOM? 모델을 저장했습니다. ")
            try:
                get_masks_Surprise(combined_data, best_model, device, config.batch_size, name=f'{note}_{split_idx}')
                print("✅ Mask retrieved succesfully. yay!")
            except:
                print("❗ Mask retrieval failed.")
            
           
            wandb.finish()
        acc = accuracy_score(df_source['label'], df_source['pred'])
        auroc = roc_auc_score(df_source['label'], df_source['proba'])
        auprc = average_precision_score(df_source['label'], df_source['proba'])
        auprc_c = average_precision_calibrated(df_source['label'], df_source['proba'])
        rows.append({
            'Accuracy': acc,
            'AUROC' : auroc,
            'AUPRC' : auprc,
            'AUPRC_c' : auprc_c,
        })
    
    perf_df = pd.DataFrame(rows)
    perf_df.to_csv(f'./rd_results/performance_{note}.csv')

def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ("yes","true","t","y","1"):   return True
    if v in ("no","false","f","n","0"):   return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default='P12')
    p.add_argument("--model", default="surpstratsvar")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--similarity", type=float, default=0.95)
    p.add_argument("--note", default="nah")
    p.add_argument("--pretrain", type=str2bool, default=True), 
    p.add_argument("--input_len", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # 하나의 run만 실행
    run_experiment(args.dataset, args.model,
                   args.batch_size, 
                   note=args.note,
                   sim_threshold=args.similarity,
                   pretrain=args.pretrain,
                   exp_seed=args.seed)

    # 명시적 정리(선택)
    torch.cuda.empty_cache()
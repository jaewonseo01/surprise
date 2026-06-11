import sys
import subprocess
from itertools import product
# Run as subprocess to ensure GPU resources are freed
# Process per model

def run_one(model, dataset, pretrain, sim, note):
    cmd = [
        sys.executable, "baseline_exp.py", # file to run
        # Enter args needed
        "--dataset", dataset,
        "--model", model,
        "--batch_size", "32",
        "--similarity", sim,
        "--note", note,
        "--pretrain", pretrain,
        "--input_len", "800",
        "--seed", "42"
    ]
    # 실시간 출력 그대로 터미널에
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    model_list = [
          #'stratsgr',
          #'stratsvar',
          #'surpstratsvar'
          #'strats',
          #'surpstrats',
          #'surpstratslnfuture',
          #'surpstratsvar',
          'separatesurp',
          ]
    pretrain_list = [
         # "True",
         "False"
         ]
    dataset_list = [
        "P12",
        #"P19",        
        ]
    sim_list = [
        "1.95",
        "1.8",
        "1.6",
        "1.4",
        "1"
        #"0.9",
        # "0.8",
        #"0.5",
        #"0.1",
        #"0.0",
    ]
    # STraTS model with LN


    for dataset, model, pretrain in product(dataset_list, model_list, pretrain_list):        
        pre = '_Pre' if pretrain=="True" else ''
        if model in ['surpstrats', 'surpstratsln', 'surpstratsvar', 'surpstratslnfuture', 'separatesurp']:
            for sim in sim_list:
                similarity_string = int(float(sim) * 100)
                note = f"{model}{similarity_string}{pre}_{dataset}"
                try:
                    run_one(model=model, dataset=dataset, pretrain=pretrain, note=note, sim=sim)
                except:
                    pass
        else:
            note = f"{model}{pre}_{dataset}Ln"
            try:
                run_one(model=model, dataset=dataset, pretrain=pretrain, note=note, sim="0.95")
            except:
                pass
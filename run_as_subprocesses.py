import sys
import subprocess
from itertools import product
# Run as subprocess to ensure GPU resources are freed
# Process per model
def run_one(model, sauce, task, dann, pretrain, note, weight):
    cmd = [
        sys.executable, "run_raindrop_model.py", # file to run
        # Enter args needed
        "--source", sauce,
        "--task",  task,
        "--model", model,
        "--batch_size", "8",
        "--weighted_loss", weight,
        "--note",  note,
        "--use_dann", dann,
        "--pretrain", pretrain,
        "--input_len", "800",
        "--seed", "42"
    ]
    # 실시간 출력 그대로 터미널에
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    model_list = [
          'strats',
          #'stratsvar',
          #'surpstratsvar',
          'surpstratsln',
          #'surpstratslnpast',
          'separatesurp',
          #'separate'
          ]
    pretrain_list = [
         "True",
         "False"
         ]
    weighted_list = [
        #"True",
        "False"
    ]
    dann_list = [#"True",
                 "False"
                 ]
    sauce_list = [
        "mimic" ,
        "eicu"
        ]
    task_list  = [
                # "aki",
                #"cf",
                #"mor",
                "los",
                #'readm'
                ]

    for sauce, model, pretrain, dann, task, weight in product(sauce_list, model_list, pretrain_list, dann_list,  task_list, weighted_list):
        note = f"{model}_pre_{pretrain}_gapdata"
        run_one(model=model, sauce=sauce, task=task, dann=dann, pretrain=pretrain, note=note, weight=weight)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract_values(txtFilepath):
    val_accuracy, val_loss = [], []
    with open(txtFilepath, "r") as f:
        lines = f.readlines()
        is_accuracy = False  
        for line in lines:
            line = line.strip()  
            if not line: 
                continue
            if "Validation Accuracy:" in line:
                is_accuracy = True 
                continue 
            val_accuracy.append(float(line)) if is_accuracy else val_loss.append(float(line)) 

    return val_loss, val_accuracy


classfication_comparison_trial = {
    "Trial_Number": [1],  
    "Optimizer": ["Adam"],
    "Number_Layers": [3],
    "List_Layers": [[512, 256, 128]],  
    "Number_epochs": [50],
    "f1_score": [1.0000],
    "result": [[]] ,
    "difference_percentage":[0]
}

trialResult = pd.read_csv("Task-2 Trial (1).csv")
reference = trialResult['Class'].to_list()
classfication_comparison_trial["result"] = [reference]
comparison_df = pd.DataFrame(classfication_comparison_trial)
print(comparison_df)



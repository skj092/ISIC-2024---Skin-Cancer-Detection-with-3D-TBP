import os

print('======= INFERENCE =======')
#os.system("python src/infer_pt.py AUROC0.5155_Loss0.4690_epoch1.bin")

# Run Ensemble
os.system("python ./src/train_lgbm_eff_ensemble.py")

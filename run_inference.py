import os

print('======= PREPROCESS TEST IMAGES =======')
os.system(f"python ./src/test_data_preprocessing.py")
print('======= INFERENCE =======')
os.system(f"python ./src/inference.py")

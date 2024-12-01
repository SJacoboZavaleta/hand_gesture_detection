import pandas as pd
import os
from datetime import datetime

TIME_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_TYPE = 'data2'
if DATA_TYPE == 'data2':
    df = pd.read_csv('src/efficientNet/asl_dataset_info.csv')
elif DATA_TYPE == 'data3':
    df = pd.read_csv('src/efficientNet/unified_data_dataset_info.csv')

print(df.head())

print(os.getcwd())
if DATA_TYPE == 'data2':
        OUTPUT_PATH =  os.path.join('..', '..', 'results', 'efficientnet_v2_s_data2', f'evaluation_{TIME_STAMP}')
elif DATA_TYPE == 'data3':
    OUTPUT_PATH =  os.path.join('..', '..', 'results', 'efficientnet_v2_s_data3', f'evaluation_{TIME_STAMP}')

print(OUTPUT_PATH)
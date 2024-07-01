import pyarrow.parquet as pq
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from glob import glob
import pandas as pd
import os

filenames = sorted(glob('./*.snappy.parquet'))
df = pd.read_csv('data.csv')
for filename in filenames:
    tmp = pq.read_table(filename).to_pandas()

    file_names = []
    texts = []
    width = []
    height = []
    for i in tqdm(range(len(tmp))):
        if i > 60000:
            break
        try:
            response = requests.get(tmp.iloc[i]['url'])
            image_data = response.content
            save_path = f'images/{tmp.iloc[i]["id"]}.jpg'

            file_names.append(save_path)
            texts.append(tmp.iloc[i]['text'])
            width.append(tmp.iloc[i]['width'])
            height.append(tmp.iloc[i]['height'])
            with open(save_path, 'wb') as f:
                f.write(image_data)
            
        except:
            continue
    result_df = pd.DataFrame({'image_file': file_names, 'text': texts, 'width': width, 'height': height})
    df = pd.concat([df, result_df]).reset_index(drop=True)
df.reset_index(drop=True).to_csv('data.csv', index=False)
print(len(df))

#%%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("metmuseum/the-metropolitan-museum-of-art-open-access")
import os

print(os.listdir(path))

import pandas as pd

import os

csv_path = os.path.join(path, "MetObjects.csv")

df = pd.read_csv(csv_path)

print(df.head())
# %%

import pandas as pd

df = pd.read_hdf(r"C:\leftImg8bit\attention_results\crop_results.h5")
for i, row in df.iterrows():

    if row['is_ignore'] == "False":
        row['is_ignore'] = False
    else:
        row['is_ignore'] = True
df['is_ignore'] = df['is_ignore'].astype(bool)
df['is_true'] = df['is_true'].astype(bool)
df['seq'] = df['seq'].astype('uint8')
df['path'] = df['path'].astype('str')


print(df['is_ignore'].unique())
print(df.dtypes)

df.to_hdf(r"C:\leftImg8bit\attention_results\crop_results2.h5", key="s", mode="w")
#print(df['is_ignore'])
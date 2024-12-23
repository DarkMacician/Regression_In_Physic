import pandas as pd

df = pd.read_csv("emails1.csv")
# for i in df['Labels']:
#     if "SPAM" in i:
#         print(i)
print(df)
import numpy as npy
import pandas as pd

data = npy.array([1, 2, 3])
df = pd.DataFrame({"values": data})
print(df)
import pandas as pd
import numpy as np




# mthod: given a datframe with col="dff", return a numpy array stacking each row of dff
def df_col_to_array(df:pd.DataFrame, col:str)->np.ndarray:
    return np.stack(df[col].values)
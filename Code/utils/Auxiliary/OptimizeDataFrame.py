### Import packages ###
import pandas as pd

### Function ###
def OptimizeDataFrame(df):
    for col in df.columns:
        if df[col].dtype == 'float64':                          # Convert float64 (64-bit) to float32 (32-bit) or smaller if possible
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':                          # Convert int64 (64-bit) to int32, int16, or int8 if possible
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df
### Packages ###
import gc
import numpy as np

### Function ###
def BatchPredictFunction(model, X, batch_size=1000):

    predictions = []
    for i in range(0, len(X), batch_size):
        batch = X.iloc[i:i + batch_size]
        pred = model.predict(batch)
        predictions.append(pred)
        gc.collect()
    return np.concatenate(predictions)
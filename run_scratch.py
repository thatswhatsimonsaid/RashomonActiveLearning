import pickle
import pandas as pd

def get_ecs(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    # This assumes your SimulationResult stores a history of ECS
    return data.get('metrics_history', {}).get('ECS', [])

# Load one seed from each
ecs_m5 = get_ecs('M5/M5_S1.pkl')
ecs_m8 = get_ecs('M8/M8_S1.pkl')

df_comp = pd.DataFrame({
    'Iteration': range(len(ecs_m5)),
    'M5_ECS (Uniform)': ecs_m5,
    'M8_ECS (BMA)': ecs_m8
})

print(df_comp.head(10))
import shutil
from pathlib import Path

RESULTS_DIR = Path("/homes/simondn/RashomonActiveLearning/results/study1_active_learning/tree_predictor")
PLOTS_DIR = RESULTS_DIR.parent / "PLOTS"

for dataset_dir in sorted(RESULTS_DIR.iterdir()):
    img_dir = dataset_dir / "accuracy_images"
    if not img_dir.exists():
        continue
    
    for img in img_dir.glob("*.png"):
        plot_type = img.stem  # e.g. "accuracy_history"
        dest_dir = PLOTS_DIR / plot_type
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dest_dir / f"{dataset_dir.name}.png")
        
print(f"Done! Check {PLOTS_DIR}")
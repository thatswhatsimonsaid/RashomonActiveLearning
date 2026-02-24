rm -f "/mnt/beegfs/homes/simondn/RashomonActiveLearning/results/study1_active_learning/tree_predictor/hypothyroid"/M*/*.pkl
find "/mnt/beegfs/homes/simondn/RashomonActiveLearning/results/study1_active_learning/tree_predictor/hypothyroid" -type d -name "M*" -empty -delete
echo "Raw .pkl files and empty method folders deleted for hypothyroid."

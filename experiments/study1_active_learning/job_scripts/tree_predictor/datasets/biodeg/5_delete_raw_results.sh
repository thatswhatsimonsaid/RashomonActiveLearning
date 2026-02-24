rm -f "/mnt/beegfs/homes/simondn/RashomonActiveLearning/results/study1_active_learning/tree_predictor/biodeg"/M*/*.pkl
find "/mnt/beegfs/homes/simondn/RashomonActiveLearning/results/study1_active_learning/tree_predictor/biodeg" -type d -name "M*" -empty -delete
echo "Raw .pkl files and empty method folders deleted for biodeg."

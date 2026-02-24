rm -f "/mnt/beegfs/homes/simondn/RashomonActiveLearning/results/study1_active_learning/tree_predictor/car_evaluation"/M*/*.pkl
find "/mnt/beegfs/homes/simondn/RashomonActiveLearning/results/study1_active_learning/tree_predictor/car_evaluation" -type d -name "M*" -empty -delete
echo "Raw .pkl files and empty method folders deleted for car_evaluation."

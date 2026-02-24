rm -f "/mnt/beegfs/homes/simondn/RashomonActiveLearning/results/study1_active_learning/tree_predictor/coffee_house"/M*/*.pkl
find "/mnt/beegfs/homes/simondn/RashomonActiveLearning/results/study1_active_learning/tree_predictor/coffee_house" -type d -name "M*" -empty -delete
echo "Raw .pkl files and empty method folders deleted for coffee_house."

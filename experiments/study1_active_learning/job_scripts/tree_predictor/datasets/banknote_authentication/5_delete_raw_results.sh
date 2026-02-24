rm -f "/mnt/beegfs/homes/simondn/RashomonActiveLearning/results/study1_active_learning/tree_predictor/banknote_authentication"/M*/*.pkl
find "/mnt/beegfs/homes/simondn/RashomonActiveLearning/results/study1_active_learning/tree_predictor/banknote_authentication" -type d -name "M*" -empty -delete
echo "Raw .pkl files and empty method folders deleted for banknote_authentication."

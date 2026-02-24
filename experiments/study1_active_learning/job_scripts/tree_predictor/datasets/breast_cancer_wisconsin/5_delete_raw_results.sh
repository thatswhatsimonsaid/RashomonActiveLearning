rm -f "/mnt/beegfs/homes/simondn/RashomonActiveLearning/results/study1_active_learning/tree_predictor/breast_cancer_wisconsin"/M*/*.pkl
find "/mnt/beegfs/homes/simondn/RashomonActiveLearning/results/study1_active_learning/tree_predictor/breast_cancer_wisconsin" -type d -name "M*" -empty -delete
echo "Raw .pkl files and empty method folders deleted for breast_cancer_wisconsin."

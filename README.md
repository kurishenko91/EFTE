# EFTE
The code for the paper "On Enhancing the Explainability and Fairness of Tree Ensembles", Emilio Carrizosa, Kseniia Kurishchenko, Dolores Romero Morales

Abstract:

Tree ensembles are one of the most powerful methodologies in Machine Learning. In this paper, we investigate how to make tree ensembles more flexible to incorporate by design explainability and fairness. While explainability helps the user understand the key features that play a role in the classification task, with fairness we ensure that the ensemble does not discriminate against a group of observations that share a sensitive attribute. We propose a Mixed Integer Linear Optimization formulation to train an ensemble of trees that apart from minimizing the misclassification error, controls for sparsity as well as the accuracy in the sensitive group. Our formulation is scalable in the number of observations since its number of binary decision variables is independent of the number of observations.
In our numerical results, we show that for standard datasets used in the fairness literature, we can dramatically enhance the fairness of the benchmark, namely the popular Random Forest, while using only a few features, all without damaging the misclassification error.

We illustrate our methodology on two binary classification datasets often used in the fairness literature, namely the PIMA diabetes dataset and the COMPAS dataset.

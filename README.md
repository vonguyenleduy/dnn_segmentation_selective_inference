# Quantifying Statistical Significance of Neural Network-based Image Segmentation by Selective Inference (NeurIPS 2022)

This package implements a novel method to quantify the reliability of neural network representation-driven hypotheses in statistical hypothesis testing framework by Selective Inference (SI). The basic idea of SI is to make conditional inferences on the selected hypotheses under the condition that they are selected. In order to use SI framework for DNN representations, we develop a new SI algorithm based on homotopy method which enables us to derive the exact (non-asymptotic) conditional sampling distribution of the DNN-driven hypotheses.

See the paper <https://openreview.net/forum?id=FlrQGoHPcvo> for more details.


## Installation & Requirements

We recommend to install or update anaconda to the latest version and use Python 3 (We used Python 3.6.4). 

Since we conduct all the experiments in parallel, please make sure that you have [mpi4py](https://mpi4py.readthedocs.io/) package in your environment. We ran the code with 40 threads.    

This package has the following requirements:

- [tensorflow](https://www.tensorflow.org) (we used v2.0.0)
- [numpy](http://numpy.org)
- [mpmath](http://mpmath.org/)
- [matplotlib](https://matplotlib.org/)
- [mpi4py](https://mpi4py.readthedocs.io/)
- [scikit-learn](http://scikit-learn.org)

## Reproducibility

Since we have already got the results in advance, you can reproduce the figures by running files in "/plot" folder. The results will be save in "/results" folder.


To reproduce the results, please see the following instructions.

**NOTE**: Due to the randomness of data generating process, we note that the results might be slightly different from the paper. However, the overall results for interpretation will not change.

All the results are shown on console.

- To create a trained network, please run
    ```
	>> python training.py
	``` 
  The model will be saved in "/model" folder.

- FPR experiments. First we need to set the value of variable n in run() function. Please set the value of n in [16, 64, 256, 1024], and then run the following commands
    - FPR for naive method
    ```
	>> mpirun -n 40 python ex1_fpr_naive.py
	``` 
    - FPR for proposed-method-oc (over-conditioning)
    ```
	>> mpirun -n 40 python ex1_fpr_proposed_oc.py
	``` 
    - FPR for proposed-method
    ```
	>> mpirun -n 40 python ex1_fpr_proposed.py
	``` 
  
- TPR experiments. First we need to set the value of delta mu. To do it, in run() function, please set the value of variable mu_2 in [0.5, 1, 1.5, 2]. Then, run the following commands
    - TPR for proposed-method-oc (over-conditioning)
    ```
	>> mpirun -n 40 python ex2_tpr_proposed_oc.py
	``` 
    - TPR for proposed-method
    ```
	>> mpirun -n 40 python ex2_tpr_proposed.py
	``` 
  
- Experiments to see the length of truncation interval.
    - For proposed-method-oc
    ```
	>> mpirun -n 40 python ex3_len_interval_proposed_oc.py
	``` 
    - For proposed-method
    ```
	>> mpirun -n 40 python ex3_len_interval_proposed.py
	``` 
  
- Experiment of counting # encounted interval when increase the dimension of input vector n. First we need to set the value of variable n in run() function. Please set the value of n in [16, 64, 256, 1024], and then run the following command
    ```
	>> mpirun -n 40 python ex4_count_no_interval.py
	``` 
    

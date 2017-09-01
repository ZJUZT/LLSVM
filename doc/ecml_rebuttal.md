We would like to first thank all the reviewers for their constructive comments. We will revise and render the manuscript according to their suggestions.
1. technical innovation (R1,2,3)
Our goal is to find the optimum local coding scheme, which is a quite challenging and non-trivial problem. Existing algorithms usually employ a predefined and fixed local coding scheme and ignore the different manifold structures of data points which may lead to a sub-optimal solution.
2. large-scale datasets (R1,2,3)
We have validated the efficacy of our algorithm on larger and more complicated datasets like MNIST and RCV1. Due to page limitation, we did not include them. Thanks for your comment and we will include them in the next version.
3. gain in accuracy (R1,2)
Our proposed algorithm achieves an obvious gain in hinge loss. The gain in accuracy will be much more obvious when all the algorithms totally converge.
4. evaluation metrics (R1)
We choose hinge loss because it is a common metric to measure the performance of SVM.
5. lack proper attribution (R1)
Our main contribution is proposing a jointly learnt local coding scheme instead of just simply combining the existing algorithms. However, we do miss some proper attributions and we are appreciated that you point it out.
6. comparison to kernel SVM (R2)
Thanks for your valuable advice and we will include the comparison to kernel SVM to make the experiments complete.
7. online fashion (R2)
We use SGD to solve optimization problem. The data points come one by one in the training phase.
8. parameters to be tuned (R3)
Our proposed method introduces a new parameter called Lipschitz to noise ratio but drops the parameter number of nearest neighbors. Actually, the time to tune the parameters is comparable to LLC-SAPL.
9. time complexity (R3)
The test time is related to the number of nearest neighbors and experiments show it is comparable LLC-SAPL. Anyway, thanks for your advice and we will provide a detailed analysis about time complexity in the next version.
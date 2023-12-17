# KNN, Data Normalisation

a) Encode each non-numerical ordinal data with numbers, i.e. (low, medium, high) => (1, 2, 3). Calculate the absolute difference between two positions, i.e. (|1 - 3| = 2 for low and high).

b) Standard normal distribution with a mean of 0 and standard deviation of 1.

a) False. When $k \neq 1$, the majority of the surrounding neighbours of a data point can be from the wrong label class, causing the data point to be misclassified.

a) Done

b) Done

c) None

d) True. Multivariate regression focuses on finding the beta coefficient, which is just the slope. The slope of the data will be in the same direction regardless of data scale. Decision tree is not affected by data scale because each rule will only compare within one feature.

# Decision Trees, Entropy/Information Gain

c) None

d) None

b) The class label will be the class with the most number of instances, the majority class. This is to keep the representation of the pruned subtree while simplifying the tree structure.

c) Done

g) If this limit is reduced, the decision tree will only be allowed to have lesser number of leaf nodes. The tree will be smaller and node impurity will increase, causing training data to decrease in accuracy. It might increase the testing accuracy if it has been overfitting before this. Otherwise, it will also decrease testing accuracy.

a)(i) P(yes) = 3/7, P(no) 4/7
Entropy = -3/7 * log_2(3/7) + -4/7 * log_2(4/7) = 0.985
a)(ii) IG = original_entropy - new_entropy
Identify best feature for root node split:

E(Stars = 2) = -1/2 * log_2(1/2) + -1/2 * log_2(1/2) = 1
E(Stars = 3) = -1/4 * log_2(1/3) + -3/4 * log_2(3/4) = 0.811
E(Stars = 4) = -1/1 * log_2(1/1) + -0/1 * log_2(0/1) = 0

E(Pool = Y) = -3/4 * log_2(3/4) + -1/4 * log_2(1/4) = 0.811
E(Pool = N) = -0/3 * log_2(0/3) + -3/3 * log_2(3/3) = 0

E(Beach = Y) = -2/3 * log_2(2/3) + -1/3 * log_2(1/3) = 0.918
E(Beach = N) = -3/4 * log_2(3/4) + -1/4 * log_2(1/4) = 0.811

E(Gym = Y) = -2/4 * log_2(2/4) + -2/4 * log_2(2/4) = 1
E(Gym = N) = -1/3 * log_2(1/3) + -2/3 * log_2(2/3) = 0.918

IG(Stars) = 0.985 - (2/7 * 1 + 3/7 * 0.811 + 1/7 * 0) = 0.352
IG(Pool) = 0.985 - (4/7 * 0.811 + 3/7 * 0) = 0.522
IG(Beach) = 0.985 - (3/7 * 0.918 + 4/7 * 0.811) = 0.128
IG(Gym) = 0.985 - (4/7 * 1 + 3/7 * 0.918) = 0.02

Take the feature with the highest IG value: Pool. When pool = Y, data is pslit into 3/4 = yes and 1/4 = no. When pool = N, data is split into 3/3 = no.

# Naive Bayes Classifier

f) Discretisation - use binning or transform numeric data into discrete intervals.
Assume distribution - assume data follow a distribution e.g. Gaussian distribution, then use the PDF to evaluate the probability for the corresponding numeric input.

d) Done

b)
P(Yes) = (1/4 * 2/4 * 2/4) * 4/10 = 0.025
P(No) = (2/6 * 2/6 * 4/6) * 6/10 = 0.044

Using learned prior, output = No

P(Yes) = (1/4 * 2/4 * 2/4) * 1/2 = 0.031
P(No) = (2/6 * 2/6 * 4/6) * 1/2 = 0.037

Using uniform prior, output = No

There is no change in the result.

a) A ranking classifier means that the algorithm predicts the order or ranking for a set of items instead of directly predicting its class label. It determines the relative order or preference for a set of data classes.

Naive Bayes is based on Bayes theorem:
P(h|D) = P(D|h) * P(h) / P(D)
Posterior = likelihood * prior / evidence

All the features in naive bayes are assumed to be conditionally independent.
The prior and likelihood (product of conditional probabilities) can be calculated from the data.
The evidence is a constant value that can be disregarded in the relative (or proportional) calculation of the posterior.

The posterior for each class is calculated after the conditional probabilities and priors are calculated from the data with the relevant class.

The posteriors are normalised to a range between 0 and 1 to look more like probabilities. This is where the ranking comes from. The classifier does not predict the class label directly, but predicts the probability of each class label (relative order).
The class label with the highest posterior is selected as the output using the maximum function.

b) Search engine optimisation - where the results are displayed from best matching with the highest probability to the worst matching with the lowest probablitty.

c) ?

# Evaluation
3)a) Done

b) Done

c) Done

e) BAR and BER are the true positive and true negative rates respectively. For highly imbalanced datasets such as classes ratio of 90:10, the classifier can easily get a high accuracy by simply predicting the majority class all the time. BAR and BER force the model to predict both the minority and majority.

e) (iii) MAE and MAPE. MAE is the $MAE = \frac{1}{n} \sum_{i=1}^n y_i - \hat{y}_i$ that measures the difference in predicted quantity. MAPE is just the percentage of that difference $MAPE = \frac{1}{n} \sum_{i=1}^n \frac{y_i - \hat{y}_i}{max(\epsilon, y_i)}$ which measures the difference in quantity relative to the truth label for easier comparison.

3)a) Cross validation randomises and segments all data into k selected folds. Each fold is held back once, and the rest is used for training. The training iterates for k times with k - 1 folds of training data. All the data points will have been part of training once and testing once, which maximises the use of available data.

b)

Truth     = [Y, Y, N, N, N]

Logits    = [0.99, 0.90, 0.60, 0.80, 0.99]

99%       = [Y, N, N, N, Y]

TP = 1
FP = 1
TN = 2
FN = 1

True positive rate = TP / (TP + FN) = 1 / (1 + 1) = 0.5
False positive rate = 1 - TN / (TN + FP) = 1 - 1 / (2 + 1) = 0.67

c) ROC curves...

# Model Selection

g) Done

h) Done

3)a) Done

b) Done

c) Done

h) (c) only. Lack of training data and model too simple will only cause underfitting. (c) might cause overfitting because the noise patterns might be captured by the model unintentionally.

b)i) Overfitting in supervised learning is when the training accuracy is much higher than the testing accuracy, e.g. 95% vs 60%. This shows the the model has memorised the training data. Underfitting is when both training and testing accuracy is low, e.g. 30% and 20%. This shows that the model has not learned any patterns in the data.

# Neural Networks

f) Done

2)a) Done

b) Raise activation threshold. Fit data without having the y-intercept passing through the origin.

c) Done

d) Done

c) False. A larger learning rate will result in a larger parameter update (w_new = w_old + alpha * gradient), which can cause the algorithm to skip pass the optimal convergence point. The learning rate should be smaller as it gets closer to convergence.

2)a) BGD is the traditional GD. For each backpropagation, all data examples are used to calculate the gradients and update the parameters. SGD will randomly select a training example to forward propagate to compute the gradient and update the weights.

BGD is determinisic. As long as initial weights stay the same, the BGD results will always be the same. SGD is probabilistic, meaning that even initial weights are the same, SGD results will vary each run due to the randomness in selecting the training example for backpropagation.

Pros: SGD is much more faster than BGD since it only needs to propagate a single example. It is much more memory efficient because the model has to hold lesser number of examples during training. Also, the randomness in SGD helps to avoid local minima because the gradient will jump erratically all around instead of descending smoothly.

Cons: SGD might jump around too much and become unstable with oscillating optimisation paths. There is no guarantee that SGD will arrive at the global minimum. It can still end up in a local one. Each run might result in a different local minima. SGD is very sensitive to feature scaling because it only selects one example to update each time.

2)b) Overfitting means train > test accuracy (much higher). Maybe NN is too complex. It starts to learn the noise. Or maybe it memorises the training data.

By reducing the number of units, the neural network has lesser parameters to train. The model capacity is reduced, preventing it from memorising the data. This is good for limited amount of data.

By reducing the number of units, the network has lesser layers to backpropagate gradient. This prevent the network from capturing noise.

6a) Done

b) Done

c) Generalise mean that the network can effectively capture the true patterns (equation) of the data and ignoring the noise. The network can correctly classify or regress many unseen input data. Essentially it means having a high training and testing accuracy e.g. 90% and 85%.

d) A single neuron has the equation of z = wx + b, which is essentially a linear equation. Without a non-linear activation function that transforms the output z into something else, the single neuron can only fit a linear equation, and therefore learn only linear separable problems.

# Ensembles

4)a)
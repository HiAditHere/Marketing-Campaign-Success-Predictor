# Marketing Campaign Success Predictor

Dataset Link: https://archive.ics.uci.edu/ml/datasets/bank+marketing

## About the Dataset

A portuguese bank wants its client to subscribe to a product which is a term deposit. The dataset comprises of client data, data related to previous contact in the current campaign, data of social - economic conditions  at that moment and some other data. The bank would like to know which clients are more likely to subscribe to the term deposit.

We have picked the from UCI repository. The dataset consists of 
- 41188 row
- 20 features and one target variable. 10 features are categorical and 10 are numeric in nature.

## EDA and Data Engineering

#### 1) Categorical Features

There are 10 columns which are categorical in nature. There is one additional numeric column which can be considered as _pseudo-numeric_.

Firstly, we have conducted the _**chi2 test**_ on the categorical features . The chi-square test is a statistical hypothesis test that is used to analyze categorical data. It helps to determine whether there is a _**significant association**_ between two categorical variables or not. The test compares the observed frequencies of a data set to the frequencies that would be expected if there was no association between the two variables.

The chi-square test in python also returns us p-values. The p-value is a statistical measure that helps to determine the significance of the results of a hypothesis test. It is the probability of obtaining a test statistic as extreme or more extreme than the observed value, assuming the null hypothesis is true.

Below are the results of the p-value test

![image](https://user-images.githubusercontent.com/39706219/235393044-1479c166-a8ce-4bb2-8e13-500c9d6a9ab7.png)

**Loan**, **month**, **housing** and **day_of_week** have no real association with the target variable and were therefore dropped.

##### Inferenence on some important categorical variables

##### 1.1) contact

The contact variable has high correlation with the target variable. If we see the plot, the ratio of yes:no in case of cellular is much higher compared to the ratio of yes:no for telephone contact. As one outcome has a high correlation with client subscribing to the product, this column has been _**label encoded**_.

![image](https://user-images.githubusercontent.com/39706219/235393102-8cce9b8c-dd46-4f26-98e1-fcab64cbf2eb.png)

##### 1.2) poutcome

Similar to contact, in case of outcome, the previous outcome being a success has a high correlation with the client subscribing to the product. Therefore this column has also been _**label encoded**_.

![image](https://user-images.githubusercontent.com/39706219/235393195-91017bec-b488-4120-bc32-5c347d930b3d.png)

##### 1.3) pdays

![image](https://user-images.githubusercontent.com/39706219/235393248-abbb577f-f408-4fd3-9a3a-ea3abba2d1f4.png)

The box plot of pdays  doesnt give out much of information.  But the count plot is a different story.999 stands for non-existent which means that there was no previous contact with the client. The other values refers to contact those many days before. The yes:no ratio for non – existent is very low. But ratios for all other values is much greater compared to non-existent. Even though the number of not non-existent rows are low, they are significant enough to have a correlation with clients subscribing to the term of deposit. Therefore this column has been label encoded with 999 replaced with 0 and all other values replaced with 1.

![image](https://user-images.githubusercontent.com/39706219/235393283-e58911bc-d628-4abf-a2f6-5e8764e85236.png)

The remaining columns were _**One Hot Encoded**_

#### 2) Numerical Features

In case of numeric data, Pearson Correlation test has been done to find if there are any numeric columns that are highly correlated and could be removed while training the model.

![image](https://user-images.githubusercontent.com/39706219/235393342-b6d43867-f5c7-4e12-a774-1e00f0c17f91.png)

We can see that columns **nr.employed**, **euribor3m** and **emp.var.rate** are highly correlated. Therefore we took a decision to omit **emp.var.rate** while training them model.

The remaining columns were _**normalized**_

#### 3) Data Imabalance

As you can see, the dataset is highly skewed highly _**skewed**_ dataset can lead to overfitting with good results for majority class but very poor results for minority class in test data.

![image](https://user-images.githubusercontent.com/39706219/235393410-6fe9f3bc-f2ac-4930-b5a9-d77a99831007.png)

There are 2 approaches of dealing with skewed dataset
 1) **Under Sampling**: Under-sample the majority class while training the model. But it will lead to loss of information
 2) **SMOTE**: SMOTE (Synthetic Minority Over-sampling Technique) is a data augmentation method used to address class imbalance in machine learning problems. It works by creating synthetic samples of the minority class by interpolating new examples between existing minority class samples. This helps to balance the class distribution and improve the performance of models trained on imbalanced data. SMOTE is widely used in classification problems, especially in scenarios where the minority class is rare and difficult to capture in the
 
 **SMOTE** has been implemented in this scenario.
 
#### 4) Performance Metrics
 
As the dataset is unbalanced, **accuracy** would **NOT** be the most ideal metric performance measure. In an imbalanced dataset, there may be one or more classes with very few instances compared to the other classes. In such cases, accuracy can be a misleading metric because it can be high even if the model performs poorly on the minority class(es).

**F1 score** is a weighted average of **precision** and **recall**, where **precision** is the proportion of true positives among all positive predictions and **recall** is the proportion of true positives among all actual positive cases. Since F1 score takes into account both precision and recall, it can be a good metric for evaluating the performance of a classifier on imbalanced datasets, as it provides a balanced measure of performance across all classes.

## Training methods

#### 1) Hold Out Set 
Hold out set, also known as validation set or development set, is a subset of a dataset that is held back from the model training process and is used for evaluating the performance of the trained model. The purpose of the holdout set is to provide an unbiased estimate of the model's performance on new, unseen data. Typically, the holdout set is randomly sampled from the original dataset, and the remaining data is used for training the model. Once the model is trained on the training set, it is evaluated on the holdout set to estimate its generalization performance. 

#### 2) K Fold Cross Validation
K-fold cross-validation is a technique used to evaluate the performance of a machine learning model. It involves dividing the dataset into k equal-sized subsets or folds. The model is then trained and tested k times, with each fold serving as the testing set once and the remaining folds serving as the training set. The performance metrics for each fold are averaged to obtain an estimate of the model's generalization performance. K-fold cross-validation helps to reduce the variance in the estimate of the model's performance and provides a more reliable evaluation than using a single training/testing split.


## Algorithms

### Logistic Regression

Logistic regression is a good baseline model for the given classification problem, which aims to predict whether a customer will subscribe to a term deposit or not. One of the key advantages of logistic regression is its ability to handle both categorical and continuous input features. It is a statistical model that can estimate the probability of a binary outcome based on a set of input variables, and it can be used to identify the most important features that influence the outcome. Moreover, logistic regression is computationally efficient and can handle large datasets with ease, making it suitable for this particular classification problem, where we have 20 variables before feature engineering.

The results for LR are given below

| | Precision | Recall| F1-Score|
|-|-----------|-------|---------| 
|Hold Out Set| 0.49 | 0.72 |0.59 |
|K Fold Cross Validation|  0.40 | 0.87 | 0.54 |

### ANN (Artificial Neural Network)

Artificial Neural Networks (ANNs) can be a good model for the given classification problem. One of the key advantages of ANNs is their ability to learn complex relationships between input features and output variables, which can be difficult to capture using traditional statistical models such as logistic regression. ANNs use multiple layers of interconnected nodes (neurons) to process and transform the input features into higher-level representations that are more informative for predicting the output variable. Furthermore, ANNs can incorporate various regularization techniques such as dropout, early stopping, and weight decay to prevent overfitting and improve generalization performance. This is particularly useful for classification problems with high-dimensional input data, such as the given dataset.

* **Autoencoders**: Autoencoder is a neural network architecture that can be used for dimensionality reduction. It consists of an encoder and a decoder that work together to learn a compressed representation of the input data. The encoder maps the high-dimensional input data to a lower-dimensional latent space representation, while the decoder maps the latent space representation back to the original input space. By training the autoencoder to minimize the reconstruction error between the input and the output, the autoencoder can learn a compressed representation of the input data that captures the most important features of the data. This compressed representation can be used for tasks such as visualization, clustering, or classification.

Autoencoders have been implemented in this scenario.

The dataset is run through a _**for**_ loop of **i** = 1 to 37 (total number of features post feature engineering) and every iteration, the dataset is autoencoded to **i** features and then **F1-score** is calculated at every step. Ideally, loss should be considered but the loss values observed very small change, therfore F1 score was considered. The following results were obtained

![image](https://user-images.githubusercontent.com/39706219/235393462-75fee7b6-04ea-426a-92b9-ed3b3cbd8cea.png)

F1 score for **16 transformed features** came out to be the highest. Thus we have trained the ANN with original 37 features and then on 16 transformed features. Finally we used both K fold and Hold out set to test the results.

|Dimensions|Hold Out Set| K Fold Cross Validation|
|-|-|-|
|16 | 0.709 | 0.573 |
|37 |0.557 | 0.589 |

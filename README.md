Final Year Project: Intrusion Detection System



•	Introduction

Machine learning is a sub-domain of computer science which evolved from the study of pattern recognition in data, and from the computational learning theory in artificial intelligence. It is the first-class ticket to the most interesting careers in data analytics today. As data sources proliferate along with the computing power to process them, going straight to the data is one of the most straightforward ways to quickly gain insights and make predictions. Machine Learning can be thought of as the study of a list of sub-problems: decision making, clustering, classification, forecasting, deep-learning, inductive logic programming, support vector machines, reinforcement learning, similarity and metric learning, genetic algorithms, sparse dictionary learning, etc. Supervised learning, or classification is the machine learning task of inferring a function from labeled data. In Supervised learning, we have a training set, and a test set. The training and test set consists of a set of examples consisting of input and output vectors, and the goal of the supervised learning algorithm is to infer a function that maps the input vector to the output vector with minimal error. In an optimal scenario, a model trained on a set of examples will classify an unseen example in a correct fashion, which requires the model to generalize from the training set in a reasonable way. In layman’s terms, supervised learning can be termed as the process of concept learning, where a brain is exposed to a set of inputs and result vectors and the brain learns the concept that relates said inputs to outputs. A wide array of supervised machine learning algorithms is available to the machine learning enthusiast, for example Neural Networks, Decision Trees, Support Vector Machines, Random Forest, Naïve Bayes Classifier…

We live in a world where everything in our lives is becoming virtualized, starting from our personal perspective, and going all the way into the corporate point of view, everything is counting on data. However, this majestic transformation in our way of living does not come without costs, as this evolutionary transformation has paved the way for criminals and bad people to find creative ways to fulfil their goals, that’s why the need for counter measures became necessary.

With the rise of cyber security threats and attacks in today’s digital world, the need for some well tested intrusion detection system became a must. To further explain the work of an IDS, it’s enough to say that this system acts as a line of defense in the face of incoming malicious activities. When deployed on a network for example, the IDS excels in monitoring, identifying, and filtering unwanted malicious anomalies.

The objective of the research is to put to the test the effectiveness of a list of machine learning algorithms commonly used in these systems. The list of algorithms consists of Random Forest, logistic Regression, Kth-nearest neighbors, Support Vector Machines, XGBoost, Stochastic Gradient Boost, AdaBoost, and gradient Boosting Machines.

Before being put to the test, the talked about dataset was submitted to preprocessing that includes dropping missing values, removing duplicates, and binary classifications. All that was done to ensure that the dataset is ready to be tested by the algorithm.

The results of this research will help in shaping up the trust of the security communities in the previously mentioned algorithms and will help in attracting an audience to participate in such studies to further improve the field of security, especially in our local area and Lebanon.

 Now to present the structure of this paper, we will follow this introduction by a literature review to give the reader a perspective regarding the context of intrusion detection systems, the used algorithms, and the reason for me to present this particular project as my final year project.



•	Literature Review

Since the creation and implementation of the first ever intrusion detection system called Network Security Network by the University of California in 1989, IDS have played a crucial role in protecting and safeguarding the networks from unwanted and harmful activities.

A various sum of algorithms has been developed and used in machine learning in order to fulfil the tasks of an intrusion detection system. These algorithms were tested in the project in order to check conclusively their accuracy in catching unwanted anomalies.
The first used algorithm is the random forest classifier. This particular algorithm work by fitting a number of decision trees and combining them, then use their predictions to compute the average to improve the accuracy and control over-fitting.

Logistic Regression is basically based on the application of the Sigmoid Formula in order to transform a linear function into a probability between 0 and 1.

Then we have the Kth-nearest neighbors’ classification, which is a non-parametric method that uses the principles of proximity to make predictions related to a specific dataset.

After that we used the Support Vector Machines Classifier which is one of the most effective algorithms in high dimensional spaces. It uses a set of training points called support vectors in order to make predictions.

XGBoost is also one the used algorithms in this research, an it stands for Extreme Gradient Boosting and it falls in the category of gradient boosting methods, that also includes Stochastic Gradient Boost, AdaBoost, and Gradient boosting machines which were all used in this research project.




•	Methodology
To correctly test a dataset, it is important to make sure that the actual data is valid. This point takes us into the first step of our project which is data reading and preprocessing. Putting things into perspective, the dataset was downloaded into google colab from Kaggle, zipped into a folder, and then read by a variable that acted as a storage for this data, and was used later on in the testing. In the preprocessing phase, the dataset went through several processes before being deemed as eligible for testing. The first thing we did was calculating the skewness of the data info, then we carried on by removing duplicates, dropping missing values, and replacing infinity with NaN. After that we enter the binary classification phase, which is basically classifying the 3-danger level of the dataset (benign – outlier – malicious) and giving them binary values (0 – 1 – 2 respectively). Of course, it all came along a couple of plots, pie charts and tables to check for any data loss during reading and preprocessing. 
After reading and preprocessing, it came down to creating a train/test split. The training set is used to fit the data based on each used algorithm, while the testing phase evaluates accuracy based on metrics like the F1 score, precision, accuracy and support.
Then we go into applying a neural network, which is used in classifications and accuracy predictions. Note that this specific process was chosen due to its ability to process and learn complex patterns in data.
Now for the choice of the classification algorithms, we are going to start with the random forest classifier, which is basically the use of several decision trees, combining them, and then computing the average predictions in order to come up with an accuracy prediction for the specific dataset. Then we used the logistic regression which is based on the application of the Sigmoid Formula σ(z) in order to transform a linear function into a probability between 0 and 1. After that we have the Kth-nearest neighbors’ classification, which is a non-parametric method that uses the principles of proximity to make predictions related to a specific dataset. After that we used the Support Vector Machines Classifier which is one of the most effective algorithms in high dimensional spaces. It uses a set of training points called support vectors in order to make predictions. Finally, We used XGBoost that stands for Extreme Gradient Boosting and it falls in the category of gradient boosting methods, that also includes Stochastic Gradient Boost, AdaBoost, and Gradient boosting machines which were all used in this research project.
Now the performance of the tested algorithms was judged based on a set of testing metrics that consist of, but not limited to, accuracy (Measures the overall classification accuracy of the models in distinguishing normal and anomalous network traffic), F1-Score (Harmonic mean of precision and recall, providing a balanced measure of model performance), precision (Quantify the model's ability to correctly identify positive instances while minimizing false positives and false negatives), and support.
For the tools used in this project, the need for an efficient programming language was a must in order to deal with the complex requirements of the machine learning world. Thus, Python was tasked with the job, along with a bunch of its libraries like scikit-learn, TensorFlow, pandas, and matplotlib for data manipulation, algorithm implementation, and result visualization.


•	System Design and implementation


This project consists of multiple phases that were discussed in the previous sections. It starts with the reading and preprocessing phase, then went on to the train/test split phase, right before applying the accuracy algorithms.
Ensuring the validity of the real data is crucial for accurate dataset testing. At this stage, we enter the data reading and preprocessing phase of our research. To put things in perspective, the dataset was extracted from Kaggle and uploaded to Google Colab. It was then compressed into a folder and read by a variable that served as a storage location for the data and was utilized in the testing. The dataset underwent a number of steps in the preprocessing stage before being approved for testing. We started by figuring out how skewed the data was, and then we went on to eliminate duplicates, eliminate missing values, and swap out infinity for NaN. Next, we proceed to the binary categorization. It was time to create a train/test split after reading and preprocessing. While the testing phase assesses accuracy based on criteria like the F1 score, precision, accuracy, and support, the training set is used to fit the data based on each utilized method. In order to choose the classification algorithms, we are going to start with a random forest classifier which is basically using several decision trees and combining them in order to come up with an average prediction of accuracy for one particular dataset. Then we used the logistic regression which is based on the application of the Sigmoid Formula σ(z) in order to transform a linear function into a probability between 0 and 1. Then we've got the Kth-Nearest Neighbor classification, which is a nonparametric method that uses the principles of proximity to predict a specific dataset. Then we used the Support Vector Machines Classifier, which is one of the most effective algorithms for high dimensional spaces. To make predictions, it will use a set of training points known as support vectors. To make predictions, it will use a set of training points known as support vectors. 


•	Results

The outcome of this project positively exceeded my personal expectations considering the fact that I am new to the machine learning field. The predicted accuracy of each algorithm was as follows : 

-	Neural Network : 0.90
-	Random Forest : 0.96
-	Logistic Regression : 0.85
-	Kth – nearest : 0.97
-	Support vector machines : 0.96
-	XGBoost : 0.98
-	Stochastic gradient boost : 0.98
-	AdaBoost : 0.94
-	Gradient Boosting : 0.97



•	Conclusion
This study followed the applied algorithms and their predictions, including Random Forest, Logistic Regression, k-Nearest Neighbors, Support Vector Machines, Gradient Boosting Machines, Stochastic Gradient Boosting, and AdaBoost.
Random Forest emerged as a standout with high accuracy and adaptability to complex decision boundaries, while ensemble techniques like Gradient Boosting and AdaBoost significantly boosted overall model accuracy.

Overall, this research advances the field by providing practical insights into building more robust intrusion detection systems, thereby bolstering cybersecurity practices and threat mitigation strategies.



•	Future work

The accomplishment of this project marks the first steppingstone of my career, as I intend to turn this project into a lifelong work in process. I will be aiming next into integrating this particular project into an ai based companion that can be used on networks and can catch malicious activities using advanced technologies.



 
	




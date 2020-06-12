# StakeoverFlow-Tag-Prediction
Predict tags on StackOverflow with linear models.

In this notebook we will predict tags for posts from StackOverflow. To solve this task you will use multilabel classification approach.

**Libraries**
In this task you will need the following libraries:

**Numpy** — a package for scientific computing.
**Pandas** — a library providing high-performance, easy-to-use data structures and data analysis tools for the Python
**scikit-learn** — a tool for data mining and data analysis.
**NLTK** — a platform to work with natural language.

Now Upload the data use any of the source -
In this notebook I have taken **kaggle StakeOverflow** Dataset

**Text Processing**
For this we will need to use a list of stop words. It can be downloaded from nltk.

**Transforming text to a vector**
Machine Learning algorithms work with numeric data and we cannot use the provided text data "as is". There are many ways to transform text data to numeric vectors. In this task you will try to use two of them.

Bag of words
One of the well-known approaches is a bag-of-words representation. To create this transformation, follow the steps:

Find N most popular words in train corpus and numerate them. Now we have a dictionary of the most popular words.
For each title in the corpora create a zero vector with the dimension equals to N.
For each text in the corpora iterate over words which are in the dictionary and increase by 1 the corresponding coordinate.

**TF-IDF**
The second approach extends the bag-of-words framework by taking into account total frequencies of words in the corpora. It helps to penalize too frequent words and provide better features space.

Implement function tfidf_features using class TfidfVectorizer from scikit-learn. Use train corpus to train a vectorizer. Don't forget to take a look into the arguments that you can pass to it. We suggest that you filter out too rare words (occur less than in 5 titles) and too frequent words (occur more than in 90% of the titles). Also, use bigrams along with unigrams in your vocabulary.

**MultiLabel classifier**
As we have noticed before, in this task each example can have multiple tags. To deal with such kind of prediction, we need to transform labels in a binary form and the prediction will be a mask of 0s and 1s. For this purpose it is convenient to use MultiLabelBinarizer from sklearn.

Implement the function train_classifier for training a classifier. In this task we suggest to use One-vs-Rest approach, which is implemented in OneVsRestClassifier class. In this approach k classifiers (= number of tags) are trained. As a basic classifier, use LogisticRegression. It is one of the simplest methods, but often it performs good enough in text classification tasks. It might take some time, because a number of classifiers to train is large.

**Evaluation**
To evaluate the results we will use several classification metrics:

-Accuracy
-F1-score
-Area under ROC-curve
-Area under precision-recall curve

Make sure you are familiar with all of them. How would you expect the things work for the multi-label scenario? Read about micro/macro/weighted averaging following the sklearn links provided above.

**Analysis of the most important features**
Finally, it is usually a good idea to look at the features (words or n-grams) that are used with the largest weigths in your logistic regression model.

Implement the function print_words_for_tag to find them. Get back to sklearn documentation on OneVsRestClassifier and LogisticRegression if needed.

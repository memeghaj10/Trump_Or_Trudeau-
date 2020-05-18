# Trump Or Trudeau?

This is a simple project that based on the tweet-datasets of Donald Trump and Justin Trudeau predicts the real owner or author of the tweet.

This `Jupyter Notebook` uses the following libraries/functionalitites :-

1. `sklearn.feature_exrtaction.text`
	`CountVectorizer` and `TfidfVectorizer`

2. `sklearn.model_selection`
	`train_test_split`

3. `sklearn.naive_bayes`
	`MultinomialNB`

4. `sklearn.svm`
	`LinearSVC`

5. `sklearn`
	`metrics`

Since the data has been collected via the Twitter API and not split into test and training sets. We use `test_train_split()` with `random_state=53` and a `test size` of `0.33`. This will ensure we have enough test data and we'll get the same results no matter where or when we run this code.

After that, we need to create vectorized representations of the tweets in order to apply machine learning.To do so, we will utilize the `CountVectorizer` and `TfidfVectorizer` classes which we will first need to fit to the data.

Now that we have the data in vectorized form, we can train the first model. We will be using the `Multinomial Naive Bayes` model with both the `CountVectorizer` and `TfidfVectorizer` data. 

We see that the TF-IDF model performs better than the count-based approach

 A better evaluation can be made if we look at the confusion matrix, which shows the number correct and incorrect classifications based on each class. We can use the metrics, True Positives, False Positives, False Negatives, and True Negatives, to determine how well the model performed on a given class. 

 The `LinearSVC` model is even better than the Multinomial Bayesian one. Via the `confusion matrix` we can see that, although there is still some confusion where Trudeau's tweets are classified as Trump's, the False Positive rate is better than the previous model.

 Using the `LinearSVC Classifier` with two classes (Trump and Trudeau) we can sort the features (tokens), by their weight and see the most important tokens for both Trump and Trudeau.

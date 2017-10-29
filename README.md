# Spooky Authors - Kaggle Playground Competition

[Link to Challenge](https://www.kaggle.com/c/spooky-author-identification)

Team name: DSKT

Goal is to minimise the Kaggle Evaluation score.

- `bag_of_words_baseline.ipynb` follows [sklearn's tutorial for Working with Text data](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html). It uses a simple bag of words feature space and trains a Logistic Regression, Multinomial Naive Bayes, and Stochastic Gradient Descent SVM. No hyperparameters have been tuned. The models are evaluated using a 10-fold stratified Cross Validation. Multinomial Naive Bayes fares best out of the box.

	```
	Average results of 10-fold stratified CV
	
	Precision: 0.8454029412621074
	Recall:    0.8449165473497455
	Macro f1:  0.8444483672019389
	
	Kaggle Evaluation score: 0.47941
	
	```
	
- `doc2vec.ipynb` trains a doc2vec model on the labeled training data using `gensim`. The approach goes as follows: 
For each test sentence, create a doc2vec vector and find the most similar sentence vector in the model. Use the author label of that most similar sentence as the prediction.

	```
	Average accuracy out of 50 sentences is 0.82
	
	Kaggle evaluation score: 8.72731
	
	```
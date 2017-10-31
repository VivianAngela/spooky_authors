# Spooky Authors - Kaggle Playground Competition

[Link to Challenge](https://www.kaggle.com/c/spooky-author-identification)

Team name: DSKT

Goal is to minimise the Kaggle Evaluation score.


- `bag_of_words_stats_ner.ipynb` uses a Bag of Words, text statistics (text length and sentence count) and Named Entity Recognition as features. It uses a Multinomial Naive Bayes model.

```
Classification report for 80/20 train/test split

             precision    recall  f1-score   support

        EAP       0.83      0.86      0.84      1507
        HPL       0.85      0.86      0.86      1055
        MWS       0.87      0.82      0.85      1354

avg / total       0.85      0.85      0.85      3916

Kaggle Evaluation Score: 0.46454
```


- `bag_of_words_baseline.ipynb` follows [sklearn's tutorial for Working with Text data](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html). It uses a simple bag of words feature space and trains a Logistic Regression, Multinomial Naive Bayes, and Stochastic Gradient Descent SVM. No hyperparameters have been tuned. The models are evaluated using a 10-fold stratified Cross Validation. Multinomial Naive Bayes fares best out of the box.

	```
	Average results of 10-fold stratified CV
	
	Precision: 0.8454029412621074
	Recall:    0.8449165473497455
	Macro f1:  0.8444483672019389
	
	Kaggle Evaluation score: 0.47941
	
	```
- `bag_of_words_naive_bayes_spacy.ipynb` uses above Multinomial Naive Bayes estimator with a Spacy tokeniser. Alpha=0.05.

	```
	Average results of 10-fold stratified CV

	Precision: 0.8453698709260064
	Recall:    0.8482562332086235
	Macro f1:  0.8462804690851018
	
	Kaggle Evaluation score: 0.53489

	```

- `doc2vec.ipynb` trains a doc2vec model on the labeled training data using `gensim`. The approach goes as follows: 
For each test sentence, create a doc2vec vector and find the most similar sentence vector in the model. Use the author label of that most similar sentence as the prediction.

	```
	Average accuracy out of 50 sentences is 0.82
	
	Kaggle evaluation score: 8.72731
	
	```
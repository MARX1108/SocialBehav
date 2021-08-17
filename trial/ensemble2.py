"""
filename: ensemble_brain_agg*.py
author: Tai-Jung Chen
-----------------------------
Model: Decision tree x Logistic Regression
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt


def main():
	# data pre-processing
	env_data = pd.read_csv('results/result_75.csv')
	brain_data = pd.read_csv('results/brain_result_75.csv')
	merge_data = pd.merge(env_data, brain_data, on='subjectkey')
	#merge_data.to_csv('merge_processed.csv')
	print(merge_data[merge_data.columns[32]])

	# train level 0
	env_predict = env_pred(merge_data)
	brain_predict = brain_pred(merge_data)
	print('len(env_predict)', len(env_predict))
	print('len(brain_predict)', len(brain_predict))


	# train test split
	X = np.column_stack((env_predict, brain_predict))
	y = np.array(merge_data['aggressive_sumscore_x'])


	# train level 1
	X_train, X_test, y_train, y_test = train_test_split(X, y)
	model = LogisticRegression()
	scores = cross_val_score(model, X, y, cv=10)
	print('accuracy_history: ', scores)
	model.fit(X_train, y_train)

	folds = StratifiedKFold(n_splits=10)
	clf_history = []
	ensemble_history = []
	train_test_history = []
	for train_index, test_index in folds.split(X, y):
		X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
		model.fit(X_train, y_train)
		ensemble_history.append(model)
		clf = tree.DecisionTreeClassifier()
		clf.fit(X_train, y_train)
		clf_history.append(clf)
		train_test_history.append((X_train, X_test, y_train, y_test))


	min_error = max(scores)
	best_ensemble = ensemble_history[0]
	validation_set = train_test_history[0]    # X_train, X_test, y_train, y_test
	for i in range(len(scores)):
		if scores[i] == min_error:
			print('best iteration: ', i)
			best_ensemble = ensemble_history[i]
			validation_set = train_test_history[i]

	y_pred = best_ensemble.predict(X_test)

	# evaluation
	print('Confusion Matrix\n', confusion_matrix(validation_set[3], y_pred))
	print('accuracy: ', accuracy_score(validation_set[3], y_pred))
	print('f1 score: ', f1_score(validation_set[3], y_pred))
	print('recall: ', recall_score(validation_set[3], y_pred))
	print('precision: ', precision_score(validation_set[3], y_pred))
	print('kappa: ', cohen_kappa_score(validation_set[3], y_pred))


def env_pred(data):
	"""

	:param data:(df): data set
	:return pred:(lst): prediction
	"""
	# load data
	agg_data = data

	# train test split and train
	X = np.array(agg_data[agg_data.columns[3:29]])
	y = np.array(agg_data['aggressive_sumscore_x'])
	folds = StratifiedKFold(n_splits=10)
	clf_history = []
	train_test_history = []
	for train_index, test_index in folds.split(X, y):
		X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
		clf = tree.DecisionTreeClassifier()
		clf.fit(X_train, y_train)
		clf_history.append(clf)
		train_test_history.append((X_train, X_test, y_train, y_test))

	scores = cross_val_score(tree.DecisionTreeClassifier(), X, y, cv=10)
	print('accuracy_history: ', scores)
	min_error = max(scores)
	best_clf = clf_history[0]
	validation_set = train_test_history[0]    # X_train, X_test, y_train, y_test
	for i in range(len(scores)):
		if scores[i] == min_error:
			print('best iteration: ', i)
			best_clf = clf_history[i]
			validation_set = train_test_history[i]

	y_pred = best_clf.predict(X)
	return y_pred


def brain_pred(data):
	"""

	:param data:(df): data set
	:return pred:(lst): prediction
	"""
	# load data
	agg_data = data

	X = np.array(agg_data[agg_data.columns[31:]])
	y = np.array(agg_data['aggressive_sumscore_y'])
	folds = StratifiedKFold(n_splits=10)
	clf_history = []
	train_test_history = []
	for train_index, test_index in folds.split(X, y):
		X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
		clf = tree.DecisionTreeClassifier()
		clf.fit(X_train, y_train)
		clf_history.append(clf)
		train_test_history.append((X_train, X_test, y_train, y_test))

	scores = cross_val_score(tree.DecisionTreeClassifier(), X, y, cv=10)
	print('accuracy_history: ', scores)
	min_error = max(scores)
	best_clf = clf_history[0]
	validation_set = train_test_history[0]    # X_train, X_test, y_train, y_test
	for i in range(len(scores)):
		if scores[i] == min_error:
			print('best iteration: ', i)
			best_clf = clf_history[i]
			validation_set = train_test_history[i]

	y_pred = best_clf.predict(X)
	return y_pred


if __name__ == '__main__':
	main()
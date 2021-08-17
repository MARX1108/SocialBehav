"""
filename: aggressive_sumscore_decision_tree_classification
author: Tai-Jung Chen
-----------------------------
label = 0; if 0 <= prosocial_parent <= 3
label = 1; if prosocial_parent = 6
Model: Decision tree
"""


import pandas as pd
import random
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt


def main():
	# data pre-processing
	parent_data = data_preprocess()
	# parent_data.to_csv('result_prosocial_parent.csv')
	parent_data = parent_data.sample(frac=1)
	parent_data = np.split(parent_data, [6955], axis=0)
	parent_data_testing = parent_data[1]
	parent_data_training = parent_data[0]

	# bootstrapping
	parent_data_training0 = parent_data_training[parent_data_training.prosocial_parent != 1]
	parent_data_training1 = parent_data_training[parent_data_training.prosocial_parent != 0]

	X_test = np.array(parent_data_testing[parent_data_testing.columns[2:]])
	y_test = np.array(parent_data_testing['prosocial_parent'])
	X_train_0 = np.array(parent_data_training0[parent_data_training0.columns[2:]])
	X_train_1 = np.array(parent_data_training1[parent_data_training1.columns[2:]])
	y0 = np.array(parent_data_training0['prosocial_parent'])
	y1 = np.array(parent_data_training1['prosocial_parent'])
	print('X_train0 shape: ', X_train_0.shape)
	print('X_train1 shape: ', X_train_1.shape)
	print('y0 shape: ', y0.shape)
	print('y1 shape: ', y1.shape)
	X_train_1 = X_train_1[:X_train_0.shape[0]]
	y0 = y0.reshape(y0.shape[0], 1)
	y1 = y1.reshape(y1.shape[0], 1)
	y1 = y1[:y0.shape[0]]
	X_train = np.concatenate((X_train_0, X_train_1), axis=0)
	y_train = np.concatenate((y0, y1), axis=0)
	print('X_train shape: ', X_train.shape)
	print('y_train shape: ', y_train.shape)

	# modeling
	clf = tree.DecisionTreeClassifier()
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print('y_prediction: ', y_pred)


	"""
	# train test split and train
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

	y_pred = best_clf.predict(validation_set[1])
	"""

	# evaluation
	print('Confusion Matrix\n', confusion_matrix(y_test, y_pred))
	print('accuracy: ', accuracy_score(y_test, y_pred))
	print('f1 score: ', f1_score(y_test, y_pred))
	print('recall: ', recall_score(y_test, y_pred))
	print('precision: ', precision_score(y_test, y_pred))
	print('kappa: ', cohen_kappa_score(y_test, y_pred))

	print('tree depth: ', clf.tree_.max_depth)
	"""

	# graph
	tree.plot_tree(best_clf, max_depth=3, fontsize=5)
	plt.show()
	text_representation = tree.export_text(best_clf)
	# print(text_representation)
	with open("decistion_tree.log", "w") as fout:
		fout.write(text_representation)
	"""

def data_preprocess():
	"""
	:return .csv:  processed data
	"""
	# import data
	data = pd.read_csv("data/alldata.csv")

	# peek data
	print('initial data frame: ')
	print(data.head())
	print(data.tail())
	print('data.shape: ', data.shape)

	# drop label 2 and label 3
	data = data.drop(['prosocial_child', 'aggressive_sumscore'], axis=1)
	# drop redundant attribute
	data = data.drop(['asr_scr_perstr_t', 'asr_scr_somaticpr_t', 'asr_scr_inattention_t', 'crpbi_bothcare', 'kbi_p_c_best_friend', 'kbi_p_c_reg_friend_group', 'macv_p_ss_fs', 'macv_p_ss_fo', 'macv_p_ss_isr'], axis=1)
	data = data.drop(['macv_p_ss_fr', 'macv_p_ss_r', 'demo_prnt_age_v2', 'demo_prnt_marital_v2', 'demo_comb_income_v2', 'demo_yrs_1', 'demo_yrs_2', 'parent_rules_q1', 'parent_rules_q4', 'parent_rules_q7'], axis=1)
	data = data.drop(['su_risk_p_1', 'su_risk_p_2_3', 'su_risk_p_4_5', 'interview_age', 'interview_date', 'sex'], axis=1)
	# drop nan
	data = data.dropna()
	# ..print(data.isnull().sum())
	# drop meaningless value (-1: not acceptable), (3: not sure)
	data = data[data.kbi_p_conflict != -1.0]
	data = data[data.kbi_p_c_mh_sa != 3.0]
	# ..print(data['kbi_p_conflict'].value_counts())
	# ..print(data['kbi_p_c_mh_sa'].value_counts())

	# data description
	print('--------------------------------------------------------')
	print('<describe>\n', data['prosocial_parent'].describe())

	# classify labels
	data['prosocial_parent'] = data['prosocial_parent'].replace([0, 1, 2, 3], 0)
	data['prosocial_parent'] = data['prosocial_parent'].replace([6], 1)
	data = data[data.prosocial_parent != 4]
	data = data[data.prosocial_parent != 5]

	print('------------------------------------------')
	print('value count: ')
	print(data['prosocial_parent'].value_counts())

	# peek data
	print('-------------------------------------------')
	print('processed data frame')
	print(data.head())
	print(data.tail())
	print('data.shape: ', data.shape)

	return data


if __name__ == '__main__':
	main()
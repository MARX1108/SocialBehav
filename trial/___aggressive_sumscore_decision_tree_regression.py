"""
filename: aggressive_sumscore_decision_tree_regression
author: Tai-Jung Chen
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


def main():
	# data pre-processing
	agg_data = data_preprocess()
	print(agg_data.shape)

	# train test split
	# print('test', agg_data[agg_data.columns[2:]])
	X = agg_data[agg_data.columns[2:]]
	y = agg_data['aggressive_sumscore']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	# training
	clf = tree.DecisionTreeRegressor()
	clf.fit(X_train, y_train)
	X_prediction = clf.predict(X_test)
	print('prediction: ', X_prediction)

	# MSE
	print(mean_squared_error(y_test, X_prediction))


def data_preprocess():
	# import data
	data = pd.read_csv("data/alldata.csv")

	# peek data
	print(data.head())
	print(data.tail())
	print(data.shape)

	# drop label 2 and label 3
	data = data.drop(['prosocial_child', 'prosocial_parent'], axis=1)
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

	# peek data
	print(data.head())
	print(data.tail())
	print(data.shape)

	return data


if __name__ == '__main__':
	main()
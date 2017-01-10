# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-09-28'
    Email:   '383287471@qq.com'
    Describe: RF(CNN(W2V)),直接使用CNN(w2v)特征做RF输入，特征已经保存在文件中
"""
from __future__ import print_function
import numpy as np

# ****************************************************************
# +++++++++++++ region start : 参数设置 +++++++++++++
# ****************************************************************

print('=' * 30)
config = {
    'dataset_type': 'v2.3(Sa)',
    'label_version': 'v2.0',
    'verbose': 1,
}
word2vec_to_solve_oov = False
feature_type = 'word'
seed = 64003
num_of_filters = 110
merge_feature = True
standard_scaler = True
print('num_of_filters:%d,merge_feature:%s,standard_scaler:%s'%(num_of_filters,merge_feature,standard_scaler))

estimator_paramter_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
# estimator_paramter_list = [2000]
print('word2vec_to_solve_oov:%s\nrand_seed:%s\nfeature_type:%s' % (word2vec_to_solve_oov, seed,
                                                                   feature_type))
print('树：%s' % estimator_paramter_list)
print('=' * 30)

# ****************************************************************
# ------------- region end : 参数设置 -------------
# ****************************************************************

import pickle

middle_layer_output_file = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/bow_model/bow_WORD2VEC_oov_randomforest/IALP2016_experiment/conv_features/conv_middle_output_%dfilters.pkl'%(num_of_filters)

with open(middle_layer_output_file, 'r') as fout:
    cv_data_tmp = pickle.load(fout)
    middle_output_dev = pickle.load(fout)
    middle_output_val = pickle.load(fout)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=True, with_std=True)

cv_data = []
for (flag, dev_X, dev_y, val_X, val_y, feature_encoder),middle_dev,middle_val in zip(cv_data_tmp,middle_output_dev,middle_output_val):
    if standard_scaler:
        dev_X = scaler.fit_transform(dev_X)
        val_X = scaler.transform(val_X)
    if merge_feature:
        merge_features_dev = np.concatenate((middle_dev[4], dev_X), axis=1)
        merge_features_val = np.concatenate((middle_val[4], val_X), axis=1)
    else:
        merge_features_dev = middle_dev[4]
        merge_features_val = middle_val[4]

    cv_data.append([flag, merge_features_dev, dev_y, merge_features_val, val_y, feature_encoder])


# region -------------- cross validation -------------
if config['verbose'] > 0:
    print('-' * 20)
    print('cross validation')

from traditional_classify.bow_rf.bow_rf_model import BowRandomForest

BowRandomForest.cross_validation(
    train_data=None,
    test_data=None,
    cv_data=cv_data,
    shuffle_data=True,
    n_estimators_list=estimator_paramter_list,
    # feature_type=feature_type,
    word2vec_to_solve_oov=False,
    # word2vec_model_file_path=None,
    verbose=config['verbose'],
    cv=3,
    # 直接输入
    need_transform_input=False,
    # need_segmented=False,
    need_validation=True,
    include_train_data=True,
)

if config['verbose'] > 0:
    print('-' * 20)
# endregion -------------- cross validation ---------------

# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-09-27'; 'last updated date: 2016-09-27'
    Email:   '383287471@qq.com'
    Describe: IALP paper - Dialogue Act Recognition for Chinese Out-of-Domain Utterances using Hybrid CNN-RF
        RF（BOC/BOW） 模型
"""
from __future__ import print_function

from coprocessor.Corpus.ood_dataset.stable_vesion.data_util import DataUtil

data_util = DataUtil()

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
feature_type = 'seg'
seed = 64003
estimator_paramter_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
# estimator_paramter_list = [2000]
print('word2vec_to_solve_oov:%s\nrand_seed:%s\nfeature_type:%s' % (word2vec_to_solve_oov, seed,
                                                                   feature_type))
print('树：%s' % estimator_paramter_list)
word2vec_model_file_path = data_util.transform_dataset_name('50d_weibo_100w')
print('=' * 30)

# ****************************************************************
# ------------- region end : 参数设置 -------------
# ****************************************************************

# region -------------- 加载训练数据和测试数据 -------------

train_data, test_data = data_util.load_train_test_data(config)
label_to_index, index_to_label = data_util.get_label_index(version=config['label_version'])
# train dataset X-y
train_X = train_data['SENTENCE'].as_matrix()
train_y = train_data['LABEL_INDEX'].as_matrix()
# test dataset X-y
test_X = test_data['SENTENCE'].as_matrix()
test_y = test_data['LABEL_INDEX'].as_matrix()
# endregion -------------- 加载训练数据和测试数据 ---------------

# region -------------- cross validation -------------
if config['verbose'] > 0:
    print('-' * 20)
    print('cross validation')

from traditional_classify.bow_rf.bow_rf_model import BowRandomForest

BowRandomForest.cross_validation(
    train_data=(train_X,train_y),
    test_data=(test_X,test_y),
    shuffle_data=True,
    n_estimators_list=estimator_paramter_list,
    feature_type=feature_type,
    word2vec_to_solve_oov=False,
    # word2vec_model_file_path=None,
    verbose=config['verbose'],
    cv=3,
    need_segmented=True,
    need_validation=True,
    include_train_data=True,
)

if config['verbose'] > 0:
    print('-' * 20)
# endregion -------------- cross validation ---------------

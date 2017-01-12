# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-09-28'; 'last updated date: 2016-09-28'
    Email:   '383287471@qq.com'
    Describe:
        IALP paper - Dialogue Act Recognition for Chinese Out-of-Domain Utterances using Hybrid CNN-RF
            ++++++++++++++++++++++ RF(CNN(static-W2V)) ++++++++++++++++++++++
            输入是原始数据，完整的混合模型，包括特征提取和分类
"""

from __future__ import print_function
from ood_dataset.stable_version.data_util import DataUtil
from deep_learning.cnn.wordEmbedding_cnn.example.RF_one_conv_layer_wordEmbedding_cnn import \
    RFAndRFAndWordEmbeddingCnnMerge

data_util = DataUtil()
# region +++++++++++++  1 : 参数设置 +++++++++++++

print('=' * 30)
config = {
    'dataset_type': 'v2.3(Sa)',
    'label_version': 'v2.0',
    'verbose': 0,
}
word2vec_to_solve_oov = False
feature_type = 'word'
seed = 64003
# num_filter_list = [10, 30, 50, 80, 100, 110, 150, 200, 300, 500, 1000]
# estimator_paramter_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
# 验证最高的参数
# 110 filters 和 500 trees is best validation parameter
num_filter_list = [110]
estimator_paramter_list = [500]

print('数据集：%s，标注版本：%s' % (config['dataset_type'], config['label_version']))

print('word2vec_to_solve_oov:%s\nrand_seed:%s\nfeature_type:%s' % (word2vec_to_solve_oov, seed,
                                                                   feature_type))
print('num_filter_list：%s' % num_filter_list)
# word2vec_model_file_path = data_util.transform_dataset_name('50d_weibo_100w')
print('=' * 30)
# endregion

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

input_length = 14
word_embedding_dim = 50

RFAndRFAndWordEmbeddingCnnMerge.cross_validation(
    train_data=(train_X, train_y),
    test_data=(test_X, test_y),
    # 是否要验证
    need_validation=True,
    include_train_data=True,
    vocabulary_including_test_set=True,
    cv=3,
    feature_type=feature_type,
    num_labels=24,
    input_length=input_length,
    # batch_size = 50,
    # num_filter_list=[8],
    num_filter_list=num_filter_list,
    verbose=config['verbose'],
    embedding_weight_trainable=False,
    word2vec_model_file_path=data_util.transform_word2vec_model_name('%dd_weibo_100w' % word_embedding_dim),
    shuffle_data=True,
    n_estimators_list=estimator_paramter_list,
    need_segmented=True,
)

if config['verbose'] > 0:
    print('-' * 20)
# endregion -------------- cross validation ---------------

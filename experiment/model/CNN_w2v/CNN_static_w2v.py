# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-09-27'; 'last updated date: 2017-01-10'
    Email:   '383287471@qq.com'
    Describe:
        IALP paper - Dialogue Act Recognition for Chinese Out-of-Domain Utterances using Hybrid CNN-RF
            ++++++++++++++++++++++ CNN（static-w2v-BOC/BOW） 模型 ++++++++++++++++++++++
"""
from __future__ import print_function
from ood_dataset.stable_version.data_util import DataUtil
from deep_learning.cnn.wordEmbedding_cnn.example.one_conv_layer_wordEmbedding_cnn import WordEmbeddingCNNWithOneConv

data_util = DataUtil()

# region +++++++++++++  1 : 参数设置 +++++++++++++

print('=' * 30 + ' 参数设置 ' + '=' * 30)

config = {
    'dataset_type': 'v2.3(Sa)',
    'label_version': 'v2.0',
    'verbose': 0,
}
word2vec_to_solve_oov = False
feature_type = 'word'
seed = 64003
num_filter_list = [10, 30, 50, 80, 100, 110, 150, 200, 300, 500, 1000]
# 110 filters is best validation parameter
# num_filter_list = [100]
print('数据集：%s，标注版本：%s' % (config['dataset_type'], config['label_version']))

print('word2vec_to_solve_oov:%s\nrand_seed:%s\nfeature_type:%s' % (word2vec_to_solve_oov, seed,
                                                                   feature_type))
print('num_filter_list：%s' % num_filter_list)
# word2vec_model_file_path = data_util.transform_dataset_name('50d_weibo_100w')
print('=' * 30 + ' 参数设置 ' + '=' * 30)

# endregion


# region -------------- 2 : 加载训练数据和测试数据 -------------

train_data, test_data = data_util.load_train_test_data(config)
label_to_index, index_to_label = data_util.get_label_index(version=config['label_version'])
# train dataset X-y
train_X = train_data['SENTENCE'].as_matrix()
train_y = train_data['LABEL_INDEX'].as_matrix()
# test dataset X-y
test_X = test_data['SENTENCE'].as_matrix()
test_y = test_data['LABEL_INDEX'].as_matrix()
print('=' * 30 + '数据加载完毕' + '=' * 30)

# endregion

# region -------------- cross validation -------------
if config['verbose'] > 0:
    print('-' * 20)
    print('cross validation')

input_length = 14
word_embedding_dim = 50
WordEmbeddingCNNWithOneConv.cross_validation(
    train_data=(train_X, train_y),
    test_data=(test_X, test_y),
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
    # 获取中间层输出
    get_cnn_middle_layer_output=True,
    # 保存到以下地址
    middle_layer_output_file = 'result/conv_middle_output_%dfilters.pkl'%num_filter_list[0],
    word2vec_model_file_path=data_util.transform_word2vec_model_name('%dd_weibo_100w' % word_embedding_dim)
)

if config['verbose'] > 0:
    print('-' * 20)
# endregion -------------- cross validation ---------------

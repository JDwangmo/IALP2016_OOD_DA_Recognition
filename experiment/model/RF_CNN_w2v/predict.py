# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-09-28'; 'last updated date: 2016-09-28'
    Email:   '383287471@qq.com'
    Describe:
        IALP paper - Dialogue Act Recognition for Chinese Out-of-Domain Utterances using Hybrid CNN-RF
            ++++++++++++++++++++++ RF(CNN(static-W2V)) ++++++++++++++++++++++
                模型测试
"""

from __future__ import print_function
from ood_dataset.stable_version.data_util import DataUtil
from deep_learning.cnn.wordEmbedding_cnn.example.RF_one_conv_layer_wordEmbedding_cnn import \
    RFAndWordEmbeddingCnnMerge
from data_processing_util.cross_validation_util import transform_cv_data
import os

data_util = DataUtil()

# region +++++++++++++  1 : 参数设置 +++++++++++++

print('=' * 30 + ' 参数设置 ' + '=' * 30)
config = {
    'dataset_type': 'v2.3(Sa)',
    'label_version': 'v2.0',
    'verbose': 1,
}
word2vec_to_solve_oov = False
feature_type = 'word'
seed = 64003
input_length = 14
word_embedding_dim = 50
# num_filter_list = [10, 30, 50, 80, 100, 110, 150, 200, 300, 500, 1000]
# estimator_paramter_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
# 验证最高的参数 --- 110 filters 和 500 trees is best validation parameter
num_filter_list = [110]
estimator_paramter_list = [300]

# 测试类型
# 0 - 批量检验
# 1 - 单句测试
TEST_TYPE = 0

print('数据集：%s，标注版本：%s' % (config['dataset_type'], config['label_version']))
print('句子截断长度为： %d ， 词向量维度为： %d ' % (input_length, word_embedding_dim))
print('word2vec_to_solve_oov:%s\nrand_seed:%s\nfeature_type:%s' % (word2vec_to_solve_oov, seed,
                                                                   feature_type))
print('num_filter_list：%s' % num_filter_list)
print('estimator_paramter_list：%s' % estimator_paramter_list)
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
print('=' * 30 + ' 数据加载完毕 ' + '=' * 30)
# endregion

# region -------------- 3: predict -------------
print('=' * 30 + ' 开始预测 ' + '=' * 30)

if os.path.exists('model/RF_CNN_w2v.pkl'):
    # 构建 空 模型，
    model = RFAndWordEmbeddingCnnMerge(
        None,
        num_filter=110,
        num_labels=24,
        n_estimators=500,
        word2vec_model_file_path='',
        dataset_flag=0,
        verbose=0,
        init_model=False
    )
    # 恢复模型
    model.model_from_pickle('model/RF_CNN_w2v.pkl')
    # 预测
    y_pred_result, y_pred_score = model.batch_predict_bestn(
        [
            u'你好',
            u'买手机'
        ],
        transform_input=True,
        bestn=1
    )
    print(y_pred_result, y_pred_score)

    y_pred_result = model.predict(
        '喜欢',
        transform_input=True,
    )

    print(index_to_label[y_pred_result])

else:
    # 还没模型文件，先训练并保存
    # 批量测试检验
    train_data = (train_X, train_y)
    test_data = (test_X, test_y)
    need_validation = True
    include_train_data = True
    cv = 3
    num_labels = 24
    input_length = input_length
    # batch_size = 50,
    # num_filter_list=[8],
    num_filter_list = num_filter_list
    verbose = config['verbose']
    embedding_weight_trainable = False
    word2vec_model_file_path = data_util.transform_word2vec_model_name('%dd_weibo_100w' % word_embedding_dim)
    shuffle_data = True
    need_segmented = True
    feature_encoder = RFAndWordEmbeddingCnnMerge.get_feature_encoder(
        need_segmented=need_segmented,
        input_length=input_length,
        verbose=1,
        feature_type=feature_type,
        # 设置字典保持一致
        update_dictionary=False,
        vocabulary_including_test_set=True,
    )

    # 将数据集进行特征转换
    cv_data = transform_cv_data(
        feature_encoder=feature_encoder,
        cv_data=[
            [0, train_X, train_y, test_X, test_y]
        ],
        shuffle_data=True,
        diff_train_val_feature_encoder=1,
        verbose=1,
    )[0]
    # 构建RF(CNN(static-w2v))模型
    model = RFAndWordEmbeddingCnnMerge(
        feature_encoder,
        num_filter=110,
        num_labels=24,
        n_estimators=500,
        word2vec_model_file_path=word2vec_model_file_path,
        dataset_flag=0,
        verbose=0,
        init_model=True,
    )

    print(model.fit(
        train_data=(cv_data[1], cv_data[2]),
        validation_data=(cv_data[3], cv_data[4]),
    ))

    model.save_model('model/RF_CNN_w2v.pkl')


# endregion

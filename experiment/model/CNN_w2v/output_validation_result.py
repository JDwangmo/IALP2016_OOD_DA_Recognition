# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-20'
    Email:   '383287471@qq.com'
    Describe: 验证的终端输出 数据提取 脚本的 模板，分三步分别提取 参数、测试结果、训练结果
"""

step = 2

with open('result/CNN_static-w2v_boc(word)_cv3.txt','r') as fout:
    for line in fout:
        line = line.strip()
        if step == 1:
            if line.startswith('num_filter is '):
                print(line.replace('num_filter is ','').replace('.',''))
        if step == 2:
            if line.startswith('测试结果汇总：') :
                print(line.replace('测试结果汇总：[','').replace(']',''))
        if step == 3:
            if line.startswith('验证中训练数据结果：'):
                print(line.replace('验证中训练数据结果：[','').replace(']',''))

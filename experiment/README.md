## IALP2016_OOD_DA_Recognition --- Experiment 部分
### Describe
- IALP2016_OOD_DA_Recognition --- Experiment 部分

### Project Structure
- [model](https://github.com/JDwangmo/IALP2016_OOD_DA_Recognition/tree/master/experiment/model)
    - [RF_bow](https://github.com/JDwangmo/IALP2016_OOD_DA_Recognition/tree/master/experiment/model/RF_bow)
    - [CNN_w2v](https://github.com/JDwangmo/IALP2016_OOD_DA_Recognition/tree/master/experiment/model/CNN_w2v)
        - CNN(static-w2v)
    - [RF_CNN_w2v](https://github.com/JDwangmo/IALP2016_OOD_DA_Recognition/tree/master/experiment/model/RF_CNN_w2v)
        - `RF_CNN_w2v.py`： RF(CNN(static-W2V)) 完整模型  --- 交叉验证
        - `RF_conv_feature.py`：RF(CNN(static-W2V)) 模型 --- 分步骤，先用 CNN_w2v 再 RF_bow
        - `predict.py`： RF(CNN(static-W2V)) 完整模型 -- 取验证最高的模型进行 预测
        
### Dependence lib

### User Manual
- 1 
- 2 
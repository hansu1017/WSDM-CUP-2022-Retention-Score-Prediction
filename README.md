# WSDM2022-Retention-Score-Prediction
WSDM2022留存预测挑战赛 第1名解决方案

## **1. 环境配置**

- python--3.8
- scipy==1.7.1
- pandas==1.3.2
- numpy==1.19.5
- lightgbm==2.0.0
- catboost==0.25.1
- xgboost==1.2.1
- torch==1.9.0+cu111
- deepctr-torch==0.2.7
- tensorflow-decison-forests==0.2.2


## **2. 目录结构**

```
./
├── README.md
├── codes
│   ├── data
│   ├── features
│   ├── res
│   ├── get_features.ipynb
│   ├── get_sequence.ipynb
│   ├── cnn_binary_0.ipynb
│   ├── cnn_binary_7.ipynb
│   ├── tfdf_online.ipynb
│   ├── inference.ipynb
│   ├── function.py
```


## **4. 运行流程**
- 安装环境
- 将a榜和b榜数据集解压后放到data目录下
- 依次运行get_features.ipynb，get_sequence.ipynb，cnn_binary_0.ipynb，cnn_binary_7.ipynb，
tfdf_online.ipynb， inference.ipynb，最终结果生成在在codes/res文件夹中

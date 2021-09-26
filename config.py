# -*- coding: utf-8 -*-

import os 
from enum import Enum


class Config:
    K1 = 1.6667
    B1 = 1.5333
    K2 = 3.5
    B2 = 3.28
    confidence = 0.95
    confidence_second = 0.98
    java_host_evaluate = "http://192.168.18.28:30069/model/bizModel/getModelEvaluationResult"
    java_host_train = "http://192.168.18.28:30069/modelTrain/getModelTrainResult"
    java_host_trend = "http://192.168.18.28:30069/model/evaluation/warnRecord/trendCallback"
    java_host_train_batch = ""

    redis_host = '192.168.18.13'
    redis_port = 30144
    redis_db = 1
    redis_password = 123456
    
    suddenChangeThreshold = 0.1
    trend_Threshold_times =100
    trendThreshold = 30

    QUSHI_MODELS_STATUS ={}

    EM_Threshold = 4.0
    java_host_trainEM = ""
    model_name = "model.pkl"

class Qushi(Enum):
    
    CHIXU_PING = 1
    CHIXU_SHENG = 2
    
    CHIXU_JIANG = 3
    TUBIAN_SHENG = 4
    TUBIAN_JIANG= 5 
    TUBIAN_PING = 6 

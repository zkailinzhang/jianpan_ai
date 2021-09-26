
13-05-2021     20:47:36 werkzeug:INFO:172.17.224.171 - - [13/May/2021 20:47:36] "POST /predict HTTP/1.1" 200 -
13-05-2021     20:47:36 root:INFO:******evaluate_task modelid 531,excep:int() argument must be a string, a bytes-like object or a number, not 'tuple',trace Traceback (most recent call last):
  File "/home/jp/anaconda3/envs/tf12/lib/python3.6/site-packages/redis/connection.py", line 1185, in get_connection
    connection = self._available_connections.pop()
IndexError: pop from empty list

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/jp/jianpan/jianpan/serving.py", line 312, in evaluate_task
    re.set(keys_redis,prediction_series)
  File "/home/jp/anaconda3/envs/tf12/lib/python3.6/site-packages/redis/client.py", line 1801, in set
    return self.execute_command('SET', *pieces)
  File "/home/jp/anaconda3/envs/tf12/lib/python3.6/site-packages/redis/client.py", line 898, in execute_command
    conn = self.connection or pool.get_connection(command_name, **options)
  File "/home/jp/anaconda3/envs/tf12/lib/python3.6/site-packages/redis/connection.py", line 1187, in get_connection
    connection = self.make_connection()
  File "/home/jp/anaconda3/envs/tf12/lib/python3.6/site-packages/redis/connection.py", line 1227, in make_connection
    return self.connection_class(**self.connection_kwargs)
  File "/home/jp/anaconda3/envs/tf12/lib/python3.6/site-packages/redis/connection.py", line 509, in __init__
    self.port = int(port)
TypeError: int() argument must be a string, a bytes-like object or a number, not 'tuple'



# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np 
from flask import Flask, jsonify, request
#import paramiko
import pickle
import statsmodels
from statsmodels.sandbox.regression.predstd import wls_prediction_std
#import xlrd
from matplotlib import pyplot as plt
import statsmodels.api as sm #最小二乘
from statsmodels.formula.api import ols #加载ols模型
import wget
import json
import requests
import subprocess
import logging 
from enum import Enum
import redis 
#import happybase
from concurrent.futures import ThreadPoolExecutor
import traceback
from config import Config,Qushi
import math
import time
import itertools

executor = ThreadPoolExecutor(16)

pathcwd = os.path.dirname(__file__)
app = Flask(__name__)

class States(Enum):
    
    XUNLIAN_ZHONG = 1
    XL_WANCHENG = 2
    
    PINGGU_ZHONG = 3
    PG_CX_ZHONG = 4
    PG_WANCHENG= 5 
    PG_CX_WANCHENG = 6 
    PG_QUXIAO = 7
    
    FABU = 8
    FB_QUXIAO = 9  
    
MODELS_STATUS = {}
logpath = 'log/serving.std.out'
logging.basicConfig(filename=logpath,filemode='a',format='%(asctime)s %(name)s:%(levelname)s:%(message)s',datefmt="%d-%m-%Y \
    %H:%M:%S",level=logging.DEBUG)


# # "train_version":0123,"release_verison":0123}  
#评分的线性回归

header = {'Content-Type': 'application/json','Accept': 'application/json'} 

@app.route('/predict', methods=['POST','GET'])
def predict():
    try:
        request_json = request.get_json()   
        model_id = request_json["modelId"]
        delta1 = request_json["firstConfidence"]
        delta2 = request_json["secondConfidence"]
        logging.info("******predicting modelid {},".format(model_id))
 
        if (delta1>=0.98 or delta1 <0.95 or delta2>=1.0 or delta2 <0.98): 
            logging.info("******predicting modelid {},excp:{}".format(model_id,"置信度异常"))
            return (bad_request(504)) 

        loaded_model = None
        clf = 'model.pkl'
        #if str(model_id) in request_json.keys():
        #    loaded_model = request_json[str(model_id)]
        local_path = './model/train/' + str(model_id)+'/'
        if not os.path.exists(local_path):
            logging.info("******predicting modelid {},excp:{}".format(model_id,"模型文件找不到"))
            return(bad_request(502))
        #if str(model_id) not in MODELS_MAP.keys():
        with open(local_path + clf,'rb') as f:
            loaded_model = pickle.load(f)
        
            
        #logging.info("******predicting summary {},".format(loaded_model.summary()  )) 
        params  = loaded_model.params.index
        columns = list(params[1:])
        for col in columns:
            if col not in request_json.keys():
                logging.info("******predicting modelid {},excp:{}".format(model_id,"相关测点异常"))
                return(bad_request(501))
                      

        data=[]
        for i in range(1,len(params)):
            data.append(request_json[params[i]]) 
        
        if len(data)==0 or '' in data:
            logging.info("******predicting modelid {},excp:{}".format(model_id,"数据为空"))
            return(bad_request(400))
        else:
            data =  np.expand_dims(data,1)       
            df = pd.DataFrame(dict(zip(columns,data)))
            
            df.insert(0,'const',1.0)
            df_const = df
            predictions = loaded_model.predict(df_const)
                
            paras  = loaded_model.conf_int(Config.confidence)
            d_pre = paras[0][0]
            up_pre = paras[1][0]
            
            for i in range(1,len(paras)):
                d_pre += df_const.loc[0][i] * paras[0][i] 
                up_pre += df_const.loc[0][i] * paras[1][i] 

            paras  = loaded_model.conf_int(Config.confidence_second)
            d_pre2 = paras[0][0]
            up_pre2 = paras[1][0]
            
            for i in range(1,len(paras)):
                d_pre2 += df_const.loc[0][i] * paras[0][i] 
                up_pre2 += df_const.loc[0][i] * paras[1][i] 
            up_pre += up_pre* (Config.K1*delta1-Config.B1)
            up_pre2 += up_pre2 *(Config.K2*delta2-Config.B2)
            d_pre -=d_pre* (Config.K1*delta1-Config.B1)
            d_pre2-=d_pre2*(Config.K2*delta2-Config.B2)
            pred_interval = {"prediction":predictions.loc[0],"first_upper":up_pre,"first_lower":d_pre,"second_upper":up_pre2,"second_lower":d_pre2}
                
            message = {
                'status': True,
                'message': "请求成功",
                'data':pred_interval
            }
            logging.info("******predicting finished modelid {},".format(model_id))
            responses = jsonify(message)
            
            responses.status_code = 200
    except Exception as e:
        logging.info("******predicting modelid {},excp:{}".format(model_id,e))
        raise e 
    
    return (responses)



@app.errorhandler(400)
def bad_request(error=400):
    
    message = {
        'status': False,
		'message': '',
	    'data':'',
        }
    if error == 400:
        message.update( {
			'message': '-->数据为空错误，请检查相关测点数据',
	})
        resp = jsonify(message)
        resp.status_code = 400
    elif error ==501:
        message.update( {
			'message': '-->请求参数不一致错误，请检查相关测点、模型ID、请求接口',
	})
        resp = jsonify(message)
        resp.status_code = 501
    elif error ==502:
        message.update( {
			'message': '-->模型文件找不到',
	})
        resp = jsonify(message)
        resp.status_code =502
    elif error ==503:
        message.update( {
			'message': '-->模型未发布',
	})
        resp = jsonify(message)
        resp.status_code =503
    elif error ==504:
        message.update( {
			'message': '-->置信度超出范围',
	})
        resp = jsonify(message)
        resp.status_code =504
    elif error ==505:
        message.update( {
			'message': '-->远程数据文件不存在',
	})
        resp = jsonify(message)
        resp.status_code =505
    elif error ==506:
        message.update( {
			'message': '-->模型未发布状态预测',
	})
        resp = jsonify(message)
        resp.status_code =506
    elif error ==507:
        message.update( {
			'message': '-->模型训练数据少于一天',
	})
        resp = jsonify(message)
        resp.status_code =507      
    elif error ==508:
        message.update( {
			'message': '-->模型训练数据为空',
	})
        resp = jsonify(message)
        resp.status_code =508  
    return resp



@app.route('/publish', methods=['POST'])
def publish():
    request_json = request.get_json()
    if 'modelId' not in request_json.keys:
        return(bad_request(400))
    
    model_id = request_json["modelId"]   
    
    if str(model_id) not in MODELS_STATUS.keys():return(bad_request(502))
        
    MODELS_STATUS[str(model_id)]["status"] = States.FABU    
    logging.info("******publish  modelid {} ,".format(model_id))
    
    message = {
			'status': True,
			'message': '-->模型发布成功',
	}
    resp = jsonify(message)
    resp.status_code = 200
    return resp




@app.route('/publish_cancel', methods=['POST'])
def publish_cancel():
    request_json = request.get_json()
    if 'modelId' not in request_json.keys:
        return(bad_request(400))
    
    model_id = request_json["modelId"]   
    
    if str(model_id) not in MODELS_STATUS.keys():return(bad_request(502))
        
    MODELS_STATUS[str(model_id)]["status"] = States.XL_WANCHENG   
     
    logging.info("******publish_cancel  modelid {} ,".format(model_id))
    
    message = {
			'status': True,
			'message': '-->模型取消发布成功',
	}
    resp = jsonify(message)
    resp.status_code = 200
    return resp



def evaluate_task(delta2,evaluationId,delta1,local_path_csv,assistKKS,model_id,loaded_model,epochs,chunks):
    
    try:
        logging.info("******evaluate_task  modelid {} ,".format(model_id))
        data = pd.read_csv(local_path_csv)
        X = data.loc[:,tuple(assistKKS)]
        #y = data.loc[:,mainKKS]
        X_const = sm.add_constant(X)
        times_start = data.iloc[:1,0].values[0]

        #df.insert(0,'const',1.0)
        
        #批量预测
        predictions = loaded_model.predict(X_const)
        df_const = X_const
        paras  = loaded_model.conf_int(Config.confidence)
        d_pre = paras[0][0]
        up_pre = paras[1][0]
        
        for i in range(1,len(paras)):
            d_pre += df_const[assistKKS[i-1]] * paras[0][i] 
            up_pre += df_const[assistKKS[i-1]] * paras[1][i] 

        paras  = loaded_model.conf_int(Config.confidence_second)
        d_pre2 = paras[0][0]
        up_pre2 = paras[1][0]
        
        for i in range(1,len(paras)):
            d_pre2 += df_const[assistKKS[i-1]] * paras[0][i] 
            up_pre2 += df_const[assistKKS[i-1]] * paras[1][i] 

        up_pre += (Config.K1*delta1-Config.B1)*up_pre

        up_pre2 += up_pre2 *(Config.K2*delta2-Config.B2)
        d_pre -=d_pre* (Config.K1*delta1-Config.B1)
        d_pre2-=d_pre2*(Config.K2*delta2-Config.B2)


        pred_interval = {"prediction":list(predictions.values),"first_upper":list(up_pre.values),\
            "first_lower":list(d_pre.values),"second_upper":list(up_pre2.values),\
                "second_lower":list(d_pre2.values)}
        #不带索引就 列值
        logging.info("******evaluate_task modelid {},{}".format(model_id,pred_interval))
        prediction_series = json.dumps(pred_interval)
        logging.info("******evaluate_task modelid {},{}".format(model_id,prediction_series))
        re = redis.StrictRedis(host=Config.redis_host,port=Config.redis_port,db=Config.redis_db,password=Config.redis_password)

        keys_redis = "evaluate_"+ str(evaluationId)+"_"+str(model_id) +"_"+ str(epochs) +"_"+ str(chunks)

        re.set(keys_redis,prediction_series)
    except Exception as e:
        logging.info("******evaluate_task modelid {},excep:{},trace {}".format(model_id,e,traceback.format_exc()))
        message = {
        'status': False,
        'message': "python评估中异常",
        
        #'data':prediction_series
        "model_id": model_id,
        "keys_redis": "evaluate_"+ str(evaluationId)+"_"+str(model_id) +"_"+ str(epochs) +"_"+ str(chunks)

        }
        requests.post(Config.java_host_evaluate, \
                    data = json.dumps(message),\
                    headers= header)

        raise e
    
    message = {
        'status': True,
        'message': "评估完成",
        #'data':prediction_series
        "keys_redis": keys_redis,
        "times_start": times_start
    }
    logging.info("******evaluate_task finished modelid {} ,{},".format(model_id,keys_redis))
    #java 接口
    requests.post(Config.java_host_evaluate, \
                    data = json.dumps(message),\
                    headers= header) 



@app.route('/evaluate', methods=['POST'])
def evaluate():

    try:
        request_json = request.get_json()  
        model_id = request_json["modelId"]
        dataUrl = request_json["dataUrl"]
        mainKKS = request_json["mainKKS"]
        assistKKS = request_json["assistKKS"]        
        delta1 = request_json["firstConfidence"]
        delta2 = request_json["secondConfidence"]
        evaluationId = request_json["evaluationId"]
        epochs = request_json["epochs"]
        chunks = request_json["chunks"]

        clf = 'model.pkl'     
        
        logging.info("******evaluating modelid {}".format(model_id))
        loaded_model = None
        with open('./model/train/' + str(model_id)+'/' + clf,'rb') as f:
            loaded_model = pickle.load(f)
        params  = loaded_model.params.index
        columns = list(params[1:])
        logging.info("******evaluating modelid {},assistKKS:{},colums:{},".format(model_id,assistKKS,columns))
        #logging.info(loaded_model.summary())
        message = {
        'status': False,
        'message': "相关测点不是list类型",
        "model_id": model_id
        }
        if not isinstance(assistKKS,list):
            requests.post(Config.java_host_evaluate,data = json.dumps(message),
                        headers= header)
            raise ValueError("相关测点不是list类型")

        for col in columns:
            if col not in assistKKS:
                logging.info("******evaluating modelid {},assistKKS:{},".format(model_id))
                return(bad_request(501))

        filename = dataUrl[dataUrl.rindex('/') +1:-4]  
        local_path = os.path.join(pathcwd,'dataset/evaluate/' + str(model_id)+'/')
        if not os.path.exists(local_path):
            os.makedirs( local_path )
        
        #哪个是绝对路径 哪个是文件名
        local_path_csv = os.path.join(local_path,filename +'.csv')
        #filename_ = wget.download(datasetUrl, out=local_path)
        p=subprocess.Popen(['wget','-N',dataUrl,'-P',local_path])
        if p.wait()==8:return(bad_request(505))
           
    except Exception as e:
        logging.info("******evaluating modelid {},excep:{}".format(model_id,e))
        message = {
        'status': False,
        'message': "python评估预处理异常",
        #'data':prediction_series
        "model_id": model_id,
        "keys_redis": "evaluate_"+str(model_id) +"_"+ str(epochs) +"_"+ str(chunks)
        }

        requests.post(Config.java_host_evaluate, \
                    data = json.dumps(message),\
                    headers= header)

        raise e
    
    

    executor.submit(evaluate_task,delta2,evaluationId,delta1,local_path_csv,assistKKS,model_id,loaded_model,epochs,chunks)
    

    message = {
        'status': True,
        'message': "评估开始",
        #'data':prediction_series

    }
    logging.info("******evaluating modelid {} ,".format(model_id))


    responses = jsonify(message)
    
    responses.status_code = 200
   
    return (responses)

@app.route('/evaluate_cancel', methods=['POST'])
def evaluate_cancel():

    request_json = request.get_json()       
    model_id = request_json["modelId"]
    delta1 = request_json["firstConfidence"]
    delta2 = request_json["secondConfidence"]
        
        #若已经运行，线程有取消接口，若已经运行，就取消不了了，，那就会回调java，所以，
        #所以这边不能完全保证，不再回调， 还是会回调，
        #所以java那边，要确保，若页面取消了，状态就改为取消， 那就我这边即使回调显示训练完成，也不要考虑了，
    evaluate_future = MODELS_STATUS[str(model_id)+"_future"]["evaluate"]
    MODELS_STATUS[str(model_id)]["status"] = States.XL_WANCHENG
    MODELS_STATUS[str(model_id)]["firstConfidence"] = delta1
    MODELS_STATUS[str(model_id)]["secondConfidence"] = delta2
    
    if evaluate_future.cancel():
        message = {
                'status': True,
                'message': '-->模型评估取消成功',
        }
        logging.info("******evaluate cancel modelid {} success".format(model_id))
    else:
        message = {
                'status': False,
                'message': '-->模型评估取消失败',
        }
        logging.info("******evaluate cancel modelid {} failed".format(model_id))
    resp = jsonify(message)
    resp.status_code = 200
    return resp


def evaluate_renew_task(delta2,evaluationId,delta1,local_path_csv,assistKKS,model_id,loaded_model,epochs,chunks):
    
    try:
        logging.info("******evaluate_renew_task  modelid {} ,".format(model_id))
        data = pd.read_csv(local_path_csv)
        X = data.loc[:,tuple(assistKKS)]
        #y = data.loc[:,mainKKS]
        X_const = sm.add_constant(X)
        times_start = data.iloc[:1,0].values[0]

        #df.insert(0,'const',1.0)
        
        #批量预测
        predictions = loaded_model.predict(X_const)
        df_const = X_const
        paras  = loaded_model.conf_int(Config.confidence)
        d_pre = paras[0][0]
        up_pre = paras[1][0]
        
        for i in range(1,len(paras)):
            d_pre += df_const[assistKKS[i-1]] * paras[0][i] 
            up_pre += df_const[assistKKS[i-1]] * paras[1][i] 

        paras  = loaded_model.conf_int(Config.confidence_second)
        d_pre2 = paras[0][0]
        up_pre2 = paras[1][0]
        
        for i in range(1,len(paras)):
            d_pre2 += df_const[assistKKS[i-1]] * paras[0][i] 
            up_pre2 += df_const[assistKKS[i-1]] * paras[1][i] 

        up_pre += (Config.K1*delta1-Config.B1)*up_pre

        up_pre2 += up_pre2 *(Config.K2*delta2-Config.B2)
        d_pre -=d_pre* (Config.K1*delta1-Config.B1)
        d_pre2-=d_pre2*(Config.K2*delta2-Config.B2)


        pred_interval = {"first_upper":list(up_pre.values),\
            "first_lower":list(d_pre.values),"second_upper":list(up_pre2.values),\
                "second_lower":list(d_pre2.values)}
        #不带索引就 列值
        prediction_series = json.dumps(pred_interval)
        re = redis.StrictRedis(host=Config.redis_host,port=Config.redis_port,db=Config.redis_db,password=Config.redis_password)

        keys_redis = "evaluate_"+ str(evaluationId)+"_"+str(model_id) +"_"+ str(epochs) +"_"+ str(chunks)

        re.set(keys_redis,prediction_series)
    except Exception as e:
        logging.info("******evaluate_renew_task modelid {},excep:{}".format(model_id,e))
        message = {
        'status': False,
        'message': "py重新评估中异常",
        "model_id": model_id,
        "keys_redis": "evaluate_"+ str(evaluationId)+"_"+str(model_id) +"_"+ str(epochs) +"_"+ str(chunks)
                                  
        }
        requests.post(Config.java_host_evaluate, \
                    data = json.dumps(message),\
                    headers= header)

        raise e

    message = {
        'status': True,
        'message': "重新评估完成",
        #'data':prediction_series
        "keys_redis": keys_redis,
        "times_start": times_start
    }
    logging.info("******evaluate_renew_task finished modelid {} ,{},".format(model_id,keys_redis))
    #java 接口
    requests.post(Config.java_host_evaluate, \
                    data = json.dumps(message),\
                    headers= header) 



@app.route('/evaluate_renew', methods=['POST'])
def evaluate_renew():

    try:
        request_json = request.get_json()  
        model_id = request_json["modelId"]
        dataUrl = request_json["dataUrl"]
        mainKKS = request_json["mainKKS"]
        assistKKS = request_json["assistKKS"]        
        delta1 = request_json["firstConfidence"]
        delta2 = request_json["secondConfidence"]
        evaluationId = request_json["evaluationId"]
        epochs = request_json["epochs"]
        chunks = request_json["chunks"]

        clf = 'model.pkl'     
        
        logging.info("******evaluating renew modelid {},".format(model_id))
        loaded_model = None
        with open('./model/train/' + str(model_id)+'/' + clf,'rb') as f:
            loaded_model = pickle.load(f)
        params  = loaded_model.params.index
        columns = list(params[1:])
        for col in columns:
            if col not in assistKKS:
                return(bad_request(501))

        filename = dataUrl[dataUrl.rindex('/') +1:-4]  
        local_path = os.path.join(pathcwd,'dataset/evaluate/' + str(model_id)+'/')
        if not os.path.exists(local_path):
            os.makedirs( local_path )
        
        #哪个是绝对路径 哪个是文件名
        local_path_csv = os.path.join(local_path,filename +'.csv')
        #filename_ = wget.download(datasetUrl, out=local_path)
        p=subprocess.Popen(['wget','-N',dataUrl,'-P',local_path])
        if p.wait()==8:return(bad_request(505))
           
    except Exception as e:
        logging.info("******evaluating renew modelid {},excep:{}".format(model_id,e))
        message = {
        'status': False,
        'message': "py评估预处理异常",
        "model_id": model_id,
        "keys_redis": "evaluate_"+ str(evaluationId)+"_"+str(model_id) +"_"+ str(epochs) +"_"+ str(chunks)

        }
        requests.post(Config.java_host_evaluate, \
                    data = json.dumps(message),\
                    headers= header)

        raise e
    
    

    executor.submit(evaluate_renew_task,delta2,evaluationId,delta1,local_path_csv,assistKKS,model_id,loaded_model,epochs,chunks)
    

    message = {
        'status': True,
        'message': "重新评估开始",
        #'data':prediction_series

    }
    logging.info("******evaluating renew asycio modelid {} ,".format(model_id))


    responses = jsonify(message)
    
    responses.status_code = 200
    
    return (responses)



def train_task(state,local_path_data,assistKKS,mainKKS,model_id,local_path_model):
    
    try:
           
        message = {
        "status": False,
		"message": '',
	    "model_id": model_id
        }
    
        
        data = pd.read_csv(local_path_data)#,encoding = 'utf-8')#,error_bad_lines=False)
        if(len(data)==0):
            message.update( {
			"message": "训练数据为空"})
            requests.post(Config.java_host_train,data = json.dumps(message),
                        headers= header)
            
            raise ValueError("训练数据为空")
        if(len(data)<86400):
            message.update( {
			"message": "训练数据至少一天"})
            requests.post(Config.java_host_train,data = json.dumps(message),
                        headers= header)
            raise ValueError("训练数据至少一天")
        
        if not isinstance(assistKKS,list):
            message.update( {
			"message": "相关测点不是list类型"})
            requests.post(Config.java_host_train,data = json.dumps(message),
                        headers= header)
            raise ValueError("相关测点不是list类型")
        
        X = data.loc[:,tuple(assistKKS)]

        y = data.loc[:,mainKKS]

        X_const = sm.add_constant(X)
        model = sm.OLS(y,X_const ).fit() 
        
        #model.conf_int(0.05)
        logging.info("******train_task finished modelid {},rsquared:{} ".format(model_id,model.rsquared))

        filepath = local_path_model+'model.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    except Exception as e:
        logging.info("******training_task modelid {},excep:{}".format(model_id,e))
        message = {
        'status': False,
        'message':"python训练中异常",
        #'message': "训练中异常excep: " + str(e),
        #'data':prediction_series
        "model_id": model_id
 
        }
        requests.post(Config.java_host_train, \
                    data = json.dumps(message),\
                    headers= header)

        raise e
    
    
    logging.info("******train_task finished modelid {} saved".format(model_id))
    state["model"] =   model
    state["status"] = States.XL_WANCHENG
    #如果已有key呢，会直接覆盖吗，还是要先remove 我记得
    MODELS_STATUS[str(model_id)] = state 
    #state = {'modelId':1,"state":0}
    #java 接口
    message = {
        'status': True,
        'message': "训练完成",
        #'data':prediction_series
        "model_id": model_id
    }
    requests.post(Config.java_host_train, \
                    data = json.dumps(message),\
                    headers= header)


@app.route('/train', methods=['POST'])
def train():
    try:
        request_json = request.get_json()
        
        model_id = request_json["modelId"]
        dataUrl = request_json["dataUrl"]
        mainKKS = request_json["mainKKS"]
        
        #列表
        assistKKS = request_json["assistKKS"]
        
        #MODELS_MAP[str(model_id)]["status"] = STATES.training
        path_data = './dataset/train/' + str(model_id)+'/'
        path_model = './model/train/' + str(model_id)+'/'

        if not os.path.exists(path_data):   os.makedirs( path_data )
        if not os.path.exists(path_model):    os.makedirs( path_model )  
        print('*'*400)
        logging.info('*'*400)
        logging.info("******,{}****".format(dataUrl))
        print(dataUrl)
        p= subprocess.Popen(['wget','-N',dataUrl,'-P',path_data])
        if p.wait()==8:return(bad_request(505))
        filename = dataUrl[dataUrl.rindex('/') +1:-4] 
        local_path_data = os.path.join(pathcwd,'dataset/train/' + str(model_id)+'/'+filename + '.csv')
        
        local_path_model = os.path.join(pathcwd,'model/train/' + str(model_id)+'/')
    except Exception as e:
        logging.info("******training modelid {},excep:{}".format(model_id,e))
        message = {
        'status': False,
        'message': "python训练预处理异常",
        "model_id": model_id

        }
        requests.post(Config.java_host_train, \
                    data = json.dumps(message),\
                    headers= header)

        raise e

    state = {"status":States.XUNLIAN_ZHONG,"modelid":model_id,"firstConfidence":0.95,"secondConfidence":0.98}
        
    train_future = executor.submit(train_task,state,local_path_data,assistKKS,mainKKS,model_id,local_path_model)
    MODELS_STATUS[str(model_id)+"_future"] = {}
    MODELS_STATUS[str(model_id)+"_future"]["train"]=train_future
        
    #MODELS_MAP[str(model_id)]["status"] = STATES.training_finish
    
    message = {
			'status': True,
			'message': '-->模型开始训练',
	}
    resp = jsonify(message)
    resp.status_code = 200
    return resp



@app.route('/train_cancel', methods=['POST'])
def train_cancel():

    request_json = request.get_json()       
    model_id = request_json["modelId"]
        
        #若已经运行，线程有取消接口，若已经运行，就取消不了了，，那就会回调java，所以，
        #所以这边不能完全保证，不再回调， 还是会回调，
        #所以java那边，要确保，若页面取消了，状态就改为取消， 那就我这边即使回调显示训练完成，也不要考虑了，
    train_future = MODELS_STATUS[str(model_id)+"_"+"future"]["train"]
    MODELS_STATUS[str(model_id)]["status"] = States.XL_WANCHENG

    if train_future.cancel():
        message = {
                'status': True,
                'message': '-->模型训练取消成功',
        }
        logging.info("******train cancel modelid {} success".format(model_id))
    else:
        message = {
                'status': False,
                'message': '-->模型训练取消失败',
        }
        logging.info("******train cancel modelid {} failed".format(model_id))
    resp = jsonify(message)
    resp.status_code = 200
    return resp


def train_batch_task(modelIdKKS,datasetUrlList):
    
    try:
        result_bools = [False] * len(datasetUrlList)
        result_ids = []
        for i in range(len(datasetUrlList)):
            datasetUrl = datasetUrlList[i]
            print(datasetUrl)

            filename = datasetUrl[datasetUrl.rindex('/') +1:-4]

            model_id = eval(filename.split('_')[1]) 
            result_ids.append(model_id)
            state = {"status":States.XUNLIAN_ZHONG,"modelid":model_id,"firstConfidence":0.95,"secondConfidence":0.98}
            
            logging.info("***start trainbatch modelid: {}".format(model_id))
            local_path_data = './dataset/train/' + str(model_id)+'/'
            local_path_model = './model/train/' + str(model_id)+'/'
              
            if not os.path.exists(local_path_data):   os.makedirs( local_path_data )
            if not os.path.exists(local_path_model):    os.makedirs( local_path_model )  
            
            p= subprocess.Popen(['wget','-N',datasetUrl,'-P',local_path_data])
            
            if p.wait()==8:return(bad_request(505))
            local_path = os.path.join(pathcwd,'dataset/train/' + str(model_id)+'/', filename + '.csv')
                    
            data = pd.read_csv(local_path)

            data = data.dropna()
            for col in data.columns[1:]:
                data = data[np.abs(data[col]-data[col].mean()) <= (3*data[col].std())]
                
            assistKKS = data.columns[2:]
            mainKKS = data.columns[1]

            #双向校验
            if str(model_id) not in modelIdKKS.keys():continue
            
            kks = modelIdKKS[str(model_id)]
            if kks["mainKKS"] != mainKKS:continue
            
            A2B = [True if i in list(assistKKS) else True for i in kks["assistKKS"]] 
            B2A = [True if i in kks["assistKKS"] else True for i in list(assistKKS)] 
            if sum(A2B) != len(assistKKS) or sum(B2A) != len(assistKKS) :continue
            
            result_bools[i]=True       
            
            X = data.loc[:,tuple(assistKKS)]

            y = data.loc[:,mainKKS]

            X_const = sm.add_constant(X)
            model = sm.OLS(y,X_const ).fit() 
            
            modelpath = local_path_model+'model.pkl'
            with open(modelpath, 'wb') as f:
                pickle.dump(model, f)
            state["model"] =   model
            state["status"] = States.XL_WANCHENG
            MODELS_STATUS[str(model_id)] = state 
            
            logging.info("***finish trainbatch modelid: {}".format(model_id))
        
    except Exception as e:
        logging.info("******training_batch modelid {},excep:{}".format(model_id,e))
        message = {
        'status': False,
        'message': "py批量训练中异常",
        #'message': "训练中异常excep: " + str(e),
        #'data':prediction_series
        "model_id": model_id
 
        }
        pass
        #continue
        # resp = requests.post(Config.java_host_train_batch, \
        #             data = json.dumps(message),\
        #             headers= header)
        # raise e
    
    logging.info("******train_batch finished result:\n{},\n{}".format(result_ids,result_bools))

    message = {
        'status': True,
        'message': "批量训练完成",
        "train_results": result_bools,
        "train_models": result_ids
    }
    resp = requests.post(Config.java_host_train_batch, \
                    data = json.dumps(message),\
                    headers= header)




@app.route('/train_batch', methods=['POST'])
def train_batch():
    
    request_json = request.get_json()
    #{"id":{""}}
    modelIdKKS = request_json["modelIdKKSDict"]
    datasetUrlList = request_json["datasetUrlList"]
    
    #MODELS_MAP[str(model_id)]["status"] = STATES.training

    train_batch_future = executor.submit(train_batch_task,modelIdKKS,datasetUrlList)
    
    message = {
			'status': True,
			'message': '-->模型批量开始训练',
	}

    resp = jsonify(message)
    resp.status_code = 200
    return resp


def angle_func(x,y):

    z1 = np.polyfit(x, y, 1)
    coefficient = z1[1]  # 获取系数
    angle = math.atan(coefficient) * 180 / math.pi
    result = 0
    if angle >= Config.trendThreshold: result = Qushi.CHIXU_SHENG.value
    elif angle <= -Config.trendThreshold: result = Qushi.CHIXU_JIANG.value
    else:result = Qushi.CHIXU_PING.value

    return result,angle


def delta_func(pre_vals):
    
    cur_val,pre_val = pre_vals[-1],(pre_vals[0]+pre_vals[1])/2.0

    delta = (cur_val-pre_val)/math.fabs(pre_val) 
    result = 0
    if delta >= Config.suddenChangeThreshold:  result = Qushi.TUBIAN_SHENG.value         
    elif delta <= -Config.suddenChangeThreshold: result = Qushi.TUBIAN_JIANG.value
    else: result = Qushi.TUBIAN_PING.value
    
    return result,delta

def send_java(result,model_id):
    message = {
    'status': True,
    'message': result,
    "model_id": model_id
    }

    requests.post(Config.java_host_trend, \
                data = json.dumps(message),\
                headers= header)  
  


def trend_task(local_path_data,mainKKS,model_id):

    
    da = pd.read_csv(local_path_data)
    #to kks 
    data = list(da[mainKKS])
    tmp = range(len(data))
    
    result,angle = angle_func(tmp,data)
    if result!= 1:send_java(result,model_id)
    logging.info("******trending modelid {},angle:{},result:{} ".format(model_id,angle,Qushi(result)))
    
    pre_dataset = data[-600:]
    pre_index = range(len(pre_dataset))
    
    cur_val = list(data)[-1]
    pre_vals = list(data)[-3:]
    
    result,delta = delta_func(pre_vals)
    if result!= 6:send_java(result,model_id)
    logging.info("******trending modelid {},delta:{},result:{} ".format(model_id,delta,Qushi(result)))
    
    dateSeries = list(da[da.columns[0]])

    curTime = dateSeries[-1]
    
    rstfive_angle =list()
    rstfive_delta =list()
    
    re = redis.StrictRedis(host=Config.redis_host,port=Config.redis_port,db=Config.redis_db,password=Config.redis_password)

    keys_redis = "REAL_TIME_VALUE:"+ mainKKS

    
    # {"kks":,"timestamp","value":}
    flag =True
    while(flag):
       
        if not flag:break
        time.sleep(1)
        
        redisDict  = eval(json.loads(re.get(keys_redis)))
        if redisDict["timestamp"] == curTime:
            
            continue
        curTime = redisDict["timestamp"]
        value = eval(redisDict["value"])

        pre_dataset.pop(0)
        pre_dataset.append(value)

        rst,angle = angle_func(pre_index,pre_dataset)
        if rst!= 1:send_java(result,model_id)
        rstfive_angle.append(rst)
        logging.info("******trending modelid {},angle:{}, result:{}, rsthistory:{}".format(\
            model_id,delta,Qushi(rst),rstfive_angle))

        pre_vals.pop(0)
        pre_vals.append(value)
        
        rst,delta = delta_func(pre_vals)
        if rst!= 6:send_java(rst,model_id)
        
        rstfive_delta.append(rst)
        logging.info("******trending modelid {},delta:{},result:{},rsthistory:{} ".format( \
            model_id,delta,Qushi(rst),rstfive_delta))
        
        flag = Config.QUSHI_MODELS_STATUS[str(model_id)+"_flag"]

        for k,v in itertools.groupby(rstfive_delta):
            if k==6 and sum(list(v))>Config.trend_Threshold_times*6:
                flag = False
                    
        for k,v in itertools.groupby(rstfive_angle):
            if k==1 and sum(list(v))>Config.trend_Threshold_times*1:
                flag = False             
        
        time.sleep(1)
        
        logging.info("******trending logg {}".format(Config.QUSHI_MODELS_STATUS))


@app.route('/trend_cancel', methods=['POST'])
def trend_cancel():
    
    request_json = request.get_json()
    model_id = request_json["modelId"]
    flag = False 
    Config.QUSHI_MODELS_STATUS[str(model_id)+"_flag"] =flag

    message = {
        'status': True,
        'message': 'AI趋势监测取消',
    }
    resp = jsonify(message)
    resp.status_code = 200
    time.sleep(1)

    logging.info("******trend_cancel modelid {}, ".format(model_id))
    return resp    
    

@app.route('/trend', methods=['POST'])
def trend():
    try:
        
        request_json = request.get_json()

        mainKKS = request_json["mainKKS"]
        dataUrl = request_json["dataUrl"]
        model_id = request_json["modelId"] 
        
        path_data = './dataset/trend/' + str(model_id)+'/'

        if not os.path.exists(path_data):   os.makedirs( path_data )
        
        p= subprocess.Popen(['wget','-N',dataUrl,'-P',path_data])
        if p.wait()==8:return(bad_request(505))
        filename = dataUrl[dataUrl.rindex('/') +1:-4] 
        local_path_data = os.path.join(pathcwd,'dataset/trend/' + str(model_id)+'/'+filename + '.csv')
        
    except Exception as e:
        logging.info("******trending modelid {},excep:{}".format(model_id,e))
        message = {
        'status': False,
        'message': "python趋势预处理异常",
        "model_id": model_id

        }
        print(message)

        raise e
    flag = str(model_id)+"_flag"
    Config.QUSHI_MODELS_STATUS[flag] = True
    trend_future = executor.submit(trend_task,local_path_data,mainKKS,model_id)
    
 
    
    message = {
			'status': True,
			'message': '-->开始趋势AI监测',
	}
    resp = jsonify(message)
    resp.status_code = 200
    return resp

   
  
@app.route('/confidence', methods=['POST'])
def confidence():
    # lmm.conf_int(0.05)  默认 0.05
    # 0.05   即 0.025-0.975 即涵盖95%  置信度95
    # 0.1  即 0.05-0.95  即涵盖90%    置信度90
    # 0.02 即  0.01- 0.99 即涵盖98%   置信度 98
    request_json = request.get_json()
        
    model_id = request_json["modelId"]
    confidence = request_json["confidence"]
    confidence = 1 - confidence
    #是在什么时候修改的，是在评估，是在发布后，
    #置信度，预测还要带上  
    loaded_model = None
    if str(model_id) in MODELS_MAP.keys():
        loaded_model = MODELS_MAP[str(model_id)]


    loaded_model.conf_int(0.05)
    
    

if __name__ == "__main__":
    # 将host设置为0.0.0.0，则外网用户也可以访问到这个服务
    app.run(host="0.0.0.0", port=8383)#, debug=True)


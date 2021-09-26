import os
import pandas as pd
import numpy as np 
#import dill as pickle
from flask import Flask, jsonify, request
import statsmodels.api as sm
import paramiko
import pickle
import os
import statsmodels
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import redis
import statsmodels.api as sm
import xlrd
from matplotlib import pyplot as plt
import pandas as pd

import statsmodels.api as sm #最小二乘
from statsmodels.formula.api import ols #加载ols模型
import wget
import json
import requests
import logging 
from config import Config
import subprocess
from concurrent.futures import ThreadPoolExecutor


executor = ThreadPoolExecutor(8)
app = Flask(__name__)


from enum import Enum


pathcwd = os.path.dirname(__file__)

logpath = 'log/serving.std.out'
logging.basicConfig(filename=logpath,filemode='a',format='%(asctime)s %(name)s:%(levelname)s:%(message)s',datefmt="%d-%m-%Y \
    %H:%M:%S",level=logging.DEBUG)


class STATES(Enum):
    pretrain = 0
    training=1
    training_finish = 2
    

STATES_ENUM ={"初始化":0,"待训练":1,"训练中":2,'待发布':3,'已发布':4,"训练失败":5}

MODELS_MAP = {}  #"id":id,  "id":model,  
#"id":{"status":,"model":只放发布的模型扔内存里，"confidence"，
# "train_version":0123,"release_verison":0123}  


#评分的线性回归


@app.route('/predict_test', methods=['POST','GET'])
def predict():
    try:
        request_json = request.get_json()   
        model_id = request_json["modelId"]
        delta1 = request_json["firstConfidence"]
        delta2 = request_json["secondConfidence"]
        
        if (delta1>=0.98 or delta1 <0.95 or delta2>=1.0 or delta2 <0.98): 
            return (bad_request(504)) 
        print("Loading the model...")
        loaded_model = None
        clf = 'model.pkl'
        #if str(model_id) in request_json.keys():
        #    loaded_model = request_json[str(model_id)]
        local_path = './model/' + str(model_id)+'/'
        if not os.path.exists(local_path):return(bad_request(402))
        #if str(model_id) not in MODELS_MAP.keys():
        with open(local_path + clf,'rb') as f:
            loaded_model = pickle.load(f)
        
            
            
        params  = loaded_model.params.index
        columns = list(params[1:])
        for col in columns:
            if col not in request_json.keys():
                return(bad_request(401))
                      
    except Exception as e:
        raise e 
    data=[]
    for i in range(1,len(params)):
        print(request_json[params[i]])
        data.append(request_json[params[i]]) 
    
    if len(data)==0 or '' in data:
        return(bad_request(400))
    else:
        data =  np.expand_dims(data,1)       
        df = pd.DataFrame(dict(zip(columns,data)))

        print("The model has been loaded...doing predictions now...")
        
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

        prediction_series = pd.DataFrame(pred_interval,index=[0])
              
        message = {
			'status': True,
			'message': "请求成功",
            'data':pred_interval
	    }
        
        responses = jsonify(message)
        
        responses.status_code = 200
    
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
    elif error ==401:
        message.update( {
			'message': '-->请求参数不一致错误，请检查相关测点、模型ID、请求接口',
	})
        resp = jsonify(message)
        resp.status_code = 401
    elif error ==402:
        message.update( {
			'message': '-->模型文件找不到',
	})
        resp = jsonify(message)
        resp.status_code = 402
    return resp



@app.route('/publish', methods=['POST'])
def publish():
    request_json = request.get_json()
    if 'modelId' not in request_json.keys:
        return(bad_request(401))
     
    model_id = request_json["modelId"]
    #若已经发布了，
    path = ''
    
    clf = 'model.pkl'
    loaded_model = None
    
    if str(model_id) not in MODELS_MAP.keys():
        with open('./model/publish/' + str(model_id)+'/' + clf,'rb') as f:
            if os.path.getsize()
            loaded_model = pickle.load(f)
    
    MODELS_MAP[str(model_id)] = loaded_model
    
    message = {
			'status': True,
			'message': request.url+'-->模型发布成功',
	}
    resp = jsonify(message)
    resp.status_code = 200
    return resp

@app.route('/publish_cancel', methods=['POST'])
def publish_cancel():
    request_json = request.get_json()
    if 'modelId' not in request_json.keys:
        return(bad_request(401))
    
    model_id = request_json["modelId"]   
    
    if model_id not in MODELS_MAP.keys:
        return(bad_request(401))
    
    #已经pop 在pop 报错
    MODELS_MAP.pop[str(model_id)]
    
    message = {
			'status': True,
			'message': request.url+'-->模型取消发布成功',
	}
    resp = jsonify(message)
    resp.status_code = 200
    return resp


def evaluate_task(delta2,delta1,local_path_csv,assistKKS,model_id,loaded_model,epochs,chunks):
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
    prediction_series = json.dumps(pred_interval)
    re = redis.StrictRedis(host='172.17.224.171',port=6379,db=0,password=123456)

    keys_redis = "evaluate_"+ str(model_id) +"_"+ str(epochs) +"_"+ str(chunks)

    re.set(keys_redis,prediction_series)

    message = {
        'status': True,
        'message': "评估完成",
        #'data':prediction_series
        "keys_redis": keys_redis,
        "times_start": times_start
    }
    logging.info("******evaluate_task finished modelid {} ,".format(model_id))
    #java 接口
    header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}  
    resp = requests.post("http://172.17.231.79:30069/model/bizModel/getModelEvaluationResult", \
                    data = json.dumps(message),\
                    headers= header) 





@app.route('/evaluate', methods=['POST'])
def evaluate():

    try:
        request_json = request.get_json()  
        model_id = request_json["modelId"]
        datasetUrl = request_json["datasetUrl"]
        mainKKS = request_json["mainKKS"]
        assistKKS = request_json["assistKKS"]        
        delta1 = request_json["firstConfidence"]
        delta2 = request_json["secondConfidence"]
        epochs = request_json["epochs"]
        chunks = request_json["chunks"]

        clf = 'model.pkl'     
        
        logging.info("******evaluating modelid {},".format(model_id))
        loaded_model = None
        with open('./model/' + str(model_id)+'/' + clf,'rb') as f:
            loaded_model = pickle.load(f)
        params  = loaded_model.params.index
        columns = list(params[1:])
        for col in columns:
            if col not in assistKKS:
                return(bad_request(401))

        filename = datasetUrl[datasetUrl.rindex('/') +1:-4]  
        local_path = os.path.join(pathcwd,'dataset/evaluate/' + str(model_id)+'/')
        if not os.path.exists(local_path):
            os.makedirs( local_path )
        
        #哪个是绝对路径 哪个是文件名
        local_path_csv = os.path.join(local_path,filename +'.csv')
        #filename_ = wget.download(datasetUrl, out=local_path)
        p=subprocess.Popen(['wget','-N',datasetUrl,'-P',local_path])
        if p.wait()==8:return(bad_request(505))
           
    except Exception as e:
        raise e
    
    

    executor.submit(evaluate_task,delta2,delta1,local_path_csv,assistKKS,model_id,loaded_model,epochs,chunks)
    

    message = {
        'status': True,
        'message': "评估完成",
        #'data':prediction_series

    }
    #logging.info("******evaluating finished modelid {} ,".format(model_id))


    responses = jsonify(message)
    
    responses.status_code = 200
    
    return (responses)



@app.route('/evaluate_', methods=['POST'])
def evaluate_():
    

    try:
        request_json = request.get_json()  
        model_id = request_json["modelId"]
        datasetUrl = request_json["datasetUrl"]
        mainKKS = request_json["mainKKS"]
        assistKKS = request_json["assistKKS"]        
        delta1 = request_json["firstConfidence"]
        delta2 = request_json["secondConfidence"]
        
        clf = 'model.pkl'     
        print("Loading the model...")
        loaded_model = None
        with open('./model/' + str(model_id)+'/' + clf,'rb') as f:
            loaded_model = pickle.load(f)
        params  = loaded_model.params.index
        columns = list(params[1:])
        for col in columns:
            if col not in assistKKS:
                return(bad_request(401))
                      
    except Exception as e:
        raise e
    
    print(datasetUrl)
    filename = datasetUrl[datasetUrl.rindex('/') +1:-4]  
    local_path = os.path.join(pathcwd,'dataset/evaluate/' + str(model_id)+'/')
    if not os.path.exists(local_path):
        os.makedirs( local_path )
    
    #哪个是绝对路径 哪个是文件名
    local_path_csv = os.path.join(local_path,filename +'.csv')
    #filename_ = wget.download(datasetUrl, out=local_path)
    p=subprocess.Popen(['wget','-N',datasetUrl,'-P',local_path])
    p.wait()
    data = pd.read_csv(local_path_csv)
    X = data.loc[:,tuple(assistKKS)]
    #y = data.loc[:,mainKKS]
    X_const = sm.add_constant(X)


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
    prediction_series = json.dumps(pred_interval)
    
    re = redis.StrictRedis(host='172.17.224.171',port=6379,db=0,password=123456)

    key = "evaluate_modelid_pinggu_fenge_data"

    re.set(key,prediction_series)

    #re.get("evaluate_modelid_pinggu_fenge_data")    

    #带索引 1 2  3  是否改成时间戳
   ##pred_interval = {"prediction":predictions.values,"first_upper":up_pre.values,\
    #"first_lower":d_pre.values,"second_upper":up_pre2.values,\
    #    "second_lower":d_pre2.values}
    #prediction_series = pd.DataFrame(pred_interval)
    #print(prediction_series)
    #prediction_series = prediction_series.to_json()
    message = {
        'status': True,
        'message': "请求成功",
        #'data':prediction_series
        'key':key
    }
    
    responses = jsonify(message)
    
    responses.status_code = 200
    
    return (responses)


@app.route('/evaluate_redis', methods=['POST'])
def evaluate_redis():

    try:
        request_json = request.get_json()  
        model_id = request_json["modelId"]
        datasetUrl = request_json["datasetUrl"]
        mainKKS = request_json["mainKKS"]
        assistKKS = request_json["assistKKS"]        
        delta1 = request_json["firstConfidence"]
        delta2 = request_json["secondConfidence"]
        epochs = request_json["epochs"]
        chunks = request_json["chunks"]

        clf = 'model.pkl'     
        print("Loading the model...")
        loaded_model = None
        with open('./model/' + str(model_id)+'/' + clf,'rb') as f:
            loaded_model = pickle.load(f)
        params  = loaded_model.params.index
        columns = list(params[1:])
        for col in columns:
            if col not in assistKKS:
                return(bad_request(401))
                      
    except Exception as e:
        raise e
    
    print(datasetUrl)
    filename = datasetUrl[datasetUrl.rindex('/') +1:-4]  
    local_path = os.path.join(pathcwd,'dataset/evaluate/' + str(model_id)+'/')
    if not os.path.exists(local_path):
        os.makedirs( local_path )
    
    #哪个是绝对路径 哪个是文件名
    local_path_csv = os.path.join(local_path,filename +'.csv')
    #filename_ = wget.download(datasetUrl, out=local_path)
    p=subprocess.Popen(['wget','-N',datasetUrl,'-P',local_path])
    if p.wait()==8:return(bad_request(505))
    data = pd.read_csv(local_path_csv)
    X = data.loc[:,tuple(assistKKS)]
    #y = data.loc[:,mainKKS]
    X_const = sm.add_constant(X)


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
    prediction_series = json.dumps(pred_interval)
    re = redis.StrictRedis(host='172.17.224.171',port=6379,db=0,password=123456)

    key = "evaluate_"+ str(model_id) +"_"+ str(epochs) +"_"+ str(chunks)

    re.set(key,prediction_series)

    message = {
        'status': True,
        'message': "请求成功",
        #'data':prediction_series
        "key_redis": key
    }
    
    responses = jsonify(message)
    
    responses.status_code = 200
    
    return (responses)



@app.route('/evaluate_renew', methods=['POST'])
def evaluate_renew():

    try:
        request_json = request.get_json()  
        model_id = request_json["modelId"]
        datasetUrl = request_json["datasetUrl"]
        mainKKS = request_json["mainKKS"]
        assistKKS = request_json["assistKKS"]        
        delta1 = request_json["firstConfidence"]
        delta2 = request_json["secondConfidence"]
        
        clf = 'model.pkl'     
        print("Loading the model...")
        loaded_model = None
        with open('./model/' + str(model_id)+'/' + clf,'rb') as f:
            loaded_model = pickle.load(f)
        params  = loaded_model.params.index
        columns = list(params[1:])
        for col in columns:
            if col not in assistKKS:
                return(bad_request(401))
                      
    except Exception as e:
        raise e
    
    print(datasetUrl)
    filename = datasetUrl[datasetUrl.rindex('/') +1:-4]  
    local_path = './dataset/evaluate/' + str(model_id)+'/'
    if not os.path.exists(local_path):
        os.makedirs( local_path )
    
    local_path_csv = os.path.join(pathcwd,local_path+filename +'.csv')
    #filename_ = wget.download(datasetUrl, out=local_path)
    subprocess.Popen(['wget','-N',datasetUrl,'-P',local_path])

    data = pd.read_csv(local_path_csv)
    X = data.loc[:,tuple(assistKKS)]
    #y = data.loc[:,mainKKS]
    X_const = sm.add_constant(X)

    
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

    up_pre += up_pre* (Config.K1*delta1-Config.B1)
    up_pre2 += up_pre2 *(Config.K2*delta2-Config.B2)
    d_pre -=d_pre* (Config.K1*delta1-Config.B1)
    d_pre2-=d_pre2*(Config.K2*delta2-Config.B2)

    pred_interval = {"prediction":list(predictions.values),"first_upper":list(up_pre.values),\
        "first_lower":list(d_pre.values),"second_upper":list(up_pre2.values),\
            "second_lower":list(d_pre2.values)}
    #不带索引就 列值
    prediction_series = json.dumps(pred_interval)

    message = {
        'status': True,
        'message': "请求成功",
        'data':prediction_series
    }
    
    responses = jsonify(message)
    
    responses.status_code = 200
    
    return (responses)
 

@app.route('/evaluate_interval', methods=['POST'])
def evaluate_interval():

    try:
        request_json = request.get_json()  
        model_id = request_json["modelId"]
        
        clf = 'model.pkl'     
        print("Loading the model...")
        loaded_model = None
        with open('./model/' + str(model_id)+'/' + clf,'rb') as f:
            loaded_model = pickle.load(f)
        params  = loaded_model.params.index
        columns = list(params[1:])
        for col in columns:
            if col not in request_json.keys():
                return(bad_request(401))
                      
    except Exception as e:
        raise e
    
    data=[]
    for i in range(1,len(params)):
        data.append(request_json[params[i]]) 
    
    if len(data)==0 or '' in data:
        return(bad_request())
    else:
        data =  np.expand_dims(data,1)  
        df = pd.DataFrame(dict(zip(columns,data)))   
        print("The model has been loaded...doing predictions now...")
        
        df.insert(0,'const',1.0)
        df_const = df
        predictions = loaded_model.predict(df_const)
        
        paras  = loaded_model.conf_int()
        d_pre = paras[0][0]
        up_pre = paras[1][0]
        
        for i in range(1,len(paras)):
            d_pre += df_const.loc[0][i] * paras[0][i] 
            up_pre += df_const.loc[0][i] * paras[1][i] 

        pred_interval = {"prediction":predictions.loc[0],"upper":up_pre,"lower":d_pre}
        prediction_series = pd.DataFrame(pred_interval,index=[0])
          
        message = {
			'status': True,
			'message': "请求成功",
            'data':pred_interval
	    }
        
        responses = jsonify(message)
        
        responses.status_code = 200
    
    return (responses)
 

@app.route('/train', methods=['POST'])
def train():
    request_json = request.get_json()
    
    model_id = request_json["modelId"]
    datasetUrl = request_json["dataUrl"]
    mainKKS = request_json["mainKKS"]
    
    #列表
    assistKKS = request_json["assistKKS"]
    
    #MODELS_MAP[str(model_id)]["status"] = STATES.training

    print(datasetUrl)
    filename = datasetUrl[datasetUrl.rindex('/') +1:-4]  
    local_path = './dataset/' + str(model_id)+'/'
    if not os.path.exists(local_path):
        os.makedirs( local_path )
    subprocess.Popen(['wget','-N',datasetUrl,'-P',local_path])
    #哪个是绝对路径 哪个是文件名
    local_data = os.path.join(pathcwd,local_path, filename + '.csv')


    data = pd.read_csv(local_data)
    #去除null 空值
    data = data.dropna()
    #去除离群点
    for col in data.columns[1:]:
        data = data[np.abs(data[col]-data[col].mean()) <= (3*data[col].std())]

    X = data.loc[:,tuple(assistKKS)]

    y = data.loc[:,mainKKS]

    X_const = sm.add_constant(X)
    model = sm.OLS(y,X_const ).fit() 
    print(model.summary())

    model.conf_int(0.05)
    
    header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}
    
    state = {'modelId':1,"state":0}
    #java 接口
    resp = requests.post("http://0.0.0.0:8383/predict_single", \
                    data = json.dumps(state),\
                    headers= header)


    filepath = r'model/'+str(model_id)+'/model.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
        
    #MODELS_MAP[str(model_id)]["status"] = STATES.training_finish
    
    message = {
			'status': True,
			'message': request.url+'-->模型训练成功',
	}
    resp = jsonify(message)
    resp.status_code = 200
    return resp

@app.route('/train_renew', methods=['POST'])
def train_renew():
    request_json = request.get_json()
    
    model_id = request_json["modelId"]
    datasetUrl = request_json["dataUrl"]
    mainKKS = request_json["mainKKS"]
    
    #列表
    assistKKS = request_json["assistKKS"]
    
    #MODELS_MAP[str(model_id)]["status"] = STATES.training

    print(datasetUrl)
    filename = datasetUrl[datasetUrl.rindex('/') +1:-4]  
    local_path = './dataset/train/' + str(model_id)+'/'
    if not os.path.exists(local_path):
        os.makedirs( local_path )
    subprocess.Popen(['wget','-N',datasetUrl,'-P',local_path])

    #哪个是绝对路径 哪个是文件名
    local_data =  os.path.join(pathcwd,local_path, filename + '.csv')


    data = pd.read_csv(local_data)
    #去除null 空值
    data = data.dropna()
    #去除离群点
    for col in data.columns[1:]:
        data = data[np.abs(data[col]-data[col].mean()) <= (3*data[col].std())]
        
    X = data.loc[:,tuple(assistKKS)]

    y = data.loc[:,mainKKS]

    X_const = sm.add_constant(X)
    model = sm.OLS(y,X_const ).fit() 
    print(model.summary())

    model.conf_int(0.05)
    
    header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}
    
    state = {'modelId':1,"state":0}
    #java 接口
    resp = requests.post("http://0.0.0.0:8383/predict_single", \
                    data = json.dumps(state),\
                    headers= header)


    filepath = r'model/'+str(model_id)+'/model.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
        
    #MODELS_MAP[str(model_id)]["status"] = STATES.training_finish
    
    message = {
			'status': True,
			'message': request.url+'-->模型训练成功',
	}
    resp = jsonify(message)
    resp.status_code = 200
    return resp


@app.route('/train_batch', methods=['POST'])
def train_batch():
    
    request_json = request.get_json()
    
    #model_id = request_json["modelId"]
    datasetUrl_list = request_json["datasetUrlList"]
    #mainKKS = request_json["mainKKS"]
    
    #列表
    #assistKKS = request_json["assistKKS"]
    
    #MODELS_MAP[str(model_id)]["status"] = STATES.training
    
    for datasetUrl in datasetUrl_list:
        print(datasetUrl)

        filename = datasetUrl[datasetUrl.rindex('/') +1:-4]
        model_id = eval(filename.split('_')[1]) 
        logging.info("***start train modelid: {}".format(model_id))
        local_path_data = './dataset/train/' + str(model_id)+'/'
        local_path_model = './model/train/' + str(model_id)+'/'

        if not os.path.exists(local_path_data):   os.makedirs( local_path_data )
        if not os.path.exists(local_path_model):    os.makedirs( local_path_model )  
        
        subprocess.Popen(['wget','-N',datasetUrl,'-P',local_path_data])
        
        local_path = os.path.join(pathcwd,local_path_data, filename + '.csv')
        
        
        data = pd.read_csv(local_path)

        data = data.dropna()
        for col in data.columns[1:]:
            data = data[np.abs(data[col]-data[col].mean()) <= (3*data[col].std())]
            
        assistKKS = data.columns[2:]
        mainKKS = data.columns[1]

        X = data.loc[:,tuple(assistKKS)]

        y = data.loc[:,mainKKS]

        X_const = sm.add_constant(X)
        model = sm.OLS(y,X_const ).fit() 
        
        modelpath = local_path_model+'model.pkl'
        with open(modelpath, 'wb') as f:
            pickle.dump(model, f)
        
        logging.info("***finish modelid: {}".format(model_id))
    message = {
			'status': True,
			'message': request.url+'-->模型训练成功',
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
    app.run(host="0.0.0.0", port=8384, debug=True)
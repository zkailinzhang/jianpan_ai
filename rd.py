import redis
import struct
import numpy as np
import pickle
import json

def toRedis(r,a,n):
   """Store given Numpy array 'a' in Redis under key 'n'"""
   h, w = a.shape
   shape = struct.pack('>II',h,w)
   encoded = shape + a.tobytes()

   # Store encoded data in Redis
   r.set(n,encoded)
   return

def fromRedis(r,n):
   """Retrieve Numpy array from Redis key 'n'"""
   encoded = r.get(n)
   h, w = struct.unpack('>II',encoded[:8])
   # Add slicing here, or else the array would differ from the original
   a = np.frombuffer(encoded[8:]).reshape(h,w)
   return a


path = 'train/1115/1/model.pkl'
ff = open(path,'rb')
mm = pickle.load(ff)


'''
redis:

key即modelid

pp = np.asarray(mm.params)
conf1 = np.array( mm.conf_int(0.95))
conf2 = np.array( mm.conf_int(0.98))

"898" : {"id":898,"param":pp,"conf1":conf1, "conf2":conf2}

后面的字典 json dumps  json.loads
判断key 模型id 还没某存，conn[str(90)]==None即可
'''

redis.StrictRedis(host='172.17.224.171',port=6379,db=0,password=123456)
conn = redis.StrictRedis(host='127.0.0.1')
conn = redis.StrictRedis(host='127.0.0.1',port=6379,db=1,password=123456,charset='utf-8',decode_responses=True)


xxx =conn.get('con1')

xxx
b'[[2.55023055 2.55051747]\n [0.02578415 0.02578457]]'

yy =xxx.decode()

yy
'[[2.55023055 2.55051747]\n [0.02578415 0.02578457]]'

import numpy as np

np.asarray(yy)
array('[[2.55023055 2.55051747]\n [0.02578415 0.02578457]]', dtype='<U50')


a1 = np.asarray([[2.55023055,2.55051747],[0.02578415,0.02578457]])
 
array([[2.55023055, 2.55051747],
       [0.02578415, 0.02578457]])




toRedis(conn,a1,'a1')

fromRedis(conn,'a1')

array([[2.55023055, 2.55051747],
       [0.02578415, 0.02578457]])


a2 = np.array(mm.conf_int(0.95))
a1
array([[2.55023055, 2.55051747],
       [0.02578415, 0.02578457]])

a2
array([[2.55023055, 2.55051747],
       [0.02578415, 0.02578457]])

np.asarray(mm.conf_int(0.95))
array([[2.55023055, 2.55051747],
       [0.02578415, 0.02578457]])


toRedis(conn,a2,'a2')
fromRedis(conn,'a2')
array([[2.55023055, 2.55051747],
       [0.02578415, 0.02578457]])



val = {}
val["id"]=1115
val["param"]= np.array(mm.params).tolist()
val["conf1"]= np.array(mm.conf_int(0.95)).tolist()
val["conf2"]= np.array(mm.conf_int(0.98)).tolist()


val["param"]= np.array(mm.params).tolist()

val
 
{'id': 1115,
 'param': [2.5503740140350097, 0.025784359966865796],
 'conf1': [[2.550230554572805, 2.5505174734972145],
  [0.025784146234931804, 0.025784573698799787]],
 'conf2': [[2.550316661828208, 2.5504313662418117],
  [0.02578427452113855, 0.025784445412593043]]}


conn.set(str(1115),json.dumps(val))
True

conn.get('1115')
b'{"id": 1115, "param": [2.5503740140350097, 0.025784359966865796], "conf1": [[2.550230554572805, 2.5505174734972145], [0.025784146234931804, 0.025784573698799787]], "conf2": [[2.550316661828208, 2.5504313662418117], [0.02578427452113855, 0.025784445412593043]]}'


gv = conn.get('1115')

getv = json.loads(gv)
getv['conf2']

[[2.550316661828208, 2.5504313662418117],
 [0.02578427452113855, 0.025784445412593043]]

getv['conf2'][0][0]
2.550316661828208

#预测
data = [1.0,20]
np.sum(np.array(data)*np.array(getv['param']))

#置信1
直接取值就行，


然后回传，
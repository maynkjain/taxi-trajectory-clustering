import pandas as pd
from matplotlib import pyplot as plt
from ast import literal_eval
from flask import request, jsonify
import urllib
import zipfile
import os
import scipy.io
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.cluster import DBSCAN

from kmedoid import kMedoids


# columns = ["POLYLINE"]
# df = pd.read_csv("train.csv", usecols=columns,nrows = 200)
# df.to_csv("trajectory.csv", index = False, header = True)

# columns = ["CLUSTER_ID","COLOR","POLYLINE"]
# df = pd.read_csv("test.csv", usecols=columns)
# #print("Contents in csv file:\n",df)


def visualizeTrajectory(traj_lst, cluster_lst):
    cluster_count = np.max(cluster_lst) + 1
    for traj, cluster in zip(traj_lst, cluster_lst):
        if cluster == -1:
            plt.plot(traj[:, 0], traj[:, 1], c='k', linestyle='dashed')
        else:
            plt.plot(traj[:, 0], traj[:, 1], c=color_lst[cluster % len(color_lst)])
    #plt.show()

sns.set()
plt.rcParams['figure.figsize'] = (12, 12)
color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_lst.extend(['firebrick', 'olive', 'indigo', 'khaki', 'teal', 'saddlebrown', 
                 'skyblue', 'coral', 'darkorange'])

columns = ["POLYLINE"]
df = pd.read_csv("trajectory.csv", usecols=columns)

traj_lst = []
for index,row in df.iterrows():
    temp = {}
    temp['trajectory'] = literal_eval(row[0])
    traj_lst.append(temp)
    
traj_arr = []

for it in traj_lst:
    arr = np.asarray(it['trajectory'])
    traj_arr.append(arr)

#for traj in traj_arr:
    #plt.plot(traj[:, 0], traj[:, 1])

degree_threshold = 5

for traj_index, traj in enumerate(traj_arr):
    
    hold_index_lst = []
    previous_azimuth= 1000
    
    for point_index, point in enumerate(traj[:-1]):
        next_point = traj[point_index + 1]
        diff_vector = next_point - point
        azimuth = (math.degrees(math.atan2(*diff_vector)) + 360) % 360
        
        if abs(azimuth - previous_azimuth) > degree_threshold:
            hold_index_lst.append(point_index)
            previous_azimuth = azimuth
    hold_index_lst.append(traj.shape[0] - 1)
    
    traj_arr[traj_index] = traj[hold_index_lst, :]


def hausdorff( u, v):
    d = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    return d

traj_count = len(traj_arr)
D = np.zeros((traj_count, traj_count))

for i in range(traj_count):
    for j in range(i + 1, traj_count):
        distance = hausdorff(traj_arr[i], traj_arr[j])
        D[i, j] = distance
        D[j, i] = distance


k = 6 # The number of clusters
medoid_center_lst, cluster2index_lst = kMedoids(D, k)

cluster_lst = np.empty((traj_count,), dtype=int)

for cluster in cluster2index_lst:
    cluster_lst[cluster2index_lst[cluster]] = cluster

# print("Cluster list: ")
# print(cluster_lst)
# print("Traj list: ")
# print(traj_ls[3])

visualizeTrajectory(traj_arr, cluster_lst)


# mdl = DBSCAN(eps=1000, min_samples=5)
# cluster_lst = mdl.fit_predict(D)

# visualizeTrajectory(traj_arr, cluster_lst)

api_res = []
color_lst = ['firebrick', 'olive', 'indigo', 'khaki', 'teal', 'saddlebrown','skyblue', 'coral', 'darkorange']
for i in range(0,len(traj_arr)):
    temp = {}
    temp['cluster_id'] = int(cluster_lst[i])
    temp['color'] = color_lst[int(cluster_lst[i])]
    temp['trajectory'] = traj_arr[i].tolist()
    api_res.append(temp)

api_res1 = []
for i in range(0,len(traj_arr)):
    temp = {}
    temp['trajectory'] = traj_arr[i].tolist()
    api_res1.append(temp)

from flask import Flask
app = Flask(__name__)
@app.route('/getclusters/', methods=['GET'])
def getClusters():
    res_dict = {}
    res_dict['data'] = api_res
    response = jsonify(res_dict)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/gettrajectories/', methods=['GET'])
def getTrajectories():
    res_dict = {}
    res_dict['data'] = api_res1
    
    response = jsonify(res_dict)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

app.run(host="0.0.0.0")
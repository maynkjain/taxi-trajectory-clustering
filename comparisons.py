import pandas as pd
import csv
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


def triangle_area(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = a - b
    ab_dist = np.linalg.norm(ab)

    cb = c - b
    cb_dist = np.linalg.norm(cb)
    # if ab_dist * cb_dist == 0:
    #     return 0
    fraction = np.dot(ab, cb) / (ab_dist * cb_dist)
    #print(fraction)
    if fraction > 1 or fraction < -1:
        fraction = 0
    theta = math.acos(fraction)
    return 0.5 * ab_dist * cb_dist * math.sin(theta)


def visvalingham_whyatt(points, **kwargs):
    """
    Visvalingham-Whyatt algorithm for polyline simplification

    Runs in  linear O(n) time

    Parameters:
        points(list): list of sequential points in the polyline

    Keyword Arguments:
        min_points(int):  Minimum number of points in polyline, defaults to 2
        inplace (bool):  Indicates if the input polyline should remove points from the input list, defaults to False

    Returns:
        list: A list of min_points of the simplified polyline
    """
    point_count = len(points)
    if not kwargs.get('inplace', False):
        new_points = list(points)
    else:
        new_points = points
    areas = [float('inf')]
    point_indexes = list(range(point_count -1))
    for i in range(1, point_count - 1):
        area = triangle_area(points[i - 1], points[i], points[i + 1])
        areas.append(area)

    min_points = kwargs.get('min_points', 2)
    while len(new_points) > min_points:
        smallest_effective_index = min(point_indexes, key=lambda i: areas[i])
        new_points.pop(smallest_effective_index)
        areas.pop(smallest_effective_index)
        point_count = len(new_points)
        point_indexes = list(range(point_count -1))
        # recompute area for point after previous_smallest_effective_index
        if smallest_effective_index > 1:
            areas[smallest_effective_index - 1] = triangle_area(new_points[smallest_effective_index - 2], new_points[smallest_effective_index - 1], new_points[smallest_effective_index])
        # recompute area for point before previous smallest_effective_index
        if smallest_effective_index < point_count - 1:
            areas[smallest_effective_index] = triangle_area(new_points[smallest_effective_index - 1], new_points[smallest_effective_index], new_points[smallest_effective_index + 1])
    return new_points

def visualizeTrajectory(traj_lst, cluster_lst):
    cluster_count = np.max(cluster_lst) + 1
    for traj, cluster in zip(traj_lst, cluster_lst):
        if cluster == -1:
            plt.plot(traj[:, 0], traj[:, 1], c='k', linestyle='dashed')
        else:
            plt.plot(traj[:, 0], traj[:, 1], c=color_lst[cluster % len(color_lst)])
    plt.show()

sns.set()
plt.rcParams['figure.figsize'] = (12, 12)
color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_lst.extend(['firebrick', 'olive', 'indigo', 'khaki', 'teal', 'saddlebrown', 
                 'skyblue', 'coral', 'darkorange'])

columns = ["POLYLINE"]
df = pd.read_csv("trajectory.csv", usecols=columns,nrows =200) #Reading only 20 lines to see comparison between original and rdp trajectory

traj_lst = []
for index,row in df.iterrows():
    temp = {}
    temp['trajectory'] = literal_eval(row[0])
    traj_lst.append(temp)
    
traj_arr = []

for it in traj_lst:
    arr = np.asarray(it['trajectory'])
    traj_arr.append(arr)

beforeRDP_traj_arr = traj_arr.copy()
beforeVW_traj_arr = traj_arr.copy()

degree_threshold = 30  #Can be modified to see variation in trajectory after rdp

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


rdp_traj = traj_arr
# for traj in rdp_traj:
#     print("After reduction length")
#     print(len(traj))
#     plt.plot(traj[:, 0], traj[:, 1])
# plt.show()

vw_points = []
for traj in beforeVW_traj_arr:
    #print("Intitial length")
    # print(len(traj))
    point_count = len(traj)
    #print(point_count)
    arr = np.asarray(visvalingham_whyatt(traj, min_points=max(2,point_count - 10)))
    #print("Final length")
    #print(len(arr))
    #print(arr)
    vw_points.append(arr)

def plotGraphForComparison(initial,afterRDP,afterVW):
    header = ['original points', 'after rdp points', 'after vw points']

    finaldata =[]
    for i in range(0, len(initial)):
        originalpoints = len(initial[i])
        afterRDPpoints = len(afterRDP[i])
        afterVWpoints = len(afterVW[i])
        data = [originalpoints, afterRDPpoints, afterVWpoints]
        finaldata.append(data)

    with open('comparison.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)
        for d in finaldata:
            writer.writerow(d)
        # write the data
        # writer.writerow(data)

plotGraphForComparison(beforeRDP_traj_arr,traj_arr,vw_points)
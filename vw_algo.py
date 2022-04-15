import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
from ast import literal_eval
def triangle_area(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = a - b
    ab_dist = np.linalg.norm(ab)

    cb = c - b
    cb_dist = np.linalg.norm(cb)
    fraction = np.dot(ab, cb) / (ab_dist * cb_dist)
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


if __name__ == '__main__':
    # points = [[-8.618643,41.141412],[-8.618499,41.141376],[-8.620326,41.14251],[-8.622153,41.143815],[-8.623953,41.144373],[-8.62668,41.144778],[-8.627373,41.144697],[-8.630226,41.14521],[-8.632746,41.14692],[-8.631738,41.148225],[-8.629938,41.150385],[-8.62911,41.151213],[-8.629128,41.15124],[-8.628786,41.152203],[-8.628687,41.152374],[-8.628759,41.152518],[-8.630838,41.15268],[-8.632323,41.153022],[-8.631144,41.154489],[-8.630829,41.154507],[-8.630829,41.154516],[-8.630829,41.154498],[-8.630838,41.154489]]

    # point_count = len(points)
    # new_points = visvalingham_whyatt(points, min_points=point_count - 5)
    # print(len(points))
    # print(len(new_points))

    columns = ["POLYLINE"]
    df = pd.read_csv("trajectory.csv", usecols=columns,nrows =40) #Reading only 20 lines to see comparison between original and rdp trajectory

    traj_lst = []
    for index,row in df.iterrows():
        temp = {}
        temp['trajectory'] = literal_eval(row[0])
        traj_lst.append(temp)
        
    traj_arr = []

    for it in traj_lst:
        arr = np.asarray(it['trajectory'])
        traj_arr.append(arr)

    new_points = []
    for traj in traj_arr:
        print("Intitial length")
        print(len(traj))
        point_count = len(traj)
        #print(point_count)
        arr = np.asarray(visvalingham_whyatt(traj, min_points=max(2,point_count - 10)))
        print("Final length")
        print(len(arr))
        #print(arr)
        new_points.append(arr)


    for traj in traj_arr:
        plt.plot(traj[:, 0], traj[:, 1])
    plt.show()

    for traj in new_points:
        plt.plot(traj[:, 0], traj[:, 1])
    plt.show()
#! /usr/bin/env python
# -*- coding: utf-8 -*-

#############################
# utils files.
#############################
from numpy import loadtxt, ndarray, min, max
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json
import pickle
import math


def load_dataset(filepath):
    """
        Return:
            dataset: dict
    """
    with open(filepath, 'rb') as fr:
        dataset = pickle.load(fr)
    return dataset


def sample_points_in_bin(bin_mid, total_points, quant_eps):
    """
        Input:
            bin_mid: numpy.array (d,)
            total_points: points needed to be generated
            quant_eps: quantization region length
    """
    sampled_shifts = np.random.uniform(-quant_eps / 2.0,
                                       quant_eps / 2.0,
                                       size=[total_points, bin_mid.size])
    sampled_points = sampled_shifts + bin_mid
    return sampled_points


def clustering_loss(data, centroids):
    """
        Computes the clustering loss on a dataset given a fixed set of centroids
        Input:
            centroids: numpy.array (k,d)
            data: numpy.array (n,d)
    """
    loss = 0.0
    for i_data in range(data.shape[0]):
        d = np.linalg.norm(data[i_data, :] - centroids, axis=1)
        loss += np.min(d)**2
    return loss


def induced_loss(data, centroids, assignments):
    """
        Compute the loss based on the induced clustering results
        Input:
            centroids: numpy.array (k,d)
            data: numpy.array (n,d)
            assignments: numpy.array (n,). Values are between [0,k-1]
    """
    loss = 0.0
    for i_data in range(data.shape[0]):
        d = np.linalg.norm(data[i_data, :] - centroids[assignments[i_data], :])
        loss += d**2
    return loss


def induced_loss_return_max(data, centroids, assignments):
    """
        Compute the loss based on the induced clustering results
        Input:
            centroids: numpy.array (k,d)
            data: numpy.array (n,d)
            assignments: numpy.array (n,). Values are between [0,k-1]
    """
    loss = 0.0
    argmax_idx = -1
    max_loss = -1
    for i_data in range(data.shape[0]):
        d = np.linalg.norm(data[i_data, :] - centroids[assignments[i_data], :])
        loss += d**2
        if d > max_loss:
            max_loss = d
            argmax_idx = i_data
    return loss, argmax_idx

def find_index(orgin,sub):
    index=[]
    number = len(list(orgin))
    for i in range(number):
        subnum=len(list(sub))
        for j in range(subnum):
            if list(sub[j])==list(orgin[i]):
                index.append(i)
    return index
def split_data(data_combined,# 所有数据
               num_clusters,# 最终类的个数
               num_clients=None,#客户端数量
               split='iid',
               k_prime=None):
    json_data = {}#保存所有客户端数据。
    # K-means optimal loss
    clf = KMeans(n_clusters=num_clusters).fit(data_combined)
    # 对kmeans进行聚类。
    kmeans_loss = clf.inertia_
    kmeans_label = clf.labels_# kmeans聚类结果。
    json_data['kmeans_loss'] = kmeans_loss

    if num_clients is None:
        num_clients = int(
            data_combined.shape[0] /
            100)  # make sure each client does not have too much data
        # 保障每个客户端不能有太多数据。num_clients 默认设置为每个客户端最大数据量为100

    # initialize for each client
    for i in range(num_clients):
        json_data['client_' + str(i)] = []# 每个客户端设置一个列表保存。

    # iid split
    if split == 'iid':
        for k in range(num_clusters):
            data_cluster = data_combined[kmeans_label == k, :]
            size_per_client = math.floor(data_cluster.shape[0] / num_clients)
            for i in range(num_clients - 1):
                json_data['client_' + str(i)].append(
                    data_cluster[i * size_per_client:(i + 1) *
                                 size_per_client, :])
            # fill the rest into the last client
            json_data['client_' + str(num_clients - 1)].append(
                data_cluster[(num_clients - 1) * size_per_client:, :])

        tmp_count = 0
        # concatenate the data for all clients
        for i in range(num_clients):
            json_data['client_' + str(i)] = np.concatenate(
                json_data['client_' + str(i)], axis=0)
            tmp_count += json_data['client_' + str(i)].shape[0]
        # have a final check on the sizes
        assert tmp_count == data_combined.shape[
            0], "Error: data size does not match"
    # non-iid split
    elif split == 'non-iid':
        if k_prime is None:
            k_prime = int(num_clusters / 2)
        assert k_prime <= num_clusters, "Error: not valid k_prime"
        print('k_prime',k_prime)
        # first get data for each cluster
        data_by_cluster = {}
        data_by_cluster_used = [0] * num_clusters
        print('data_by_cluster_used',data_by_cluster_used)
        size_per_client = int(data_combined.shape[0] / num_clients)# 每个客户端的数据量

        print('size_per_client',size_per_client)
        for k in range(num_clusters):
            data_by_cluster[k] = data_combined[kmeans_label == k, :]
            # 存储每个类的数据
        #print('data_by_cluster',data_by_cluster)

        valid_cluster_idx = [k for k in range(num_clusters)]
        print('valid_cluster_idx',valid_cluster_idx)
        # first fill in the data for first n-1 clients
        for i in range(num_clients - 1):
            print('+++++++++第',i,'次循环+++++++++++++++')
            tmp_client_data = []
            tmp_client_size = 0#当前客户端里面有多少数据点

            tmp_client_clusters = np.random.choice(valid_cluster_idx,
                                                   min([k_prime,len(valid_cluster_idx)]),
                                                   replace=False)
            print('tmp_client_clusters',tmp_client_clusters)
            # 任意选择两个类形成一个客户端。
            for tmp_client_cluster_idx in tmp_client_clusters:
                # some intermediate variables
                tmp_1 = data_by_cluster_used[tmp_client_cluster_idx]
                #tmp_1是当前类已经分配多少个数据点。
                #print('data_by_cluster[tmp_client_cluster_idx]',data_by_cluster[tmp_client_cluster_idx])
                tmp_2 = data_by_cluster[tmp_client_cluster_idx].shape[0]
                #tmp_2 是对应kmeans类中有多少数据点
                print('tmp_1',tmp_client_cluster_idx,'类，已经分配数据点个数',tmp_1)
                print('tmp_2当前类有数据点个数',tmp_2)
                if tmp_client_size < size_per_client and tmp_1 < tmp_2:
                    #如果当前的数据点少于规定数据点
                    tmp_count = min([
                        np.random.randint(
                            int(size_per_client / k_prime) - 1,
                            size_per_client),# 在一半客户端大小到客户端size之间随机一个整数
                        size_per_client - tmp_client_size, tmp_2 - tmp_1
                    ])
                    print(np.random.randint(
                            int(size_per_client / k_prime) - 1,
                            size_per_client))
                    print('size_per_client - tmp_client_size',size_per_client,'-',tmp_client_size,size_per_client - tmp_client_size)
                    print('tmp_2 - tmp_1',tmp_2,'-',tmp_1,tmp_2 - tmp_1)
                    print('tmp_count',tmp_count)
                    tmp_client_data.append(
                        data_by_cluster[tmp_client_cluster_idx][tmp_1:tmp_1 +
                                                                tmp_count, :])
                    # 从第tmp_1
                    print('从类中第',tmp_1,'行到第',tmp_1 +tmp_count,'行')
                    # 从对应的的2个类中拿出一部分数据存到tmp_client_data

                    # update each value
                    data_by_cluster_used[tmp_client_cluster_idx] += tmp_count
                    #每个kmean类用了多少数据点
                    print('data_by_cluster_used',data_by_cluster_used)
                    if data_by_cluster_used[tmp_client_cluster_idx] == tmp_2:
                        valid_cluster_idx.remove(
                            tmp_client_cluster_idx
                        )  # will not selected by future clients
                        #kmeans类里面的数据点用完了，就去除这个类，下次分配不再使用这个类中的点了。
                    tmp_client_size += tmp_count#更新当前客户端的点的数量。
                    if tmp_client_size == size_per_client:
                        break#如果

            json_data['client_' + str(i)] = np.concatenate(tmp_client_data,
                                                           axis=0)

        # leave all other data points to the last client
        cluster_size_last_client = 0
        tmp_client_data = []
        for k in range(num_clusters):
            if data_by_cluster_used[k] < data_by_cluster[k].shape[0]:
                tmp_client_data.append(
                    data_by_cluster[k][data_by_cluster_used[k]:, :])
                cluster_size_last_client += 1
        assert cluster_size_last_client <= k_prime, "Error: k_prime is violated"
        json_data['client_' + str(num_clients - 1)] = np.concatenate(
            tmp_client_data, axis=0)
        # have a final check on the sizes
        tmp_count = 0
        for i in range(num_clients):
            tmp_count += json_data['client_' + str(i)].shape[0]
        assert tmp_count == data_combined.shape[
            0], "Error: data size does not match"
    else:
        raise NotImplementedError

    return json_data
# 总的方法就是：每个客户端都从随机两个kmenas类中取出设定好数量的数据点，先随机一个数字，从第一个类中取
# 剩下的就从第二个类中取。
#基本上可以保障，每个客户端中的类别分布是不同的。



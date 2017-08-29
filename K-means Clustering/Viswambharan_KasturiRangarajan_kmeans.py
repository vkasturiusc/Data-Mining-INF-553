#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
The K-means algorithm written from scratch against PySpark. In practice,
one may prefer to use the KMeans algorithm in ML, as shown in
examples/src/main/python/ml/kmeans_example.py.

This example requires NumPy (http://www.numpy.org/).
"""
from __future__ import print_function

import sys
from collections import defaultdict
from scipy.sparse import csc_matrix,linalg,SparseEfficiencyWarning
import numpy as np
import math,warnings
from pyspark.sql import SparkSession


def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])


def closestPoint(p, centers):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = cosine_distance(p,centers[i])
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex

def cosine_distance(u, v):
    return 1.0 - u.dot(v.T)[0,0] / ((linalg.norm(u)) * (linalg.norm(v)))

if __name__ == "__main__":

    #if len(sys.argv) != 4:
       # print("Usage: kmeans <file> <k> <convergeDist>", file=sys.stderr)
       # exit(-1)

    #print("""WARN: This is a naive implementation of KMeans Clustering and is given
     #  as an example! Please refer to examples/src/main/python/ml/kmeans_example.py for an
      # example on how to use ML's KMeans implementation.""", file=sys.stderr)

    spark = SparkSession\
        .builder\
        .appName("PythonKMeans")\
        .getOrCreate()
    warnings.simplefilter('ignore',SparseEfficiencyWarning)
    file = open(sys.argv[1], 'r+')
    inputfile = file.readlines()
    doc_count = int(inputfile[0])
    vocabulary_count = int(inputfile[1])
    word_count = int(inputfile[2])
    input_lines = inputfile[3:]
    heap = []
    doc_list = defaultdict(list)
    word_list = defaultdict(list)
    euclid_sum=0
    euclid=0
    unit_vector = csc_matrix((doc_count,vocabulary_count+1))
    for records in input_lines:
        doc_id, word_id, tf = [int(y) for y in records.split(' ')]
        doc_list[doc_id-1] += [(word_id,tf)]
        word_list[word_id] += [doc_id-1]
    for doc in doc_list:
        for items in doc_list[doc]:
            idf = (doc_count + 1.0)/(len(word_list[items[0]])+1)
            tfidf = items[1]*math.log(idf,2)
            unit_vector[doc,items[0]] = tfidf
            euclid_sum += math.pow(tfidf,2)
        euclid = math.sqrt(euclid_sum)
        for items in doc_list[doc]:
            unit_vector[doc,items[0]] /= euclid
        euclid_sum=0
        euclid=0
    sc = spark.sparkContext
    data = [unit_vector[i,:] for i in xrange(doc_count)]
    data = sc.parallelize(data)
    K = int(sys.argv[2])
    convergeDist = float(sys.argv[3])

    kPoints = data.takeSample(False, K, 1)
    tempDist = 1.0

    while tempDist > convergeDist:
        closest = data.map(
            lambda p: (closestPoint(p, kPoints), (p, 1)))
        pointStats = closest.reduceByKey(
            lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
        newPoints = pointStats.map(
            lambda st: (st[0], st[1][0] / st[1][1])).collect()

        tempDist = sum((kPoints[iK] - p).multiply((kPoints[iK] - p)).sum() for (iK, p) in newPoints)

        for (iK, p) in newPoints:
            kPoints[iK] = p

    #print("Final centers: " + str(kPoints))
    file = open(sys.argv[4],'w+')
    for i in kPoints:
        file.write(str(i.getnnz())+'\n')
    
    


    spark.stop()

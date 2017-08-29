import math
from collections import defaultdict
import scipy
from scipy.sparse import csc_matrix,linalg,SparseEfficiencyWarning
import heapq
import itertools
import sys,warnings
import timeit

#start = timeit.default_timer()

def cosine_distance(u, v):
    return 1 - (u.dot(v.T)[0,0] / ((linalg.norm(u)) * (linalg.norm(v))))

file = open(sys.argv[1], 'r+')
inputfile = file.readlines()
doc_count = int(inputfile[0])
vocabulary_count = int(inputfile[1])
word_count = int(inputfile[2])
input_lines = inputfile[3:]   
heap = []
warnings.simplefilter('ignore',SparseEfficiencyWarning)
def unit_vector(input_lines,doc_count,vocabulary_count):
    
    doc_list = defaultdict(list)
    word_list = defaultdict(list)
    euclid_sum=0.0
    euclid=0.0
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
        euclid_sum=0.0
        euclid=0.0
    return unit_vector

unit_vector = unit_vector(input_lines,doc_count,vocabulary_count)
clusters_list = {str(x):0 for x in [x for x in range(doc_count)]}
doc_pairs = list(itertools.combinations([x for x in clusters_list.keys()],2))

for pair in doc_pairs:
    heapq.heappush(heap,(cosine_distance(unit_vector[int(pair[0])],unit_vector[int(pair[1])]),pair))
removed_keys = set()

k=int(sys.argv[2])
while(len(clusters_list.keys()) != k):
    flag = 0
    closest_pair = heapq.heappop(heap)
    best_cos = closest_pair[0]
    closest_pair = closest_pair[1]

    for x in closest_pair:
        if x in removed_keys:
            flag = 8
    if flag !=8:
        for key in closest_pair:
               clusters_list.pop(key)
               removed_keys.add(key)
        closest_pair_key = '#'.join(x for x in closest_pair)
        new_pairs = [(closest_pair_key,x) for x in clusters_list.keys()]
        clusters_list[closest_pair_key] = 0
        list_in_key = [int(x) for x in closest_pair_key.split('#')]

        centroid = csc_matrix((1,vocabulary_count+1))
        for x in sorted(list_in_key):
            centroid = scipy.sparse.vstack((centroid,unit_vector[x,:]),format="csc")

        for pair in new_pairs:
            list_in_pair = [int(x) for x in pair[1].split('#')]
            centroid_of_pair = csc_matrix((1,vocabulary_count+1))
            for x in sorted(list_in_pair):
                centroid_of_pair = scipy.sparse.vstack((centroid_of_pair, unit_vector[x,:]), format="csc")
            cosine_dist = cosine_distance(csc_matrix(centroid.mean(axis=0)),csc_matrix(centroid_of_pair.mean(axis=0)))
            heapq.heappush(heap,(cosine_dist,pair))
        
    
for keys in clusters_list.keys():
    print(str([int(x)+1 for x in keys.split('#')]).strip("[]"))

#stop  = timeit.default_timer()
#print("Time Elapsed: "+ str(stop - start))

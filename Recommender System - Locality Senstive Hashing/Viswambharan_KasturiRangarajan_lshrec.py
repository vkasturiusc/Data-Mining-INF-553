from pyspark import SparkContext
import sys
import itertools
def minhasher(list_items):
    minhash = []
    hashes = list(map(lambda x: [(3*x + 13*y)%100 for y in range(20)],list_items))
    for i in list(map(list,zip(*hashes))):
        minhash.append(min(i))
    return minhash

def jaccard(set1,set2):
    return 1.0*len(set(set1).intersection(set(set2)))/len(set(set1).union(set(set2)))

def top5users(list_users):
    return list_users[:5] if len(list_users) > 5 else list_users

sc = SparkContext(appName = "inf553")
text = sc.textFile(sys.argv[1]).map(lambda line: [int(y) for y in line.strip('U').split(',')])
minhash_values = text.map(lambda x: (x[0],minhasher(x[1:])))
band1_candidates = minhash_values.map(lambda x: [tuple(x[1][:4]),x[0]]).groupByKey().values().map(lambda x: tuple(x)).filter(lambda x: len(x) > 1)
band2_candidates = minhash_values.map(lambda x: [tuple(x[1][4:8]),x[0]]).groupByKey().values().map(lambda x: tuple(x)).filter(lambda x: len(x) > 1)
band3_candidates = minhash_values.map(lambda x: [tuple(x[1][8:12]),x[0]]).groupByKey().values().map(lambda x: tuple(x)).filter(lambda x: len(x) > 1)
band4_candidates = minhash_values.map(lambda x: [tuple(x[1][12:16]),x[0]]).groupByKey().values().map(lambda x: tuple(x)).filter(lambda x: len(x) > 1)
band5_candidates = minhash_values.map(lambda x: [tuple(x[1][16:]),x[0]]).groupByKey().values().map(lambda x: tuple(x)).filter(lambda x: len(x) > 1)
candidates = band1_candidates.union(band2_candidates).union(band3_candidates).union(band4_candidates).union(band5_candidates).distinct()
candidate_pairs = candidates.map(lambda x: list(itertools.permutations(x,2))).flatMap(lambda x:x).distinct()
text_checker = text.map(lambda line: line[1:]).collect()
jaccard_pairs = candidate_pairs.map(lambda (k,v): (k,v,jaccard(text_checker[k-1],text_checker[v-1])))
rdd_sorter_helper = jaccard_pairs.sortBy(lambda x: x[1],True).sortBy(lambda x: x[2],False)
top5 = rdd_sorter_helper.map(lambda (k,v,j): (k,[v])).reduceByKey(lambda a,b: a+b).map(lambda (k,v): (k,top5users(v))).sortByKey(True).map(lambda (k,v): (k,sorted(v)))
outval =  top5.map(lambda (k,v): (k,sorted(v))).map(lambda (k,v): 'U'+str(k)+":"+','.join(['U'+str(i) for i in v])).collect()

out_file = open(sys.argv[2],'w+')
for val in outval:
    out_file.write(val+'\n')
out_file.close()

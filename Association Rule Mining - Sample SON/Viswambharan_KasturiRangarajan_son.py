import sys

from pyspark import SparkContext
from collections import defaultdict
#------------------------------------------------------------------
# START OF APRIORI
#------------------------------------------------------------------
def candidates(itemSet, length):
    return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j))  == length])

def frequent_items(items,baskets,support):
    _itemSet = set()
    local_items = defaultdict(int)
    for item in items:
        for basket in baskets:
            if item.issubset(basket):
                local_items[item] +=1

    for item,count in local_items.items():
        if count >= support:
            _itemSet.add(item)
    return _itemSet


def frequent_itemsets(data,threshold):

    baskets = list()
    items = defaultdict(lambda: 0)
    f_itemsets = dict()
    L = set()


    for entry in data:
        basket = set(entry)
        baskets.append(basket)
        for item in basket:
            items[item]+=1
    
    support = threshold*len(baskets)    

    for item,count in items.iteritems():
        if count >= support:
            L.add(frozenset([item]))

    k=2
    f_itemsets[k-1] = L
    while L != set([]):
        L = candidates(L,k)
        C = frequent_items(L,baskets,support)
        if C != set([]):
            f_itemsets[k] = C
            L = C
        k+=1
    return f_itemsets

#       END OF APRIORI
#------------------------------------------------------------

fileName = str(sys.argv[1])
threshold = float(sys.argv[2])
outputFileName = str(sys.argv[3])

sc = SparkContext(appName="inf553")

baskets = sc.textFile(fileName).map(lambda line: set(sorted([int(y) for y in line.strip().split(',')])))
support = threshold*len(baskets.collect())
local_itemsets = baskets.mapPartitions(lambda chunk: [x for y in frequent_itemsets(chunk,threshold).values() for x in y],True)

itemsets = local_itemsets.map(lambda x: (x,1))

merged_itemsets = itemsets.reduceByKey(lambda x,y: x).map(lambda (x,y): x)
candidate_itemsets = merged_itemsets.collect()
counts = baskets.flatMap(lambda l: [(x,1) for x in candidate_itemsets if l.issuperset(x)])

final_itemsets = counts.reduceByKey(lambda x,y: x+y).filter(lambda (i,v): v>=support).map(lambda (x,y): x).collect()

out_file = open(outputFileName,'w')

for item in final_itemsets:
     out_file.write(','.join(str(i) for i in list(item))+"\n")

out_file.close()

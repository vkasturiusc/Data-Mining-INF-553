from pyspark import SparkContext
from operator import add
import sys
sc = SparkContext(appName="inf553")
A_lines = sc.textFile(sys.argv[1]).map(lambda line: [int(y) for y in line.strip().split(',')])
B_lines = sc.textFile(sys.argv[2]).map(lambda line: [int(y) for y in line.strip().split(',')])
A = A_lines.map(lambda x: (int(x[1]),('A',int(x[0]),int(x[2]))))
B = B_lines.map(lambda x: (int(x[0]),('B',int(x[1]),int(x[2]))))

phaseone = A.cartesian(B).filter(lambda (a,b): a[0]==b[0]).map(lambda (k1,k2): ((k1[1][1],k2[1][1]),k1[1][2]*k2[1][2] ))
phasetwo = phaseone.reduceByKey(add)
output = phasetwo.map(lambda (k, v): "" + str(k[0]) + "," + str(k[1]) + "\t" + str(v)).collect()

out_file = open(sys.argv[3],'w')
for v in output:
    out_file.write(v+'\n')
out_file.close()


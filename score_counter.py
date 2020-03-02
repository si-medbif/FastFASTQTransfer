import sys
from collections import defaultdict
from operator import itemgetter

total = 0

result = defaultdict(int)
for line in sys.stdin:
  data = line.strip()
  for c in data:
    result[c] += 1
    total += 1

# key = list(result.keys())
# key.sort()
scores = []
for i in sorted(result.keys()):
#   print(ord(i)-33, result[i]*100/total)
#  scores.append(result[i]*100/total)
  scores.append([i, result[i]*100/total])

print(scores)
print(sorted(scores))
print(sorted(scores,key=itemgetter(1)))
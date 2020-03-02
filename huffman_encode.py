import heapq
import os
import sys
from collections import defaultdict

def encode(frequency):
    heap = [[weight, [symbol, '']] for symbol, weight in frequency.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def get_frequency (data_path) :
    frequency_dict = dict()
    data_file = open(data_path, 'r')

    line = data_file.readline()

    while line != None and line != "" :
        line = line.strip()
        for score in line :
            if score not in frequency_dict :
                frequency_dict[score] = 0
            frequency_dict[score] += 1

        line = data_file.readline()

    data_file.close()
    return frequency_dict

def main (args) :
    # data = "FGFFFFGFFGFGFFFFGFGGEFFGFDFFFGF@FFFFFGF<FFFFFGGEFFFFGFGGFFGGGGFFFFGFFFGGFFFFFFGFFGGFFGFFEFGFFGFFGGGF"
    # frequency = defaultdict(int)

    # for current_char in data :
    #     frequency[current_char] += 1

    frequency = get_frequency(args[1])
    huff = encode(frequency)
    print (huff)

if __name__ == "__main__":
    main(sys.argv)
import math
from datetime import datetime
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from collections import OrderedDict

def convert(line):
    new_line = line.strip().split(',')
    new_line[2] = datetime.utcfromtimestamp(int(new_line[2])).strftime("%m/%d/%Y")
    return new_line

if __name__ == '__main__':
    data = []
    individual = []
    timestamp = []
    t = []
    raw_data = []

    for line in open("train.csv", 'r').readlines()[1:]:
        cl = convert(line)
        data.append(cl)

    #for line in open("test.txt", 'r').readlines():
    #    cl = convert(line)
    #    data.append(cl)

    for i in data:
        input = i[1]
        time = i[2]
        for j in individual:
            target = j[0]
            if input == target:
                flag = False
                for k in j[1:]:
                    if time == k[0]:
                        k[1]+=1
                        flag = True
                if flag == False:
                    j.append([time,1])
                individual.pop()
                break
        individual.append([input, [time,1]])
        for j in timestamp:
            target = j[0]
            if time == target:
                j[1]+=1
                timestamp.pop()
                break
        timestamp.append([time, 1])

    for i in data:
        year = i[2].split("/")[2]
        t.append(int(year))

    max = 0
    min = 10000000
    for i in individual:
        length = len(i)-1
        if length > max:
            max = length
        if length < min:
            min = length

    data = sorted(data, key=lambda sorted:sorted[2])

    for i in individual:
        i.append(len(i)-1)

    encounter = 0
    max_enc = 0
    min_enc = 9999
    max_p_per_enc = 0
    min_p_per_enc = 9999
    for i in individual:
        encounter+= i[-1]
        if i[-1] > max_enc:
            max_enc = i[-1]
        if i[-1] < min_enc:
            min_enc = i[-1]
        for j in i[1:-1]:
            if j[1] > max_p_per_enc:
                max_p_per_enc = j[1]
            if j[1] < min_p_per_enc:
                min_p_per_enc = j[1]

    x = []
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0
    six = 0
    for i in individual:
        if i[-1] == 1:
            one += 1
        if i[-1] == 2:
            two += 1
        if i[-1] == 3:
            three += 1
        if i[-1] == 4:
            four += 1
        if i[-1] == 5:
            five += 1
        if i[-1] == 6:
            six += 1
        if i[-1] != 1:
            x.append(i[-1])
    
    # Some output
    print(one, two, three, four, five, six)
    nd = []
    for i in x:
        if i not in nd:
            nd.append(i)
    print(len(nd))

    print()
    print("             Individuals per encounter              ")
    ind_per_enc = {}
    min_enc_ = 10
    sum_enc = 0
    sum_ind = 0
    for i in individual:
        enc = i[-1]
        if enc < min_enc_: 
            continue
        if enc in ind_per_enc:
          ind_per_enc[enc] += 1
        else:
          ind_per_enc[enc] = 1
        sum_enc += enc
        sum_ind += 1

    print("{} individuals have {} or more encounters".format(sum_ind, min_enc_))
    print("with a total of {} encounters.".format(sum_enc))
    
    print("\nThe distribution (# encounters, # individuals) looks like this: ")
    res = sorted(ind_per_enc.items(), key=lambda x: x[0])
    print(res)

    exit(0)

    num_bins = 81
    print(len(list(dict.fromkeys(x))))
    plt.xlabel('Encounter per individual')
    plt.ylabel('number of individual')
    n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.7)
    plt.show()

    num_bins = 20
    print(len(list(dict.fromkeys(x))))
    plt.xlabel('year')
    plt.ylabel('number of photos')
    n, bins, patches = plt.hist(sorted(t), num_bins, facecolor='blue', alpha=0.7)
    plt.show()

    
    # Statistics
    print("total number of photos: %d" % len(data))
    print("total number of individuals: %d" % len(individual))
    print("total number of encounters: %d\n" % encounter)

    print("average photos per individual: %.2f" % (len(data)/len(individual)))
    print("max photos per individual: %d" % max)
    print("min photo per individual: %d\n" % min)

    print("average encounters per individual: %.2f" % (encounter/len(individual)))
    print("max encounter per individual: %d" % max_enc)
    print("min encounter per individual: %d\n" % min_enc)

    print("average photos per encounter per individual: %.2f" %(len(data)/encounter))
    print("max photos per encounter per individual: %d" % max_p_per_enc)
    print("average photos per encounter per individual: %d" %min_p_per_enc)

    print("first day encounter: %s" % data[0][2])
    print("last day encounter: %s" % data[-1][2])
    print("average photo per day: %.2f" %((len(data))/5843))







"""
Code to extract some information about the data I am using.
For Computer Vision Research for Master's Thesis.

Written by: qiyang
Heavily modified by: Stephane Nouafo
There were sooo many wrong things..

Date notes added: 06/27/2020
"""

import math
from datetime import datetime
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from collections import OrderedDict


def convert(line):
    new_line = line.strip().split(',')
    new_line[2] = datetime.utcfromtimestamp(int(new_line[2])).strftime("%m/%d/%Y")
    return new_line[:3]

if __name__ == '__main__':
    data = []
    individual = []
    t = []
    
    # Get data
    for line in open("train.txt", 'r').readlines()[1:]:
        cl = convert(line)
        data.append(cl)
    
    if 1:
        for line in open("test.txt", 'r').readlines()[1:]:
            cl = convert(line)
            data.append(cl)

    # Gather data ([ID, [time,encounters], [time,enc], ...]) in individual for further processing
    for i in data:
        input = i[1]
        time = i[2]
        notFound = True
        
        for j in individual:
            target = j[0]
            # if the IDs are the same
            if input == target:
                notFound = False
                flag = False
                # Find if multiple photos are taken at the same time (same day)
                for k in j[1:]:
                    if time == k[0]:
                        k[1]+=1
                        flag = True
                if flag == False:
                    j.append([time,1])
                #individual.pop()
                break
        # Only add this ID if it doesn't exist already
        if notFound:
            individual.append([input, [time,1]])

        # Add the year data
        year = i[2].split("/")[2]
        # This one is obviously a mistake so just ignore it
        if year == '2038':
            continue
        t.append(int(year))


    # Some data on max/min encounters/individual
    max_ = 0
    min_ = 10000000
    encounter = 0
    # Data on max/min # of encounters/individuals and photos/encounters/individuals
    max_enc = 0
    min_enc = 9999
    max_p_per_enc = 0
    min_p_per_enc = 9999
    # Gathering the list of encounters/individual for distrbution and bar graphs 1 and 2
    x = []
    x1 = []
    x2 = []
    # Gathering data about # of encounter, individuals, and distribution
    ind_per_enc = {}
    min_enc_ = 0
    sum_enc = 0
    sum_ind = 0
    
    for i in individual:
        length = len(i)-1
        if length > max_:
            max_ = length
        if length < min_:
            min_ = length

        i.append(len(i)-1)
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

        x.append(i[-1])
        if i[-1] > 1:
          x1.append(i[-1])
        if i[-1] > 2:
          x2.append(i[-1])
    
        enc = i[-1]
        if enc < min_enc_: 
            print("This is not possible.")
            continue
        if enc in ind_per_enc:
          ind_per_enc[enc] += 1
        else:
          ind_per_enc[enc] = 1
        sum_enc += enc
        sum_ind += 1

    
    # Output information about our search
    print()
    print("             Individuals per encounter              ")
    print("{} individuals have {} or more encounters".format(sum_ind, min_enc_))
    print("with a total of {} encounters.".format(sum_enc))
    
    print("\nThe distribution (# encounters, # individuals) looks like this: ")
    res = sorted(ind_per_enc.items(), key=lambda x: x[0])
    print(res)
    #exit(0)

    if 0:
        # Excluding 1 for nice visuals
        num_bins = 50
        plt.xlabel('Encounter per individual')
        plt.ylabel('Number of individuals')
        n, bins, patches = plt.hist(x1, num_bins, facecolor='blue', alpha=0.7)
        plt.minorticks_on()
        plt.grid(which='minor')
        plt.title('# of Individuals with Specific # of Encounters (excluding 1)')
        plt.show()

        # Exclude 1 and 2 for nicer visuals
        num_bins = 50
        plt.xlabel('Encounter per individual')
        plt.ylabel('Number of individuals')
        n, bins, patches = plt.hist(x2, num_bins, facecolor='blue', alpha=0.7)
        plt.minorticks_on()
        plt.grid(which='minor')
        plt.title('# of Individuals with Specific # of Encounters (excluding 1 and 2)')
        plt.show() 
        #exit(0)


    # Graph with # of photos per year
    num_bins = 20
    #print(len(list(dict.fromkeys(x))))
    #print(min(t), " ",max(t))
    plt.xlabel('Year')
    plt.ylabel('Number of photos')
    n, bins, patches = plt.hist(sorted(t), num_bins, facecolor='blue', alpha=0.7)
    plt.show()

    
    # Sort data for some stats
    data = sorted(data, key=lambda x:x[2])
    #print(data)

    # Statistics
    print("\ntotal number of photos: %d" % len(data))
    print("total number of individuals: %d" % len(individual))
    print("total number of encounters: %d\n" % encounter)

    print("average encounters (photos) per individual: %.2f" % (len(data)/len(individual)))
    print("max encounters (photos) per individual: %d" % max_)
    print("min encounters (photos) per individual: %d\n" % min_)

    print("average photos per encounter per individual: %.2f" %(len(data)/encounter))
    print("max photos per encounter per individual: %d" % max_p_per_enc)
    print("min photos per encounter per individual: %d\n" %min_p_per_enc)

    # NOTE: Last day here is not correct. Sorts them alphabetically instead of by date
    print("first day encounter: %s" % data[0][2])
    print("last day encounter: %s" % data[-1][2])
    print("average photo per day: %.2f" %((len(data))/5843))







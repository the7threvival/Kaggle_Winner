# The purpose of this file is to create a train/valid split file given a set of 
# images and their IDs - 
# 
# NOTE 
# this does not attempt to ascertain images found in 
# the valid set have IDs that are also found in the training set.

import sys
import numpy as np

if __name__=="__main__":
  
  if(len(sys.argv) < 3):
    print("Usage: python {} input.csv output_name".format(sys.argv[0]))
    exit(1)
  
  images = open(sys.argv[1])
  train_file = open("train_split_"+sys.argv[2]+".csv", "w+")
  valid_file = open("valid_split_"+sys.argv[2]+".csv", "w+")
  try:
    sys.argv[3]
    visualize = True
  except:
    visualize = False
  
  formatted = []

  # Adds this line to the top of your file for pandas
  train_file.write("Image,Id\n")
  valid_file.write("Image,Id\n")
  # Skip the header line
  next(images)
  
  for image in images:
    # This assumes that the ID and name are in the first two positions of csv
    image_f = ",".join(image.split(",")[:2])
    formatted.append(image_f)
  
  if visualize: print(labels.items())
  split = len(formatted)*0.8
  
  # Shuffle the data
  np.random.shuffle(formatted)

  for ind, line in enumerate(formatted):
    if ind < split:
      train_file.write("{}\n".format(line))
    else:    
      valid_file.write("{}\n".format(line))

  images.close()
  train_file.close()
  valid_file.close()

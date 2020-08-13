# The purpose of this file is to create a train/valid/test split 
# given a set of images and their IDs. 
# 
# NOTES 
# - This version ascertains that non-new_whale images found in 
# the valid/test set have IDs that are also found in the training set.
# - You also have the option of specifying a number of minimum occurences
# for an ID to be included in the final split.


import sys
import numpy as np

if __name__=="__main__":
  
  # Argument Checker
  if(len(sys.argv) < 3):
    print()
    print("Usage: python {} input.csv input2.csv output_name [min_occur] [T]".format(sys.argv[0]))
    print("where the input .csv files are the input files to be split, output_name")
    print("will be the suffix of the 2 output files to be created, being") 
    print("train_split_<output_name> and valid...<output_name>, [min_occur]")
    print("is optional to specify a min number of occurences for an ID")
    print("to be included in the final split, and [T] is optional to ")
    print("visualize final labels chosen.")
    print()
    exit(1)
  
  in_file = sys.argv[1]
  in_file2 = sys.argv[2]
  images = open(in_file)
  images2 = open(in_file2)
  out1 = "train_split_"+sys.argv[3]+".csv"
  out2 = "valid_split_"+sys.argv[3]+".csv"
  out3 = "test_split_"+sys.argv[3]+".csv"
  train_file = open(out1, "w+")
  valid_file = open(out2, "w+")
  test_file = open(out3, "w+")
  # See if argument for minimum occurences was included
  try:
    min_occur = int(sys.argv[4])
  except:
    min_occur = 0

  # See if argument for visualizing labels was input
  try:
    sys.argv[5]
    visualize = True
    addition = ""
  except:
    visualize = False
    addition = "not "

  print("\nAccording to your configurations:")
  print("1) Data was read in from \"{}\" and \"{}\" and the labels will be split into".format(in_file, in_file2))
  print("   output files named \"{}\", \"{}\" and \"{}\".".format(out1, out2, out3))
  #print("2) Only IDs with at least \"{}\" occurences will be included.".format(min_occur))
  print("2) Only IDs with at least \"{}\" occurences will be included for validation and testing.".format(min_occur))
  print("3) You have also chosen to \"{}visualize\" the chosen data.".format(addition))
  cont = input("\nIs this what you want? (Press Enter for \"Yes\", anything else and Enter for \"No\"): ")
  if (cont): exit(0)

  labels = {}
  # Skip the header line
  next(images)
  next(images2)

  # File 1
  # Get the information and group images by ID
  for image in images:
    # This assumes that the name,ID of the images are in the first 
    # two positions/columns of the csv in this EXACT order
    image_f = image.strip().split(",")[:2]
    if (image_f[1] in labels): labels[image_f[1]].append(image_f[0])
    else: labels[image_f[1]] = [image_f[0]]
  
  # File 2
  # Get the information and group images by ID
  for image in images2:
    # This assumes that the name,ID of the images are in the first 
    # two positions/columns of the csv in this EXACT order
    image_f = image.strip().split(",")[:2]
    if (image_f[1] in labels): labels[image_f[1]].append(image_f[0])
    else: labels[image_f[1]] = [image_f[0]]
  
  # Adds this line to the top of your file as column names for pandas
  train_file.write("Image,Id\n")
  valid_file.write("Image,Id\n")
  test_file.write("Image,Id\n")
  
  num_train, num_trainN = 0, 0
  num_valid, num_validN = 0, 0
  num_test, num_testN = 0, 0
  if visualize: print("\nChosen labels:")
  # Go through the images by ID and split the valid ones across the files
  for id_, imgs in labels.items():
    if len(imgs) < min_occur:
      #continue
      #if len(imgs) == 3:
      for ind, img in enumerate(imgs):
        line = ",".join([img, id_])
        train_file.write("{}\n".format(line))
        num_train += 1
      continue

    # TODO: 
    """
    - Separate the IDs with 4 images into 2/1/1 and everything else do a N*0.7999 for split
    - DO NOT include images with less than min_occur in validation/testing but DO in training!!! Test this.. this would hopefully increase the accuracy...
    """
    if len(imgs) == min_occur:
      split = 2
      split2 = 3
    else:
      split = int(len(imgs)*0.7999)
      split2 = len(imgs) - (len(imgs)-split)*0.8999
      #split2 = split + (len(imgs)-split)/2
      # To help the 'new_whale' class be a more equal percentage of each set
      if id_ == 'new_whale':
        split2 = len(imgs) - (len(imgs)-split)*0.5999

    for ind, img in enumerate(imgs):
      line = ",".join([img, id_])
      if ind < split:
        train_file.write("{}\n".format(line))
        num_train += 1
        if id_=='new_whale': num_trainN += 1
      elif ind < split2:    
        valid_file.write("{}\n".format(line))
        num_valid += 1
        if id_=='new_whale': num_validN += 1
      else:    
        test_file.write("{}\n".format(line))
        num_test += 1
        if id_=='new_whale': num_testN += 1

    # Visualize if wanted
    if visualize: print(id_)
  
  # Output some statistics
  print()
  print("Total number of images seen:", num_train+num_valid+num_test)
  
  print()
  print("Number of images in train:", num_train)
  print("Number of 'new_whale' images in train:", num_trainN)
  perc = num_trainN/num_train
  print("Percentage of total images in train that are of 'new_whale's:", perc)
  
  print("Number of images in valid:", num_valid)
  print("Number of 'new_whale' images in valid:", num_validN)
  perc = num_validN/num_valid
  print("Percentage of total images in valid that are of 'new_whale's:", perc)
  
  print("Number of images in test:", num_test)
  print("Number of 'new_whale' images in test:", num_testN)
  perc = num_testN/num_test
  print("Percentage of total images in test that are of 'new_whale's:", perc)
  print()
  
  # Close streams and End program
  images.close()
  images2.close()
  train_file.close()
  valid_file.close()
  test_file.close()

#END

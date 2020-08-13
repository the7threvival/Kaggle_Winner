# The purpose of this file is to create a train/valid/test/embed split 
# given a set of images and their IDs. 
# 
# NOTES 
# - This combines the creating the 'new_whale's as well as splitting all the images
# for the folds in one file. 
# - This always keeps a separate set of images to be used to test our embedding
# skills later on.
# - This version ascertains that non-new_whale images found in 
# the valid/test set have IDs that are also found in the training set.
# - You also have the option of specifying a number of minimum occurences
# for an ID to be included in the final split.


import sys
import numpy as np

if __name__=="__main__":
  
  # Argument Checker
  if(len(sys.argv) < 4):
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
  out4 = "embed_split_"+sys.argv[3]+".csv"
  train_file = open(out1, "w+")
  valid_file = open(out2, "w+")
  test_file = open(out3, "w+")
  embed_file = open(out4, "w+")
  # See if argument for minimum occurences was included
  try:
    min_occur = int(sys.argv[4])
  except:
    #min_occur = 0
    min_occur = 4

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
  print("   output files named \"{}\", \"{}\", \n   \"{}\", and \"{}\".".format(out1, out2, out3, out4))
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
  embed_file.write("Image,Id\n")
  
  # Variables for statistics and splitting
  num_train, num_trainN = 0, 0
  num_valid, num_validN = 0, 0
  num_test, num_testN = 0, 0
  num_embed, num_embedN = 0, 0
  embed_split = 200  # # of individuals
  num_new = 0
  new = 'new_whale'
  new_labels = {new:[]}

  if visualize: print("\nChosen labels:")
  # Go through the images by ID and split the valid ones across the files
  for id_, imgs in labels.items():
    
    # With less than needed to include in validation/testing
    if len(imgs) < min_occur:
      # Those with 3 should just be included in train
      if len(imgs) == 3:
        for ind, img in enumerate(imgs):
          line = ",".join([img, id_])
          train_file.write("{}\n".format(line))
          num_train += 1

      # Those with 2 are special. 
      # Some will become 'new_whale's, others will be kept apart for embedding tests
      elif len(imgs) == 2:
        # Keep for embed tests
        if num_embed/2 < embed_split:
          for ind, img in enumerate(imgs):
            line = ",".join([img, id_])
            embed_file.write("{}\n".format(line))
            num_embed += 1
        # Store our 'new_whale's to split later. Just take the first of the two..
        else:
          new_labels[new].append(imgs[0])
           
      else:
        print("No individual should have only 1 encounter")
        print("Individual:", id_, "; # of enc.:", len(imgs))
      continue

    # Those with just enough to include in validation/testing
    if len(imgs) == min_occur:
      split = 2
      split2 = 3
    # Those with more than enough to include in validation/testing
    else:
      split = int(len(imgs)*0.7999)
      split2 = len(imgs) - (len(imgs)-split)*0.8999

    for ind, img in enumerate(imgs):
      line = ",".join([img, id_])
      if ind < split:
        train_file.write("{}\n".format(line))
        num_train += 1
      elif ind < split2:    
        valid_file.write("{}\n".format(line))
        num_valid += 1
      else:    
        test_file.write("{}\n".format(line))
        num_test += 1

    # Visualize if wanted
    if visualize: print(id_)

  # Split the 'new_whale's
  # Try to get the 'new_whale' class to be a more equal percentage of each set
  id_, imgs = new, new_labels[new]
  split = int(len(imgs)*0.7999)
  split2 = len(imgs) - (len(imgs)-split)*0.5799
  for ind, img in enumerate(imgs):
    line = ",".join([img, id_])
    if ind < split:
      train_file.write("{}\n".format(line))
      num_train += 1
      num_trainN += 1
    elif ind < split2:    
      valid_file.write("{}\n".format(line))
      num_valid += 1
      num_validN += 1
    else:    
      test_file.write("{}\n".format(line))
      num_test += 1
      num_testN += 1
    num_new += 1
  if visualize: print(new)
  
  # Output some statistics
  print()
  total = num_train + num_valid + num_test + num_embed + num_new
  total2 = total - num_new
  print("Total number of images seen:", total)
  print("Total number of images included:", total2)
  print("Number of images excluded:", num_new)
  
  print()
  print("Number of images in train:", num_train)
  print("Number of 'new_whale' images in train:", num_trainN)
  perc = num_trainN/num_train
  print("Percentage of total images in train that are of 'new_whale's:", perc)
  print("Percentage of train vs total included:", num_train/total2)
  
  print()
  print("Number of images in valid:", num_valid)
  print("Number of 'new_whale' images in valid:", num_validN)
  perc = num_validN/num_valid
  print("Percentage of total images in valid that are of 'new_whale's:", perc)
  print("Percentage of valid vs total included:", num_valid/total2)
  
  print()
  print("Number of images in test:", num_test)
  print("Number of 'new_whale' images in test:", num_testN)
  perc = num_testN/num_test
  print("Percentage of total images in test that are of 'new_whale's:", perc)
  print("Percentage of test vs total included:", num_test/total2)
  
  print()
  print("Number of images in embed:", num_embed)
  print("Number of 'new_whale' images in embed:", num_embedN)
  perc = num_embedN/num_embed
  print("Percentage of total images in embed that are of 'new_whale's:", perc)
  print("Percentage of embed vs total included:", num_embed/total2)
  print()
  
  # Close streams and End program
  images.close()
  images2.close()
  train_file.close()
  valid_file.close()
  test_file.close()
  embed_file.close()

#END

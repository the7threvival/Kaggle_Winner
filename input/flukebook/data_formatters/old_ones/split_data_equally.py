# The purpose of this file is to create a train/valid split file given a set of 
# images and their IDs. 
# 
# NOTES 
# - This version ascertains that images found in 
# the valid set have IDs that are also found in the training set.
# - You also have the option of specifying a number of minimum occurences
# for an ID to be included in the final split.


import sys
import numpy as np

if __name__=="__main__":
  
  # Argument Checker
  if(len(sys.argv) < 3):
    print()
    print("Usage: python {} input.csv output_name [min_occur] [T]".format(sys.argv[0]))
    print("where input.csv is the input file to be split, output_name")
    print("will be the suffix of the 2 output files to be created, being") 
    print("train_split_<output_name> and valid...<output_name>, [min_occur]")
    print("is optional to specify a min number of occurences for an ID")
    print("to be included in the final split, and [T] is optional to ")
    print("visualize final labels chosen.")
    print()
    exit(1)
  
  in_file = sys.argv[1]
  images = open(in_file)
  out1 = "train_split_"+sys.argv[2]+".csv"
  out2 = "valid_split_"+sys.argv[2]+".csv"
  train_file = open(out1, "w+")
  valid_file = open(out2, "w+")
  # See if argument for minimum occurences was included
  try:
    min_occur = int(sys.argv[3])
  except:
    min_occur = 0

  # See if argument for visualizing labels was input
  try:
    sys.argv[4]
    visualize = True
    addition = ""
  except:
    visualize = False
    addition = "not "

  print("\nAccording to your configurations:")
  print("1) Data was read in from \"{}\" and the labels will be split into".format(in_file))
  print("   output files named \"{}\" and \"{}\".".format(out1, out2))
  print("2) IDs with at least \"{}\" occurences will be chosen.".format(min_occur))
  print("3) You have also chosen to \"{}visualize\" the chosen data.".format(addition))
  cont = input("\nIs this what you want? (Press Enter for \"Yes\", anything else and Enter for \"No\"): ")
  if (cont): exit(0)

  labels = {}
  # Skip the header line
  next(images)

  # Get the information and group images by ID
  for image in images:
    # This assumes that the name,ID of the images are in the first 
    # two positions/columns of the csv in this EXACT order
    image_f = image.strip().split(",")[:2]
    if (image_f[1] in labels): labels[image_f[1]].append(image_f[0])
    else: labels[image_f[1]] = [image_f[0]]
  
  # Adds this line to the top of your file as column names for pandas
  train_file.write("Image,Id\n")
  valid_file.write("Image,Id\n")
  
  if visualize: print("\nChosen labels:")
  # Go through the images by ID and split the valid ones across the files
  for id_, imgs in labels.items():
    if len(imgs) < min_occur:
      continue

    # Split this class a lot more
    # to get 65 out of 2066 in testing
    if id_ == 'new_whale':
      split = int(len(imgs)*0.969)
    # 80% train, 20% valid
    # Testing out 90% to see how it impacts accuracy..
    else:
      split = int(len(imgs)*0.9)
    
    for ind, img in enumerate(imgs):
      line = ",".join([img, id_])
      if ind < split:
        train_file.write("{}\n".format(line))
      else:    
        valid_file.write("{}\n".format(line))

    # Visualize if wanted
    if visualize: print(id_)
  
  # Close streams and End program
  images.close()
  train_file.close()
  valid_file.close()

#END

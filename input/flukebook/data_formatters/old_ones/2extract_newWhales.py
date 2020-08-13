# The purpose of this file is to extract the individuals with only 1 occurence
# given a set of images and their IDs. 
# 
# NOTES 


import sys
import numpy as np

if __name__=="__main__":
  
  # Argument Checker
  if(len(sys.argv) < 3):
    print()
    print("Usage: python {} input.csv input2.csv output_name [skip]".format(sys.argv[0]))
    print("where input/2.csv are the input files to be split, output_name")
    print("will be the name of the output file to be created,")
    print(" and [skip] is optional to skip 'new_whale's.")
    print()
    exit(1)
  
  # Read input data
  in_file = sys.argv[1]
  in_file2 = sys.argv[2]
  out = sys.argv[3]+".csv"
  occur = 2

  # See if argument for [skip] was input
  try:
    if sys.argv[4]: skip = True
  except:
    skip = False

  print("\nAccording to your configurations:")
  print("1) Data will be read in from \"{}\" and \"{}\" and the output file will be called \"{}\"".format(in_file, in_file2, out))
  if skip:
    print("2) Only IDs with at least \"{}\" occurence(s) will be included.".format(occur+1))
  else:
    print("2) IDs with \"{}\" occurence(s) will be called 'new_whale' and have 1 of their occurences dropped.".format(occur))
  cont = input("\nIs this what you want? (Press Enter for \"Yes\", anything else and Enter for \"No\"): ")
  if (cont): exit(0)

  # Set up file variables
  images = open(in_file)
  images2 = open(in_file2)
  out_file = open(out, "w+")
  
  labels = {}
  # Skip the header line
  next(images)
  next(images2)

  # Get the information and group images by ID
  for image in images:
    # This assumes that the name,ID of the images are in the first 
    # two positions/columns of the csv in this EXACT order
    image_f = image.strip().split(",")[:2]
    if (image_f[1] in labels): labels[image_f[1]].append(image_f[0])
    else: labels[image_f[1]] = [image_f[0]]
  for image in images2:
    image_f = image.strip().split(",")[:2]
    if (image_f[1] in labels): labels[image_f[1]].append(image_f[0])
    else: labels[image_f[1]] = [image_f[0]]
 
  visualize = False 
  # Visualize if wanted
  if visualize: print(labels.items())
  
  # Adds this line to the top of your file as column names for pandas
  out_file.write("Image,Id\n")
  
  newName = 'new_whale'
  #i=0
  # Go through the images by ID and reassign the valid ones to 'new_whale'
  for id_, imgs in labels.items():
    # If we don't want to modify these, just add them
    if len(imgs) > occur:
      for img in imgs:
        line = ",".join([img, id_])
        out_file.write("{}\n".format(line))
    elif skip:
      continue
    elif len(imgs) == occur:
      #print(imgs[0])
      #i+=1
      line = ",".join([imgs[0], newName])
      out_file.write("{}\n".format(line))
    else:
      print("An error has occured. No ID should have less than {} occurences".format(occur))

  #print("{} new_whales added".format(i))
  # Close streams and End program
  images.close()
  images2.close()
  out_file.close()

#END

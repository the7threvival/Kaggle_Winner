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
    print("Usage: python {} input.csv output_name [T]".format(sys.argv[0]))
    print("where input.csv is the input file to be split, output_name")
    print("will be the name of the output file to be created,")
    print(" and [T] is optional to {}.")
    print("")
    print()
    exit(1)
  
  # Read input data
  in_file = sys.argv[1]
  out = sys.argv[2]+".csv"
  occur = 1

  # See if argument for {} was input
  

  print("\nAccording to your configurations:")
  print("1) Data was read in from \"{}\" and the new file will be called \"{}\"".format(in_file, out))
  print("2) IDs with at most \"{}\" occurence(s) will be chosen.".format(occur))
  cont = input("\nIs this what you want? (Press Enter for \"Yes\", anything else and Enter for \"No\"): ")
  if (cont): exit(0)

  # Set up file variables
  images = open(in_file)
  out_file = open(out, "w+")
  
  labels = {}
  # Skip the header line
  next(images)

  # Get the information and group images by ID
  for image in images:
    # This assumes that the name,ID of the images are in the first 
    # two positions/columns of the csv in this EXACT order
    image_f = image.split(",")[:2]
    if (image_f[1] in labels): labels[image_f[1]].append(image_f[0])
    else: labels[image_f[1]] = [image_f[0]]
 
  visualize = False 
  # Visualize if wanted
  if visualize: print(labels.items())
  
  # Adds this line to the top of your file as column names for pandas
  out_file.write("Image,Id\n")
  
  newName = 'new_whale'
  # Go through the images by ID and reassign the valid ones to 'new_whale'
  for id_, imgs in labels.items():
    # If we don't want to modify these, just add them
    if len(imgs) > occur:
      for img in imgs:
        line = ",".join([img, id_])
        out_file.write("{}\n".format(line))
    else:
      for img in imgs:
        line = ",".join([img, newName])
        out_file.write("{}\n".format(line))

  # Close streams and End program
  images.close()
  out_file.close()

#END

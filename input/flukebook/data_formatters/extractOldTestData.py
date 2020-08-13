# The purpose of this file is to extract the individuals that are not 'new_whale's
# for testing of the old models.
# 
# NOTES 


import sys
import numpy as np

if __name__=="__main__":
  
  # Argument Checker
  if(len(sys.argv) < 5):
    print()
    print("Usage: python {} input.csv input2.csv output_name occur".format(sys.argv[0]))
    print("where input.csv is the input file the 'new_whale' IDs will be extracted") 
    print("from, input2.csv is the input file to be split, output_name.csv")
    print("will be the name of the output file to be created, and occur")
    print("is the minimum number of occurences needed to be included.")
    print()
    exit(1)
  
  # Read input data
  in_file = sys.argv[1]
  in_file2 = sys.argv[2]
  out = sys.argv[3]+".csv"
  occur = int(sys.argv[4])

  print("\nAccording to your configurations:")
  print("1) Data was read in from \"{}\" and \"{}\" and the new file will be called \"{}\"".format(in_file, in_file2, out))
  print("2) Only images with at least {} occurences will be included.".format(occur))
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
 
  # Adds this line to the top of your file as column names for pandas
  out_file.write("Image,Id\n")
  
  # Go through the second set of images by ID and exclude the 'new_whale' ones
  for image in images2:
    image_f = image.strip().split(",")[:2]
    if len(labels[image_f[1]]) < occur:
      continue
    else:
      line = ",".join(image_f)
      out_file.write("{}\n".format(line))

  # Close streams and End program
  images.close()
  images2.close()
  out_file.close()

#END

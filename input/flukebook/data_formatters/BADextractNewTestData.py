# The purpose of this file is to extract the test data for newWhale3 split
# as this was corrupted as I modified train.csv and test.csv in 3newWhale instead
# of making new folders for the "final" split.
# 
# NOTES 
# Says BAD... so do not use I guess. Do not remember why it was bad though


import sys
import numpy as np

if __name__=="__main__":
  
  # Argument Checker
  if(len(sys.argv) < 6):
    print()
    print("Usage: python {} input1 input2 input3 input4 output_name".format(sys.argv[0]))
    print("where input1 and input2 are the input files train.csv and test.csv,") 
    print("input3 and input4 are the split files to be used to figure out what")
    print("should not be in the test file, and output_name.csv")
    print("will be the name of the output file to be created.")
    print()
    exit(1)
  
  # Read input data
  in_file = sys.argv[1]
  in_file2 = sys.argv[2]
  in_file3 = sys.argv[3]
  in_file4 = sys.argv[4]
  out = sys.argv[5]+".csv"

  print("\nAccording to your configurations:")
  print("1) Data was read in from \"{}\", \"{}\",\n   \"{}\", and \"{}\" and the new file will be called \"{}\"".format(in_file, in_file2, in_file3, in_file4, out))
  print("Should be train.csv test.csv train_split.csv split_valid.cvs")
  cont = input("\nIs this what you want? (Press Enter for \"Yes\", anything else and Enter for \"No\"): ")
  if (cont): exit(0)

  # Set up file variables
  images = open(in_file)
  images2 = open(in_file2)
  images3 = open(in_file3)
  images4 = open(in_file4)
  out_file = open(out, "w+")
  
  labels = set()
  # Skip the header line
  next(images)
  next(images2)
  next(images3)
  next(images4)

  # Get the names to exclude from the split files
  for image in images3:
    # This assumes that the name,ID of the images are in the first 
    # two positions/columns of the csv in this EXACT order
    image_f = image.strip().split(",")[:2]
    if image_f == [""]: continue
    if (image_f[0] in labels): 
      print("Something went wrong. There shouldn't be this name already...")
      print(image_f[0])
    else: labels.add(image_f[0])
  for image in images4:
    image_f = image.strip().split(",")[:2]
    if image_f == [""]: continue
    if (image_f[0] in labels): 
      print("Something went wrong. There shouldn't be this name already...")
      print(image_f[0])
    else: labels.add(image_f[0])
 
  # Adds this line to the top of your file as column names for pandas
  out_file.write("Image,Id\n")
  
  # Go through the first and second set of images by name and exclude the repeated ones
  for image in images:
    image_f = image.strip().split(",")[:2]
    if image_f[0] in labels:
      continue
    else:
      line = ",".join(image_f)
      out_file.write("{}\n".format(line))
  for image in images2:
    image_f = image.strip().split(",")[:2]
    if image_f[0] in labels:
      continue
    else:
      line = ",".join(image_f)
      out_file.write("{}\n".format(line))

  # Close streams and End program
  images.close()
  images2.close()
  images3.close()
  images4.close()
  out_file.close()

#END

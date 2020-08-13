# The purpose of this file is to create a train/test split of the images
# given how they should be split by their names. 


import os
import sys
import shutil
import numpy as np

if __name__=="__main__":
  
  # Argument Checker
  if(len(sys.argv) < 4):
    print()
    print("Usage: python {} input.csv input2.csv input3.csv".format(sys.argv[0]))
    print("where the input .csv files are the input files to be split.")
    print("These should be train, valid, then test .csv files.")
    print()
    exit(1)
  
  in_file = sys.argv[1]
  in_file2 = sys.argv[2]
  in_file3 = sys.argv[3]
  images = open(in_file)
  images2 = open(in_file2)
  images3 = open(in_file3)
  out1 = "train.csv"
  out2 = "test.csv"
  train_file = open(out1, "w+")
  test_file = open(out2, "w+")

  print("\nAccording to your configurations:")
  print("1) Data was read in from \"{}\", \"{}\" and \"{}\" and the labels will be split into".format(in_file, in_file2, in_file3))
  print("   output files named \"{}\" and \"{}\".".format(out1, out2))
  print("   The images will also be split accordingly from folders \"train2\" and.")
  print("   \"test2\" into folders named train and test.")
  cont = input("\nIs this what you want? (Press Enter for \"Yes\", anything else and Enter for \"No\"): ")
  if (cont): exit(0)

  
  # Adds this line to the top of your file as column names for pandas
  train_file.write("Image,Id\n")
  test_file.write("Image,Id\n")

  names_train = set()
  names_test = set()
  # Skip the header line
  next(images)
  next(images2)
  next(images3)

  # File 1 - train
  # Get the information and group images in their correct output file
  for image in images:
    # This assumes that the name,ID of the images are in the first 
    # two positions/columns of the csv in this EXACT order
    image_f = image.strip().split(",")[:2]
    names_train.add(image_f[0]) 
    train_file.write(",".join(image_f)+"\n")
  for image in images2:
    image_f = image.strip().split(",")[:2]
    names_train.add(image_f[0]) 
    train_file.write(",".join(image_f)+"\n")

  # File 2 - test
  for image in images3: 
    image_f = image.strip().split(",")[:2]
    names_test.add(image_f[0])
    test_file.write(",".join(image_f)+"\n")
     
  num_train = 0
  num_test = 0
  num_unused = 0
  names = ["/train2/", "/test2/"]

  outs = ["train", "test", "unused"]
  for out in outs:
    try: os.mkdir(out)
    except(FileExistsError): pass
  # Go through the images by ID and split them correctly across the files
  for name in names:
    path = os.getcwd()+name
    files = os.listdir(path)
    #files.sort()
    for f in files:
      src = path+f
      if f in names_train:
        moveto = os.getcwd()+"/train/"
        num_train += 1
      elif f in names_test:
        moveto = os.getcwd()+"/test/"
        num_test += 1
      else:
        #print("Name {} does not exist in either file. Error.".format(f))
        moveto = os.getcwd()+"/unused/"
        num_unused += 1
      dst = moveto+f
      shutil.move(src,dst)
     
  print()
  # Output some statistics
  print()
  print("Total number of images seen:", num_train+num_test+num_unused)
  print("Number of images in train:", num_train)
  print("Number of images in test:", num_test)
  print("Total number of images to be used:", num_train+num_test)
  print("Number of images unused:", num_unused)
  print()

  # Close streams and End program
  images.close()
  images2.close()
  images3.close()
  train_file.close()
  test_file.close()

#END

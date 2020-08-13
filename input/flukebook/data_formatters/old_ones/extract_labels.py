# The purpose of this file is to create a labels file given a set of images
# and their IDs

import sys

if __name__=="__main__":
  
  if(len(sys.argv) < 3):
    print("Usage: python {} input.csv output.csv".format(sys.argv[0]))
    exit(1)
  
  images = open(sys.argv[1])
  output_file = open(sys.argv[2], "w+")
  try:
    sys.argv[3]
    visualize = True
  except:
    visualize = False
  
  labels = {}

  # Adds this line to the top of your file for pandas
  output_file.write("id,name\n")
  count = 0
  next(images)
  for image in images:
    # This assumes that the ID is in the second position and file is a csv
    label = image.split(",")[1]
    if label not in labels:
      labels[label] = count
      count += 1
  
  if visualize: print(labels.items())
  for label, key in labels.items():
    output_file.write("{},{}\n".format(key, label))    

  images.close()
  output_file.close()

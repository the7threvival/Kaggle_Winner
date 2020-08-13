# The purpose of this file is to create a labels file given a set of images
# and their IDs

import sys

if __name__=="__main__":
  
  if(len(sys.argv) < 4):
    print("Usage: python {} input.csv input2.csv output.csv".format(sys.argv[0]))
    exit(1)
  
  images = open(sys.argv[1])
  images2 = open(sys.argv[2])
  output_file = open(sys.argv[3], "w+")
  try:
    sys.argv[4]
    visualize = True
  except:
    visualize = False
  
  labels = {}

  # Adds this line to the top of your file for pandas
  output_file.write("id,name\n")
  count = 0
  # Add 'new_whale' separately as it is not in the files I am using
  labels['new_whale'] = count
  count += 1
  
  # Skip header line
  next(images)
  for image in images:
    # This assumes that the ID is in the second position and file is a csv
    label = image.split(",")[1]
    if label not in labels:
      labels[label] = count
      count += 1
  next(images2)
  for image in images2:
    # This assumes that the ID is in the second position and file is a csv
    label = image.split(",")[1]
    if label not in labels:
      labels[label] = count
      count += 1
  
  if visualize: print(labels.items())
  for label, key in labels.items():
    output_file.write("{},{}\n".format(key, label))    

  images.close()
  images2.close()
  output_file.close()

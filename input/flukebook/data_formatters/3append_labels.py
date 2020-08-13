# The purpose of this file is to create a labels file given an old set of
# labels and a new set of labels. The old set's name ordering must be 
# maintained. The remaining new names will just be appended to the end.

import sys

if __name__=="__main__":
  
  if(len(sys.argv) < 4):
    print("Usage: python {} old_labels.csv new_labels.csv labels.csv".format(sys.argv[0]))
    exit(1)
  
  images = open(sys.argv[1])
  images2 = open(sys.argv[2])
  output_file = open(sys.argv[3], "w+")
  try:
    sys.argv[4]
    visualize = True
  except:
    visualize = False
  
  labels = set()

  # Adds this line to the top of your file for pandas
  output_file.write("id,name\n")
  count = 0
  
  # Skip header line
  next(images)
  for image in images:
    # This assumes that the ID is in the second position and file is a csv
    label = image.strip().split(",")[1]
    if label not in labels:
      labels.add(label)
    else:
      print("Error. These should all be unique labels from the old file.")
      exit()
    output_file.write("{},{}\n".format(count, label))  
    count += 1
      
  next(images2)
  for image in images2:
    # This assumes that the ID is in the second position and file is a csv
    label = image.strip().split(",")[1]
    if label not in labels:
      labels.add(label)
      output_file.write("{},{}\n".format(count, label))    
      count += 1
  
  if visualize: print(labels)
  
  images.close()
  images2.close()
  output_file.close()

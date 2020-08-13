# The purpose of this file is to create a bounding box file given a set of images
# and their bounding boxes' starting x and y, and width and heigth

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
  output_file.write("Image,x0,y0,x1,y1\n")
  # Skip header line
  next(images)
  for image_l in images:
    # This assumes that the name is in the first position and file is a csv
    info = image_l.split(",")
    image = info[0]
    # Image info should be exactly in these positions
    x0 = info[3]
    y0 = info[4]
    x1 = int(x0) + int(info[5])
    y1 = int(y0) + int(info[6])
    output_file.write("{},{},{},{},{}\n".format(image,x0,y0,x1,y1))    
  
  next(images2)
  for image_l in images2:
    # This assumes that the name is in the first position and file is a csv
    info = image_l.split(",")
    image = info[0]
    # Image info should be exactly in these positions
    x0 = info[3]
    y0 = info[4]
    x1 = int(x0) + int(info[5])
    y1 = int(y0) + int(info[6])
    output_file.write("{},{},{},{},{}\n".format(image,x0,y0,x1,y1))    
  
  if visualize: print("No visualization, see output file")

  images.close()
  images2.close()
  output_file.close()

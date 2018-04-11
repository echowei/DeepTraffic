# Wei Wang (ww8137@mail.ustc.edu.cn)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file, You
# can obtain one at http://mozilla.org/MPL/2.0/.
# ==============================================================================

import os
import errno
from PIL import Image
from array import *
from random import shuffle

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# Load from and save to
mkdir_p('5_Mnist')
Names = [['4_Png\Train','5_Mnist\\train']]

for name in Names:	
	data_image = array('B')
	data_label = array('B')

	FileList = []
	for dirname in os.listdir(name[0]): 
		path = os.path.join(name[0],dirname)
		for filename in os.listdir(path):
			if filename.endswith(".png"):
				FileList.append(os.path.join(name[0],dirname,filename))

	shuffle(FileList) # Usefull for further segmenting the validation set

	for filename in FileList:
		print filename
		label = int(filename.split('\\')[2])
		Im = Image.open(filename)
		pixel = Im.load()
		width, height = Im.size
		for x in range(0,width):
			for y in range(0,height):
				data_image.append(pixel[y,x])
		data_label.append(label) # labels start (one unsigned byte each)
	hexval = "{0:#0{1}x}".format(len(FileList),6) # number of files in HEX
	hexval = '0x' + hexval[2:].zfill(8)
	
	# header for label array
	header = array('B')
	header.extend([0,0,8,1])
	header.append(int('0x'+hexval[2:][0:2],16))
	header.append(int('0x'+hexval[2:][2:4],16))
	header.append(int('0x'+hexval[2:][4:6],16))
	header.append(int('0x'+hexval[2:][6:8],16))	
	data_label = header + data_label

	# additional header for images array	
	if max([width,height]) <= 256:
		header.extend([0,0,0,width,0,0,0,height])
	else:
		raise ValueError('Image exceeds maximum size: 256x256 pixels');

	header[3] = 3 # Changing MSB for image data (0x00000803)	
	data_image = header + data_image
	output_file = open(name[1]+'-images-idx3-ubyte', 'wb')
	data_image.tofile(output_file)
	output_file.close()
	output_file = open(name[1]+'-labels-idx1-ubyte', 'wb')
	data_label.tofile(output_file)
	output_file.close()

# gzip resulting files
for name in Names:
	os.system('gzip '+name[1]+'-images-idx3-ubyte')
	os.system('gzip '+name[1]+'-labels-idx1-ubyte')

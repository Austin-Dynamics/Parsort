import os
from os import listdir
from os.path import isfile, join
from tkinter import Tk  # For preventing Tk GUI from opening
from tkinter.filedialog import askdirectory
import numpy as np
import cv2
from tqdm import trange
import shutil


#start and end of the vertical read chunk on images
start = 0.3
end = 0.5
margin = 0.2

brightbasket = 0.35
darkbasket = 0.35

#contrast sensitivity variable
senso = 0.23 #how much of the bright and dark of an image chunk should be averaged to get the contrast score
sensotoo = 0.15 #How much of the high-contrast image pool should go into the folder

#Get list of files
Tk().withdraw() #Stop Tk GUI from opening, aesthetic improvement
rootpath = askdirectory()
if os.path.exists(rootpath):
    print("This is a real path")
else:
    print("This is not a real path.")
    exit()

#Check if deposit folders exist, and make them if not
createfolders = ["Dark","High Contrast","Bright"]
for folder in createfolders:
    try:
        os.mkdir(join(rootpath, folder))
    except OSError:
        print("Folder ", folder, " already exists or cannot be made here.")
    else:
        print(folder, "made.")


onlyfiles = [f for f in listdir(rootpath) if isfile(join(rootpath, f))] #Creates a list of only files, not folders.
onlyjpegs = [imago for imago in onlyfiles if imago.endswith(".jpg")]
print(rootpath)
if onlyjpegs == []:
    print("This folder contains no jpeg images.")
    exit()

#setup values for the sorting loop
brightlist = [] #create empty list to store brightness values
contra = [] #create empty contrast value list
#store dimensions of image for later use (all images should be same size for effective sort)
dimensions = (cv2.imread(join(rootpath, onlyjpegs[0]))).shape
ydim = dimensions[0]
xdim = dimensions[1]
ystart = round(ydim * start) # get beginning of datachunk for brightness sampling
yend = round(ydim * end) # get end
margo = round(xdim * margin)
totality = len(onlyjpegs)
imagenow = cv2.imread(join(rootpath, onlyjpegs[0]))

for current in trange(totality):
    imagenow = cv2.imread(join(rootpath, onlyjpegs[current]), 0) #join the rootpath string with the current item in the jpeg list for a full filepath
    imagesearch = imagenow[ystart:yend,margo:(xdim - margo)]
    brighto = np.mean(imagesearch)
    contrasttime = np.sort(np.reshape(imagesearch, [1,imagesearch.size]))
    consize = contrasttime.size
    bigcontrol = (np.mean(contrasttime[0,round(consize-(consize * senso)):consize] - np.mean(contrasttime[0,0:round((consize * senso))])))
    contra.append(bigcontrol)
    brightlist.append(brighto)

brightest = brightlist.index(max(brightlist))
shutil.move(join(rootpath, onlyjpegs[brightest]), join(rootpath, "Bright"))

print(brightest, "-", onlyjpegs[brightest])
del brightlist[brightest]
del onlyjpegs[brightest]
del contra[brightest]

#print(brightlist)
print(min(brightlist),", ",max(brightlist),", ",np.mean(brightlist))
mini = min(brightlist) + (darkbasket * (max(brightlist)-min(brightlist)))
maxi = max(brightlist) - (brightbasket * (max(brightlist)-min(brightlist)))
highcontrast = max(contra) - (sensotoo * (max(contra)-min(contra)))

for decide in range(len(brightlist)):
    if contra[decide] > highcontrast:
        shutil.move(join(rootpath, onlyjpegs[decide]), join(rootpath, "High Contrast"))
        continue

    if brightlist[decide] < mini:
        shutil.move(join(rootpath, onlyjpegs[decide]), join(rootpath, "Dark"))

    if brightlist[decide] > maxi:
        shutil.move(join(rootpath, onlyjpegs[decide]), join(rootpath, "Bright"))
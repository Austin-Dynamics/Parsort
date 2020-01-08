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
start = 0.1
end = 0.6
margin = 0.2

brightbasket = 0.2
darkbasket = 0.4

#contrast sensitivity variable
senso = 0.15 #how much of the bright and dark of an image chunk should be averaged to get the contrast score
sensotoo = 0.10 #How much of the high-contrast image pool should go into the folder

#get absolute path
script_dir = os.path.dirname(__file__)
#setup face cascade trained file
face_cascade = cv2.CascadeClassifier(join(script_dir,'important/haarcascade_frontalface_default.xml'))
#create empty list for faces
facelist = []

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
facebrightness = []
squarebrightness = []
nofind = []
#store dimensions of image for later use (all images should be same size for effective sort)
dimensions = (cv2.imread(join(rootpath, onlyjpegs[0]))).shape
ydim = dimensions[0]
xdim = dimensions[1]
ystart = round(ydim * start) # get beginning of datachunk for brightness sampling
yend = round(ydim * end) # get end
margo = round(xdim * margin)
totality = len(onlyjpegs)

for current in trange(totality):
    # join the rootpath string with the current item in the jpeg list for a full filepath
    imagenow = cv2.imread(join(rootpath, onlyjpegs[current]), 0)
    imagesearch = imagenow[ystart:yend, 0:xdim]
    face = face_cascade.detectMultiScale(imagesearch, 1.2, 5)
    if type(face) == np.ndarray:
        imageface = imagesearch[(face[0,1]):(face[0,1]+face[0,3]),(face[0,0]):(face[0,0]+face[0,2])]
        brighto = np.mean(imageface)
        contrasttime = np.sort(np.reshape(imageface, [1, imageface.size]))
        consize = contrasttime.size
        facelist.append(onlyjpegs[current])
        facebrightness.append(brighto)
    else:
        imagesearch = imagenow[ystart:yend, margo:(xdim - margo)]
        brighto = np.mean(imagesearch)
        contrasttime = np.sort(np.reshape(imagesearch, [1,imagesearch.size]))
        consize = contrasttime.size
        squarebrightness.append(brighto)
        nofind.append(onlyjpegs[current])
    bigcontrol = (np.mean(contrasttime[0,round(consize-(consize * senso)):consize] - np.mean(contrasttime[0,0:round((consize * senso))])))
    contra.append(bigcontrol)
    brightlist.append(brighto)

#compensate for background color in images where no face was found
print("\nface detected brightness was", np.mean(facebrightness))
print("brightness of non-face images was ",np.mean(squarebrightness))
compensate = np.mean(facebrightness)-np.mean(squarebrightness)
print("The difference is ",compensate)
if len(squarebrightness) > 5 and abs(np.mean(facebrightness)-np.mean(squarebrightness)) > 1.5:
    for current in range(len(squarebrightness)):
        brightlist[brightlist.index(nofind[current])] = (squarebrightness[current] + (compensate * 0.9))
    print("Non-face images compensated to ",np.mean(squarebrightness),"\n")
else:
    print("Difference not compensated\n")

#identify brightest image, sort it, and remove it from consideration (to avoid greycard skewing results)
brightest = brightlist.index(max(brightlist))
shutil.move(join(rootpath, onlyjpegs[brightest]), join(rootpath, "Bright"))
print(onlyjpegs[brightest], "(",brightest,") was the brightest image.\n")
del brightlist[brightest]
del onlyjpegs[brightest]
del contra[brightest]

#report brightness range and success of face detection
print(min(brightlist),", ",max(brightlist),", ",np.mean(brightlist))
print(len(facelist)," faces found out of ",len(onlyjpegs)," images total.")
print("Faces were not found in the following images:")
print(nofind)

#calculate threshold for brightness categories based on overall gamut
mini = min(brightlist) + (darkbasket * (max(brightlist)-min(brightlist)))
maxi = max(brightlist) - (brightbasket * (max(brightlist)-min(brightlist)))
highcontrast = max(contra) - (sensotoo * (max(contra)-min(contra)))

#move images to assigned categories
for decide in range(len(brightlist)):
    if contra[decide] > highcontrast:
        shutil.move(join(rootpath, onlyjpegs[decide]), join(rootpath, "High Contrast"))
        continue

    if brightlist[decide] < mini:
        shutil.move(join(rootpath, onlyjpegs[decide]), join(rootpath, "Dark"))

    if brightlist[decide] > maxi:
        shutil.move(join(rootpath, onlyjpegs[decide]), join(rootpath, "Bright"))

cv2.waitKey(0)
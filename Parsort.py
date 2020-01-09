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
margin = 0.35

#crop amount on face-identified square
facecrop = 0.25

brightbasket = 0.15
darkbasket = 0.6

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
    print("Path not selected or cannot be found.")
    input("Press Enter to continue...")
    exit()

#Check if deposit folders exist, and make them if not
createfolders = ["Dark","High Contrast","Bright","Unsorted"]
for folder in createfolders:
    try:
        os.mkdir(join(rootpath, folder))
    except OSError:
        print("Folder ",folder," already exists or cannot be made here.")
    else:
        print(folder, "made.")


onlyfiles = [f for f in listdir(rootpath) if isfile(join(rootpath, f))] #Creates a list of only files, not folders.
onlyjpegs = [imago for imago in onlyfiles if imago.endswith(".jpg")]
print(rootpath)
if onlyjpegs == []:
    print("This folder contains no jpeg images.")
    input("Press Enter to continue...")
    exit()

#setup values for the sorting loop
brightlist = [] #create empty list to store brightness values
contra = [] #create empty contrast value list

nofind = []
facecaught = []
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
        imageface = imagesearch[
                    int(face[0,1]+((face[0,3]*facecrop))):int(face[0,1]+(face[0,3]*(1-2*facecrop))),
                    int(face[0,0]+face[0,2]*facecrop):int(face[0,0]+(face[0,2]*(1-2*facecrop)))]
        brighto = np.mean(imageface)
        contrasttime = np.sort(np.reshape(imageface, [1, imageface.size]))
        consize = contrasttime.size
        facelist.append(onlyjpegs[current])
    else:
        face_cascade = cv2.CascadeClassifier(join(script_dir, 'important/haarcascade_mcs_nose.xml'))
        nose = face_cascade.detectMultiScale(imagesearch, 1.2, 5)
        if type(nose) == np.ndarray:
            imageface = imagesearch[
                        int(nose[0, 1] - ((nose[0, 3] * 1))):int(nose[0, 1] + (nose[0, 3] * (2.2))),
                        int(nose[0, 0] - nose[0, 2] * 1.4):int(nose[0, 0] + (nose[0, 2] * (3.2)))]
            brighto = np.mean(imageface)
            contrasttime = np.sort(np.reshape(imageface, [1, imageface.size]))
            consize = contrasttime.size
            facelist.append(onlyjpegs[current])
            facecaught.append(onlyjpegs[current])
            face_cascade = cv2.CascadeClassifier(join(script_dir, 'important/haarcascade_frontalface_default.xml'))
        else:
            brighto = 404
            consize = 404
            nofind.append(onlyjpegs[current])
            face_cascade = cv2.CascadeClassifier(join(script_dir, 'important/haarcascade_frontalface_default.xml'))
    bigcontrol = (np.mean(contrasttime[0,round(consize-(consize * senso)):consize] - np.mean(contrasttime[0,0:round((consize * senso))])))
    contra.append(bigcontrol)
    brightlist.append(brighto)

#move unsorted images into unsorted folder
for unsorted in nofind:
    jpegindex = onlyjpegs.index(unsorted)
    shutil.move(join(rootpath,unsorted), join(rootpath, "Unsorted"))
    del brightlist[jpegindex]
    del contra[jpegindex]
    del onlyjpegs[jpegindex]

#identify brightest image
brightest = brightlist.index(max(brightlist))
print(onlyjpegs[brightest], "(",brightest,") was the brightest image.\n")

#report brightness range and success of face detection
print(min(brightlist),", ",max(brightlist),", ",np.mean(brightlist))
print(len(facelist)," faces found out of ",len(onlyjpegs)+len(nofind)," images total.")
print("There were ",len(facecaught)," faces caught by the backup algorithm:")
if len(facecaught) > 0:
    print(facecaught)
if len(nofind) > 0:
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

input("\nPress Enter to continue...")
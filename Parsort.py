import os
from os import listdir
from os.path import isfile, join
from tkinter import Tk  # For preventing Tk GUI from opening
from tkinter.filedialog import askdirectory
import numpy as np
import cv2
from tqdm import trange
import shutil
from scipy import stats


#start and end of the vertical read chunk on images
start = 0.1
end = 0.6
margin = 0.35

#crop amount on face-identified square
facecrop = 0.25

brightbasket = 0.3
darkbasket = 0.4
satbasket = 0.25

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
createfolders = ["Dark","High Contrast","Bright","Unsorted","Saturated"]
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
saturate = []

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
    imagenow = cv2.imread(join(rootpath, onlyjpegs[current]))
    imagesearch = imagenow[ystart:yend, 0:xdim, 0:3]
    setface = cv2.cvtColor(imagesearch, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(setface, 1.2, 5)
    if type(face) == np.ndarray:
        imageface = imagesearch[
                    int(face[0,1]+((face[0,3]*facecrop))):int(face[0,1]+(face[0,3]*(1-2*facecrop))),
                    int(face[0,0]+face[0,2]*facecrop):int(face[0,0]+(face[0,2]*(1-2*facecrop))),0:3]
    else:
        face_cascade = cv2.CascadeClassifier(join(script_dir, 'important/haarcascade_mcs_nose.xml'))
        face = face_cascade.detectMultiScale(setface, 1.2, 5)
        if type(face) == np.ndarray:
            imageface = imagesearch[
                        int(face[0, 1] - ((face[0, 3] * 1))):int(face[0, 1] + (face[0, 3] * (2.2))),
                        int(face[0, 0] - face[0, 2] * 1.4):int(face[0, 0] + (face[0, 2] * (3.2))),0:3]
            facecaught.append(onlyjpegs[current])
            face_cascade = cv2.CascadeClassifier(join(script_dir, 'important/haarcascade_frontalface_default.xml'))
        else:
            brightlist.append(404)
            contra.append(404)
            nofind.append(onlyjpegs[current])
            saturate.append(404)
            face_cascade = cv2.CascadeClassifier(join(script_dir, 'important/haarcascade_frontalface_default.xml'))
            continue
    facelist.append(onlyjpegs[current])
    imageface = cv2.cvtColor(imageface,cv2.COLOR_BGR2HSV)
    imageface_h = imageface.shape[0]
    imageface_w = imageface.shape[1]
    brightlist.append(np.mean(imageface[0:imageface_h,0:imageface_w,2]))
    saturate.append(np.mean(imageface[0:imageface_h,0:imageface_w, 1]))
    contrastsetup = imageface[0:imageface_h,0:imageface_w, 2]
    contrasttime = np.sort(np.reshape(contrastsetup, [1, contrastsetup.size]))
    consize = contrasttime.size
    bigcontrol = (np.mean(contrasttime[0, round(consize - (consize * senso)):consize] - np.mean(
        contrasttime[0, 0:round((consize * senso))])))
    contra.append(bigcontrol)


#move unsorted images into unsorted folder
for unsorted in nofind:
    jpegindex = onlyjpegs.index(unsorted)
    shutil.move(join(rootpath,unsorted), join(rootpath, "Unsorted"))
    del brightlist[jpegindex]
    del contra[jpegindex]
    del onlyjpegs[jpegindex]
    del saturate[jpegindex]

#create Z-score lists and remove outlier data
z_bright = stats.zscore(brightlist)
z_bright = z_bright.tolist()
print("Lowest Z score for this set is ",
      min(z_bright),", the average was ",np.mean(z_bright),
      ", and the highest was ",max(z_bright))
print("The image with the lowest Z score was",onlyjpegs[z_bright.index(min(z_bright))],"\n")
print("The image with the highest saturation was",onlyjpegs[saturate.index(max(saturate))])

#report success of face detection
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
satmax = max(saturate) - (satbasket * (max(saturate)-min(saturate)))

#move images to assigned categories
for decide in range(len(brightlist)):
    if contra[decide] > highcontrast:
        shutil.move(join(rootpath, onlyjpegs[decide]), join(rootpath, "High Contrast"))
        continue

    if brightlist[decide] < mini:
        shutil.move(join(rootpath, onlyjpegs[decide]), join(rootpath, "Dark"))
    elif brightlist[decide] > maxi:
        shutil.move(join(rootpath, onlyjpegs[decide]), join(rootpath, "Bright"))
    elif saturate[decide] > satmax:
        shutil.move(join(rootpath, onlyjpegs[decide]), join(rootpath, "Saturated"))

input("\nPress Enter to continue...")
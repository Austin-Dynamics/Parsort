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
import PySimpleGUI as sg


# start and end of the vertical read chunk on images
start = 0.1
end = 0.6
margin = 0.35

# crop amount on face-identified square
facecrop = 0.25
noglasses = 0.5  # attempt to remove glasses from the sample pool

brightbasket = 0.3
darkbasket = 0.3
satbasket = 0.3

# contrast sensitivity variable
sensotoo = 0.1  # How much of the high-contrast image pool should go into the folder
clothesbias = 0.9  # Multiplier for clothes contrast to balance for hair contrast.
contbias = 1.3  # compensation multiplier to prefer darker skin tones in the high contrast folder

# get absolute path
script_dir = os.path.dirname(__file__)
# setup face cascade trained file
face_cascade = cv2.CascadeClassifier(join(script_dir, 'important/haarcascade_frontalface_default.xml'))
# create empty list for faces
facelist = []

# Get list of files
Tk().withdraw()  # Stop Tk GUI from opening, aesthetic improvement
rootpath = askdirectory()
if os.path.exists(rootpath):
    print("This is a real path")
else:
    print("Path not selected or cannot be found.")
    input("Press Enter to continue...")
    exit()

# Check if deposit folders exist, and make them if not
createfolders = ["Dark", "High Contrast", "Bright", "Unsorted", "Saturated"]
for folder in createfolders:
    try:
        os.mkdir(join(rootpath, folder))
    except OSError:
        print("Folder ", folder, " already exists or cannot be made here.")
    else:
        print(folder, "made.")


onlyfiles = [f for f in listdir(rootpath) if isfile(join(rootpath, f))]  # Creates a list of only files, not folders.
onlyjpegs = [imago for imago in onlyfiles if imago.endswith(".jpg")]
print(rootpath)
if not onlyjpegs:
    print("This folder contains no jpeg images.")
    input("Press Enter to continue...")
    exit()

# setup values for the sorting loop
brightlist = []  # create empty list to store brightness values
contra = []  # create empty contrast value list
saturate = []
nofind = []
facecaught = []

# store dimensions of image for later use (all images should be same size for effective sort)
dimensions = (cv2.imread(join(rootpath, onlyjpegs[0]))).shape
ydim = dimensions[0]
xdim = dimensions[1]
ystart = round(ydim * start)  # get beginning of datachunk for brightness sampling
yend = round(ydim * end)  # get end
margo = round(xdim * margin)
totality = len(onlyjpegs)

# MAIN SORTING LOOP
for current in trange(totality):
    imagenow = cv2.imread(join(rootpath, onlyjpegs[current]))
    imagesearch = imagenow[ystart:yend, 0:xdim, 0:3]
    face = face_cascade.detectMultiScale(imagesearch, 1.2, 7)
    if type(face) == np.ndarray:
        face_y = int(face[0, 1]+(face[0, 3]*(facecrop*0.5)))
        face_h = int((face[0, 1]+(face[0, 3]*(facecrop*0.5)))+(face[0, 3]*(1-facecrop)))
        face_x = int(face[0, 0]+face[0, 2]*(facecrop*0.5))
        face_w = int((face[0, 0]+face[0, 2]*(facecrop*0.5)) + face[0, 2] * (1 - facecrop))
    else:
        face_cascade = cv2.CascadeClassifier(join(script_dir, 'important/haarcascade_mcs_nose.xml'))
        face = face_cascade.detectMultiScale(imagesearch, 1.2, 5)
        if type(face) == np.ndarray:
            face_y = int(face[0, 1] - (face[0, 3] * 2))
            face_h = int(face[0, 1] + (face[0, 3] * 1.5))
            face_x = int(face[0, 0] - face[0, 2] * 1)
            face_w = int(face[0, 0] + (face[0, 2] * 2))
            facecaught.append(onlyjpegs[current])
            face_cascade = cv2.CascadeClassifier(join(script_dir, 'important/haarcascade_frontalface_default.xml'))
        else:
            brightlist.append(404)
            contra.append(404)
            nofind.append(onlyjpegs[current])
            saturate.append(404)
            face_cascade = cv2.CascadeClassifier(join(script_dir, 'important/haarcascade_frontalface_default.xml'))
            continue
    imageface = imagesearch[face_y:face_h, face_x:face_w, 0:3]
    facelist.append(onlyjpegs[current])
    imageface = cv2.cvtColor(imageface, cv2.COLOR_BGR2HSV)
    imageface_h = imageface.shape[0]
    imageface_w = imageface.shape[1]

    brightlist.append(np.mean(imageface[int(noglasses * imageface_h):imageface_h, 0:imageface_w, 2]))
    saturate.append(np.mean(imageface[0:imageface_h, 0:imageface_w, 1]))
    contrast_h = face_h-face_y
    contrast_w = face_w-face_x
    contrastclothes = imagenow[
                      ((face_y + ystart) + contrast_h * 2):((face_y + ystart) + contrast_h * 3),
                      face_x:face_w,
                      0:3]
    contrasthair = imagenow[
                   int((face_y + ystart) - contrast_h * 0.55):int((face_y + ystart) - contrast_h * 0.2),
                   face_x:face_w,
                   0:3]
    # account for images where the hair is out of range
    if not contrastclothes.any or not contrasthair.any:
        nofind.append(onlyjpegs[current])
        contra.append(404)
        continue
    # contrast sectors are converted to HSV space, then averaged for brightness
    contrasthair = cv2.cvtColor(contrasthair, cv2.COLOR_BGR2HSV)
    contrastclothes = cv2.cvtColor(contrastclothes, cv2.COLOR_BGR2HSV)
    hairmean = np.mean(contrasthair[0:contrasthair.shape[0], 0:contrasthair.shape[1], 2])
    clothesmean = np.mean(contrastclothes[0:contrastclothes.shape[0], 0:contrastclothes.shape[1], 2])
    # if the clothes are brighter than the face, a larger contrast multiplier is added
    if abs(brightlist[-1] - clothesmean) < 0:
        if abs(hairmean - brightlist[-1]) > clothesbias * abs(clothesmean - brightlist[-1]):
            contra.append(abs(hairmean - brightlist[-1]) * contbias)
        else:
            contra.append(clothesbias * (abs(clothesmean - brightlist[-1]) * contbias))
    else:
        if abs(hairmean - brightlist[-1]) > clothesbias * abs(clothesmean - brightlist[-1]):
            contra.append(abs(hairmean - brightlist[-1]))
        else:
            contra.append(clothesbias * abs(clothesmean - brightlist[-1]))


# move unsorted images into unsorted folder
for unsorted in nofind:
    jpegindex = onlyjpegs.index(unsorted)
    shutil.move(join(rootpath, unsorted), join(rootpath, "Unsorted"))
    del brightlist[jpegindex]
    del contra[jpegindex]
    del onlyjpegs[jpegindex]
    del saturate[jpegindex]


# create Z-score lists and remove outlier data
z_bright = stats.zscore(brightlist)
z_bright = z_bright.tolist()
z_sat = stats.zscore(saturate)
z_sat = z_sat.tolist()
outlierlist = []
zdelete = []

for outlier in range(len(onlyjpegs)):
    if abs(z_bright[outlier]) > 3 or abs(z_sat[outlier]) > 3:
        shutil.move(join(rootpath, onlyjpegs[outlier]), join(rootpath, "Unsorted"))
        outlierlist.append(onlyjpegs[outlier])

if len(outlierlist) > 0:
    for deleto in outlierlist:
        bigdeleto = onlyjpegs.index(deleto)
        del z_bright[bigdeleto]
        del z_sat[bigdeleto]
        del brightlist[bigdeleto]
        del contra[bigdeleto]
        del onlyjpegs[bigdeleto]
        del saturate[bigdeleto]
    print("\nOutliers were found:\n", outlierlist)
else:
    print("\nNo outliers found.\n")

print("The image with the highest saturation was", onlyjpegs[saturate.index(max(saturate))])


# report success of face detection
print(len(facelist), " faces found out of ", totality, " images total.")
print("There were ", len(facecaught), " faces caught by the backup algorithm:")
if len(facecaught) > 0:
    print(facecaught)
if len(nofind) > 0:
    print("Faces were not found in the following images:")
    print(nofind)

# calculate threshold for brightness categories based on overall gamut
mini = min(brightlist) + (darkbasket * (max(brightlist)-min(brightlist)))
maxi = max(brightlist) - (brightbasket * (max(brightlist)-min(brightlist)))
highcontrast = max(contra) - (sensotoo * (max(contra)-min(contra)))
satmax = max(saturate) - (satbasket * (max(saturate)-min(saturate)))

# move images to assigned categories
for decide in range(len(onlyjpegs)):
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

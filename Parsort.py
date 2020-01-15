import os
from os import listdir
from os.path import isfile, join
from tkinter import Tk  # For preventing Tk GUI from opening
from tkinter.filedialog import askdirectory
import numpy as np
import cv2
import shutil
from scipy import stats
import PySimpleGUI as sg


brightwindow = 0.3
darkwindow = 0.3
satwindow = 0.3
sensowindow= 0.1  # How much of the high-contrast image pool should go into the folder
contwindow = 1.35  # compensation multiplier to prefer darker skin tones in the high contrast folder

sg.theme('DarkTeal2')  # Add a little color to your windows
Tk().withdraw()
use_default_focus = False
layout = [[sg.Text('Category Tolerances:'), sg.Button("Reset", key="resetbutton")],
          [sg.Text('Bright'), sg.Spin([i for i in range(5, 55, 5)], initial_value=30, key="brightspin"),
           sg.Text('Dark'), sg.Spin([i for i in range(5, 55, 5)], initial_value=30, key="darkspin"),
           sg.Text('Saturated'), sg.Spin([i for i in range(5, 55, 5)], initial_value=30, key="satspin"), sg.Text(" | "),
           sg.Text('Contrast'), sg.Spin([i for i in range(5, 55, 5)], initial_value=10, key="conspin"),
           sg.Text('Bias'), sg.Spin([i for i in range(0, 55, 5)], initial_value=35, key="biasspin"),
           sg.Button("?", key="question")],
          [sg.Text(' ')],
          [sg.Text('File Path:', size=(9, 1), auto_size_text=False, justification='right'),
           sg.InputText('No Path Selected', key="fileselect", focus=True), sg.Button("Browse")],
          [sg.Text(' ' * 48), sg.Button("Run", key="run"), sg.Button('Close', key="finalclose")],
          [sg.Output(size=(70, 20))]]

grootpath = ""


def parsortation(brightbasket, darkbasket, satbasket, sensotoo, contbias, rootpath):
    # start and end of the vertical read chunk on images
    start = 0.1
    end = 0.6
    margin = 0.35

    # crop amount on face-identified square
    facecrop = 0.25
    noglasses = 0.5  # attempt to remove glasses from the sample pool

    # contrast sensitivity variable
    clothesbias = 0.9  # Multiplier for clothes contrast to balance for hair contrast.

    # get absolute path
    script_dir = os.path.dirname(__file__)
    # setup face cascade trained file
    face_cascade = cv2.CascadeClassifier(join(script_dir, 'important/haarcascade_frontalface_default.xml'))
    # create empty list for faces
    facelist = []

    # Get list of files
    Tk().withdraw()  # Stop Tk GUI from opening, aesthetic improvement
    if os.path.exists(rootpath):
        print("This is a real path")
    else:
        print("Path not selected or cannot be found.")
        window.Refresh()
        return

    # Check if deposit folders exist, and make them if not
    createfolders = ["Dark", "High Contrast", "Bright", "Unsorted", "Saturated"]
    for folder in createfolders:
        try:
            os.mkdir(join(rootpath, folder))
        except OSError:
            print("Folder ", folder, " already exists or cannot be made here.")
        else:
            print(folder, "folder made.")
        window.Refresh()

    onlyfiles = [f for f in listdir(rootpath) if
                 isfile(join(rootpath, f))]  # Creates a list of only files, not folders.
    onlyjpegs = [imago for imago in onlyfiles if imago.endswith(".jpg")]
    print(rootpath)
    if not onlyjpegs:
        print("This folder contains no jpeg images.")
        return

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
    if not os.path.exists(join(script_dir, "important/haarcascade_frontalface_default.xml")):
        print("Error! Haar Cascade XML files missing.")
        window.Refresh()
        return
    # MAIN SORTING LOOP
    for current in range(totality):
        if not sg.OneLineProgressMeter('Parsort', current + 1, totality, 'keybo',
                                       join("Sorting images in\n", rootpath), bar_color=("green2", "black")):

            if current + 1 < totality:
                print("Operation Canceled.")
                window.Refresh()
                return
        window.refresh()
        imagenow = cv2.imread(join(rootpath, onlyjpegs[current]))
        imagesearch = imagenow[ystart:yend, 0:xdim, 0:3]
        face = face_cascade.detectMultiScale(imagesearch, 1.2, 7)
        if type(face) == np.ndarray:
            face_y = int(face[0, 1] + (face[0, 3] * (facecrop * 0.5)))
            face_h = int((face[0, 1] + (face[0, 3] * (facecrop * 0.5))) + (face[0, 3] * (1 - facecrop)))
            face_x = int(face[0, 0] + face[0, 2] * (facecrop * 0.5))
            face_w = int((face[0, 0] + face[0, 2] * (facecrop * 0.5)) + face[0, 2] * (1 - facecrop))
        else:
            face_cascade = cv2.CascadeClassifier(join(script_dir, 'important/haarcascade_mcs_nose.xml'))
            face = face_cascade.detectMultiScale(imagesearch, 1.2, 5)
            if type(face) == np.ndarray:
                face_y = int(face[0, 1] - (face[0, 3] * 2))
                face_h = int(face[0, 1] + (face[0, 3] * 1.5))
                face_x = int(face[0, 0] - face[0, 2] * 1)
                face_w = int(face[0, 0] + (face[0, 2] * 2))
                facecaught.append(onlyjpegs[current])
                face_cascade = cv2.CascadeClassifier(
                    join(script_dir, 'important/haarcascade_frontalface_default.xml'))
            else:
                brightlist.append(404)
                contra.append(404)
                nofind.append(onlyjpegs[current])
                saturate.append(404)
                face_cascade = cv2.CascadeClassifier(
                    join(script_dir, 'important/haarcascade_frontalface_default.xml'))
                continue
        imageface = imagesearch[face_y:face_h, face_x:face_w, 0:3]
        facelist.append(onlyjpegs[current])
        imageface = cv2.cvtColor(imageface, cv2.COLOR_BGR2HSV)
        imageface_h = imageface.shape[0]
        imageface_w = imageface.shape[1]

        brightlist.append(np.mean(imageface[int(noglasses * imageface_h):imageface_h, 0:imageface_w, 2]))
        saturate.append(np.mean(imageface[0:imageface_h, 0:imageface_w, 1]))
        contrast_h = face_h - face_y
        contrast_w = face_w - face_x
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

    # report success of face detection
    print("\n", len(facelist), " faces found out of ", totality, " images total.")
    print("\nThere were ", len(facecaught), " faces caught by the backup algorithm:")
    if len(facecaught) > 0:
        print(facecaught)
    if len(nofind) > 0:
        print("Faces were not found in the following images:")
        print(nofind)
    window.Refresh()

    # calculate threshold for brightness categories based on overall gamut
    mini = min(brightlist) + (darkbasket * (max(brightlist) - min(brightlist)))
    maxi = max(brightlist) - (brightbasket * (max(brightlist) - min(brightlist)))
    highcontrast = max(contra) - (sensotoo * (max(contra) - min(contra)))
    satmax = max(saturate) - (satbasket * (max(saturate) - min(saturate)))

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
    print("\nSort Complete.")
    window.Refresh()


question_active = False

# Create the Window
window = sg.Window('Parsort', layout)
# Event Loop to process "events"
while True:
    event, values = window.read()
    if event in (None, 'finalclose'):
        break
    if event in "Browse":
        grootpath = askdirectory(title="Select File Path:")
        if os.path.exists(grootpath):
            window.Element('fileselect').Update(grootpath)
            window.refresh()
    if event in "resetbutton":
        window.Element('brightspin').Update(30)
        window.Element('darkspin').Update(30)
        window.Element('satspin').Update(30)
        window.Element('conspin').Update(10)
        window.Element('biasspin').Update(30)
    if event in "question":
        qlayout = [[sg.Text("The first four settings, 'Bright', 'Dark', 'Saturated', and 'Contrast'")],
                   [sg.Text("control the threshold for images to fall in their respective category.")],
                   [sg.Text("A small number means less images get sorted there, and a larger")],
                   [sg.Text("number means more images (that otherwise wouldn't fit).")],
                   [sg.Text(" ")],
                   [sg.Text("Contrast works the same way, but has an additional parameter:")],
                   [sg.Text("Bias. At a bias value of zero, the contrast category will tend")],
                   [sg.Text("to select images with bright skin tones and dark hair or clothes.")],
                   [sg.Text("Higher bias values prefer darker skin tones and brighter clothes.")],
                   [sg.Text("Leaving the value at 30 works for most sets and applications.")],
                   [sg.CloseButton("Close", key="qclose"), sg.Text(" "*55), sg.Text("▓", text_color="Dark Slate Blue",
                                                                                    tooltip="Created by Austin Flory")]]
        question = sg.Window("Help", qlayout, finalize=True)
        window.Element('question').Update(disabled=True)
        question_active = True
    if question_active is True:
        question.refresh()
        event2, values2 = question.read()
        if event2 in (None, "qclose"):
            question.close()
            question_active = False
            del question
            window.Element('question').Update(disabled=False)

    if event in "run":
        brightwindow = values['brightspin']/100
        darkwindow = values['darkspin']/100
        satwindow = values['satspin']/100
        sensowindow = values['conspin']/100
        contwindow = 1 + (values['biasspin']/100)
        grootpath = values['fileselect']
        window.Element('run').Update(disabled=True)
        window.Element('finalclose').Update(disabled=True)
        parsortation(brightwindow, darkwindow, satwindow, sensowindow, contwindow, grootpath)
        window.Element('run').Update(disabled=False)
        window.Element('finalclose').Update(disabled=False)

window.close()
del window

from timeit import default_timer as timer
import tkinter
from PIL import Image
from PIL import ImageTk
import cv2
import numpy as np
from math import sqrt

MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.25
MIN_MATCHES = 10
camindex = 2
ransacReprojThreshold = 25.0
im1 = cv2.imread('template.jpg')
im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 100)
flann = cv2.FlannBasedMatcher(index_params, search_params)
stds = 1
stdd = 1
cap = cv2.VideoCapture(camindex)
if not cap.isOpened():
    print("Camera not found at index ", camindex)
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
font = cv2.FONT_HERSHEY_SIMPLEX
devthresh = 0.5
accurate = 0
detected = 1
undetected = 0
#lgdev = 100
lgdst = []
lgcen = ()
lgsz = 1
drops = 0

def eucliddist(p1, p2):
    dx = p1[0]-p2[0]
    dy = p1[1]-p2[1]
    dist = sqrt(dx*dx + dy*dy)
    return dist

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def check_convexity(points):
    val = [0,0,0,0]
    for i in range(4):
        x0, y0 = points[i][0], points[i][1]
        x1, y1 = points[(i+1)%4][0], points[(i+1)%4][1]
        x2, y2 = points[(i+2)%4][0], points[(i+2)%4][1]
        val[i] = sign((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))
    if 0 in val:
        return False
    if(val == [1,1,1,1] or val == [-1,-1,-1,-1]):
        return True
    return False

def validate(corners):
    points = [np.array(corners[i][0]) for i in range(4)]
    cvx = check_convexity(points)
    sides = [eucliddist(points[i], points[(i+1)%4]) for i in range(4)]
    #print(sides)
    dev = np.std(sides)/np.mean(sides)
    #print(dev)
    if(dev<devthresh and cvx):
        return 3
    elif(dev>=devthresh and cvx):
        return 2
    elif(dev<devthresh and not cvx):
        return 1
    else:
        return 0
    
def quadcentroid(corners): #returns centre of quadrilateral
    sumx=0.0
    sumy=0.0
    for i in range(4):
        sumx+=corners[i][0][0]
        sumy+=corners[i][0][1]
    return (int(sumx//4) , int(sumy//4))

def quadsize(corners):  #returns longest diagonal
    diag1 = eucliddist(corners[0][0], corners[2][0])
    diag2 = eucliddist(corners[1][0], corners[3][0])
    return max(diag1, diag2)

def triangulate(quantity, currs):
    return (1,1)

def reqacc (rvel , cdist, taracc):
    if(cdist[0]!=0):
        a0 = taracc[0] + 0.5*np.dot(rvel[0], rvel[0])/cdist[0]
    else:
        a0=taracc[0]
    if(cdist[1]!=0):
        a1 = taracc[1] + 0.5*np.dot(rvel[1], rvel[1])/cdist[1]
    else:
        a1 = taracc[1]
    acc = [a0, a1]
    return acc

def cendist(mcen, frame0):
    h,w,c = frame0.shape
    cen = (w//2, h//2)
    off=np.subtract(np.array(cen), np.array(mcen))
    return off
    
def displacement(pos0, pos1):
    pixshift=(pos1[0]-pos0[0], pos1[1]-pos0[1])
    return pixshift

def pixvel (pos0, pos1, framerate):
    svect = displacement(pos0, pos1)
    vvect = (svect[0]*framerate , svect[1]*framerate)
    return vvect

def pixacc (pos0, pos1, pos2, framerate):
    vel0 = pixvel(pos0, pos1, framerate)
    vel1 = pixvel(pos1, pos2, framerate)
    avect = ((vel1[0]-vel0[0])*framerate, (vel1[1]-vel0[1])*framerate)
    return avect

def pixacc (vel0, vel1, framerate):
    avect = ((vel1[0]-vel0[0])*framerate, (vel1[1]-vel0[1])*framerate)
    return avect

def pformat(pair):
    x = str(pair[0])[0:8]
    y = str(pair[1])[0:8]
    return '('+x+','+y+')'

def findmatch(im2):
    global accurate
    global detected
    global undetected
    #global lgdev
    global lgdst
    global lgcen
    global lgsz
    global drops
    try:
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
        matches = matcher.match(descriptors1, descriptors2, None)
        matches.sort(key=lambda x: x.distance, reverse=False)
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]
        if(len(matches)<MIN_MATCHES):
            raise Exception('Not enough matches')
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, ransacReprojThreshold = ransacReprojThreshold)
        height, width, channels = im1.shape
        pts = np.float32([ [0,0],[0,height-1],[width-1,height-1],[width-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,h)
        cen = quadcentroid(dst)
        qsz = quadsize(dst)
        iMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
        #cv2.polylines(im2,[np.int32(dst)],True,(255, 255, 0),3, cv2.LINE_AA)
        if(fcount%10==0):
            #lgdev = 0.6
            cv2.imwrite('frames/frame'+str(fcount)+'.jpg', iMatches)
        v = validate(dst)
        if v==3:
            accurate +=1
            detected +=1
            undetected = 0
            #if v==3:
            lgdst,lgcen,lgsz = dst,cen,qsz
            #cv2.polylines(im2,[np.int32(lgdst)],True,(0, 64, 0),3, cv2.LINE_AA)
            cv2.polylines(im2,[np.int32(dst)],True,(0, 255, 0),3, cv2.LINE_AA)
            cv2.putText(im2,'Success',(10,100), font, 1,(0,255,0),2,cv2.LINE_AA)
        elif v==1 or v==0:
            detected +=1
            if(undetected<=10 and accurate>0):
                undetected +=1
                drops+=1
                accurate+=1
                cv2.polylines(im2,[np.int32(lgdst)],True,(0, 255, 255),3, cv2.LINE_AA)
            cv2.putText(im2,'Failure',(10,100), font, 1,(0,0,255),2,cv2.LINE_AA)
            if(accurate>0):
                cen = lgcen
                qsz = lgsz
        elif v==0:
            raise Exception('Non convex overskewed bounding box')
        display = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)))    
        return cen, qsz, display

    except:
        undetected+=1
        if(undetected<=10 and accurate>0):
            drops+=1
            detected+=1
            accurate+=1
            cv2.polylines(im2,[np.int32(lgdst)],True,(128, 128, 128),3, cv2.LINE_AA)
            cv2.putText(im2,'Success',(10,100), font, 1,(0,255,0),2,cv2.LINE_AA)
        else:
            cv2.putText(im2,'Not detected',(10,100), font, 1,(255,100,0),2,cv2.LINE_AA)
        display = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)))
        if(accurate>0):
            return lgcen, lgsz, display
        return (0,0), 1, display
    
try:
    print('Starting system, Capturing camera ', camindex)
    fcount = 0
    prev = timer()
    window = tkinter.Tk()
    start = timer()
    window.title('Live Stream')
    ret, frame0 = cap.read()
    pos0, sz0, mat = findmatch(frame0)
    now = timer()
    framerate = 1/(now-prev)
    prev = now
    ret, frame1 = cap.read()
    pos1, sz1, mat = findmatch(frame1)
    h1, w1, c1 = im1.shape
    h2, w2, c2 = frame0.shape
    width = w1+w2
    height = max(h1, h2)
    canvas = tkinter.Canvas(window, width = width, height = height)
    canvas.pack()
    vel0 = pixvel(pos0, pos1, framerate)
    fcount = 2
    while(True):
        ret, frame2 = cap.read()
        fcount+=1
        pos2, sz2, mat = findmatch(frame2)
        now = timer()
        framerate = 1/(now-prev)
        prev = now
        canvas.create_image(0, 0, image = mat, anchor = tkinter.NW)
        window.update()
        vel1 = pixvel(pos1, pos2, framerate)
        acc0 = pixacc(vel0, vel1, framerate) #acceleration between frames
        cdist = cendist(pos0, frame0)    #distance of target from centre of field of view
        avgsize = (sz0+sz1+sz2)/3               #average size of target in 3 frames
        racc = reqacc(vel0, cdist, acc0)        #required pixel acceleration according to current data
        realacc = triangulate(racc, avgsize)  #required actual acceleration
        print('\033[K Frame:   ',fcount, 'Framerate:   ',int(framerate), 'Average FPS:   ', (fcount//(now-start)), 'Accuracy:    ', (accurate*100//detected), ' %')
        print('\033[K Position:                ',pformat(pos2),'   pixel')
        print('\033[K Shift:                   ',pformat(displacement(pos0, pos1)),'   pixel')
        print('\033[K Velocity:                ',pformat(vel1),'  pixel/s')
        print('\033[K Acceleration:            ',pformat(acc0),'  pixel/s^2')
        print('\033[K Required Accceleration:  ',pformat(realacc),'  metre/s^2')
        print('\033[A'*7)
        pos0, sz0 = pos1, sz1
        pos1, sz1 = pos2, sz2
        vel0 = vel1
        
except KeyboardInterrupt:
    print("\n"*7, "Releasing camera...")
    cap.release()
    print("Writing to log file")
    with open('stats.txt', 'a') as logfile:
        #total frames, Accuracy %, Buffered, Total detected, Total accurate, Actual accurate, Undetected
        logfile.write(str(fcount)+'\t\t'+str((accurate*100//detected))+'\t\t'+str(drops)+'\t'+str(detected)+'\t'+str(accurate)+'\t'+str(accurate-drops)+'\t'+str(fcount-detected))
    print("Exiting...")
    exit()

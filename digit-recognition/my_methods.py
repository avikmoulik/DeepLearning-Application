#--------------------------------
# Input mnist
#-------------------------------
def read_img_mnist(filename):
    
    import cv2    
    img = cv2.imread(filename)
        
    return (img)

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

#--------------------------------
# Input from paint
#-------------------------------
def read_img_paint(filename,img_width,img_height,w):
    
    import cv2 
    import numpy as np
    import math
    
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
        # pre processing to get mnist like data format
    while np.sum(img[0]) == 0:
        img = img[1:]
    
    while np.sum(img[:,0]) == 0:
        img = np.delete(img,0,1)
    
    while np.sum(img[-1]) == 0:
        img = img[:-1]
    
    while np.sum(img[:,-1]) == 0:
        img = np.delete(img,-1,1)
    
    rows,cols = img.shape
    
    if rows > cols:
        factor = w/rows
        rows = w
        cols = int(round(cols*factor))
        img = cv2.resize(img, (cols,rows))
    else:
        factor = w/cols
        cols = w
        rows = int(round(rows*factor))
        img = cv2.resize(img, (cols, rows))
        
    colsPadding = (int(math.ceil((img_width-cols)/2.0)),int(math.floor((img_width-cols)/2.0)))
    rowsPadding = (int(math.ceil((img_height-rows)/2.0)),int(math.floor((img_height-rows)/2.0)))
    img = np.lib.pad(img,(rowsPadding,colsPadding),'constant')
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
    return (img)

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------


#--------------------------------
# Input Image from disk
#-------------------------------
def read_img_disk(filename):
    
    import cv2
    
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, bw = cv2.threshold(gray, 128, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    return (img,bw)

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

#--------------------------------
# Edge Detection function
#-------------------------------
def auto_canny(image, sigma=0.33):
    
    import cv2
    import numpy as np
    
# compute the median of the single channel pixel intensities
    v = np.median(image) 
# apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    return edged

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

#--------------------------------
# Input Image from mobile
#--------------------------------
# Replace the URL with your own IPwebcam shot.jpg IP:port

def read_img_url(url):
    
    import urllib
    import cv2
    import numpy as np
    import imutils
    from imutils.perspective import four_point_transform
    import my_methods
    
    img = cv2.imdecode(np.array(bytearray(urllib.request.urlopen(url).read()),dtype=np.uint8),-1)
    
    # load the image and compute the ratio of the old height to the new height, clone it, and resize it
    ratio = img.shape[0] / 500.0
    orig = img.copy()
    img = imutils.resize(img, height = 500)
    
    # convert the image to grayscale, blur it, and find edges in the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
     # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)
    
    
    gray_thresh = cv2.threshold(gray_clahe, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    edged = my_methods.auto_canny(gray_thresh)
    
    # find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
     
    # loop over the contours
    for c in cnts:
    # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
     
    # if our approximated contour has four points, then we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    
    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
    
    # apply the four point transform to obtain a top-down view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
     
    # convert the warped image to grayscale, then threshold it to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    
    # threshold the warped image
    thresh = cv2.threshold(warped, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    rw,cl = thresh.shape
    bw = thresh[int(round(.02*rw,0)):int(rw-round(.02*rw,0)),int(round(.02*cl,0)):int(cl-round(.02*cl,0))]

    return (img,bw)

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

#--------------------------------
# Input Image from mobile no edge detection
#--------------------------------
# Replace the URL with your own IPwebcam shot.jpg IP:port

def read_img_url_noedge(url):
    
    import urllib
    import cv2
    import numpy as np
    
    img = cv2.imdecode(np.array(bytearray(urllib.request.urlopen(url).read()),dtype=np.uint8),-1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    bw = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    return (img,bw)

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

#--------------------------------
# function for Model Definition
#--------------------------------
def create_model(img_width,img_height):
    
    from keras.models import Sequential
    from keras.layers import Convolution2D
    from keras.layers import MaxPooling2D
    from keras.layers import Flatten
    from keras.layers import Dense
#    from keras.layers import Dropout
    
    model = Sequential()
    model.add(Convolution2D(16, 5, 5, activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(MaxPooling2D(2, 2))
    
    model.add(Convolution2D(32, 5, 5, activation='relu'))
    model.add(MaxPooling2D(2, 2))
    
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    
    model.add(Dense(10, activation='softmax'))

    return model

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

#--------------------------------
# function for digit segmentation and prediction
#--------------------------------

def img_segmentation(bw,img_width,img_height,w,weights_to_use):
    
    import cv2
    import pandas as pd
    import math
    import numpy as np
    import my_methods
    
    connectivity = 4
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity, cv2.CV_32S)
    
    df = pd.DataFrame(stats,index=range(0,nb_components),columns=('x','y','w','h','a')) #converting to data frame
    df = df.drop([0],axis =0) # drop 1st record
    df = df.sort(['x'], ascending=[1]) #sort by x coordinates
    df = df.assign(req_h=round(df.h.max()*.5,0)) #calculate minimum required heights
    df = df.drop(df[df.h < df.req_h].index,axis =0) #keep only relevant records
    df = df.assign(x1 = df.x,x2= df.x+df.w,y1=df.y,y2=df.y+df.h) #Calculating the final coordinates
    df = df.drop(['a', 'req_h','x','y','w','h'],axis = 1) #Droping unwanted cols
    tot_component = df.shape[0] #Counting no of digits
    stats=df.values #converting to numpy array
    
    #Looping over different digits
    
    dig = [None] * tot_component
    
    for i in range(tot_component):    

        img = bw[stats[i][2]:stats[i][3],stats[i][0]:stats[i][1]]
        
        # pre processing to get mnist like data format
        while np.sum(img[0]) == 0:
            img = img[1:]
        
        while np.sum(img[:,0]) == 0:
            img = np.delete(img,0,1)
        
        while np.sum(img[-1]) == 0:
            img = img[:-1]
        
        while np.sum(img[:,-1]) == 0:
            img = np.delete(img,-1,1)
        
        rows,cols = img.shape
        
        if rows > cols:
            factor = w/rows
            rows = w
            cols = int(round(cols*factor))
            img = cv2.resize(img, (cols,rows))
        else:
            factor = w/cols
            cols = w
            rows = int(round(rows*factor))
            img = cv2.resize(img, (cols, rows))
            
        colsPadding = (int(math.ceil((img_width-cols)/2.0)),int(math.floor((img_width-cols)/2.0)))
        rowsPadding = (int(math.ceil((img_height-rows)/2.0)),int(math.floor((img_height-rows)/2.0)))
        img = np.lib.pad(img,(rowsPadding,colsPadding),'constant')
        
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        
        model = my_methods.create_model(img_width,img_height)
        model.load_weights(weights_to_use)
        arr = np.array(img).reshape((img_width,img_height,3))
        arr = np.expand_dims(arr, axis=0)
        arr = arr/255
        prediction = model.predict(arr)[0]
        bestclass = ''
        bestconf = -1
        for n in [0,1,2,3,4,5,6,7,8,9]:
            if (prediction[n] > bestconf):
                bestclass = str(n)
                bestconf = prediction[n]
        dig[i]=bestclass
        
    dig_fin = ''.join(dig)
    print ('I think this digit is a ' + dig_fin )
    return dig_fin

#--------------------------------
# function prediction
#--------------------------------

def img_predict_mnist(img,img_width,img_height,weights_to_use):
    
    import numpy as np
    import my_methods

    model = my_methods.create_model(img_width,img_height)
    model.load_weights(weights_to_use)
    arr = np.array(img).reshape((img_width,img_height,3))
    arr = np.expand_dims(arr, axis=0)
    arr = arr/255
    prediction = model.predict(arr)[0]
    bestclass = ''
    bestconf = -1
    for n in [0,1,2,3,4,5,6,7,8,9]:
        if (prediction[n] > bestconf):
            bestclass = str(n)
            bestconf = prediction[n]
    dig = bestclass
    
    dig_fin = ''.join(dig)
    print ('I think this digit is a ' + dig_fin )

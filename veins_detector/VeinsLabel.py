# VeinsLabel
# Esta clase permite la clasificación del
# estado de la retina a partir de la imagen entregada
# Desarrollado por UFOTECH S.A.S.
# Parte de este script fue tomado de https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
# https://github.com/krishna1401/Digital-Image-Processing/blob/master/highBoostFiltering.py

import cv2
import numpy as np
from PIL import Image
import os
from math import atan2, cos, sin, sqrt, pi, atan
import argparse
from pathlib import Path
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from skimage.io import imread
from operator import itemgetter
from configparser import ConfigParser

input_config = ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
input_config.read(config_path)
base_directory = input_config['DEFAULT']['base_directory']
base_output_directory = input_config['DEFAULT']['base_output_directory']
fez = int(input_config['PIL']['fez'])
pixel_limit = int(input_config['PIL']['pixel_limit'])
dilate_iter = int(input_config['VEINS_LABEL']['dilate_iter'])
max_rad = int(input_config['VEINS_LABEL']['max_rad'])
mask_rad_ratio = float(input_config['VEINS_LABEL']['mask_rad_ratio'])
max_distance = int(input_config['VEINS_LABEL']['max_distance'])
low_aspect = float(input_config['VEINS_LABEL']['low_aspect'])
high_aspect = float(input_config['VEINS_LABEL']['high_aspect'])
papila=int(input_config['VEINS_LABEL']['papila'])
rad_zone_1=int(int(input_config['VEINS_LABEL']['rad_zone_1']))
rad_zone_2=int(input_config['VEINS_LABEL']['rad_zone_2'])
rad_zone_3=int(rad_zone_2*float(input_config['VEINS_LABEL']['rad_zone_3']))
im_offset=int(input_config['VEINS_LABEL']['im_offset'])


try:
    os.makedirs(base_output_directory+'mask/', exist_ok=True)
except:
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='/home/audacia/veins_detector/output/noir/temp_Cr1mLmN-noir.png', help='file/dir/URL/glob/screen/0(webcam)')
parser.add_argument('--papila-loc', type=str, default= '/home/audacia/veins_detector/output/labels/temp_Cr1mLmN.txt', help='text file with the papila coordinates')
parser.add_argument('--is-papila', default=False, action='store_true', help='decide if process optic nerve')
args = vars(parser.parse_args())
is_papila=args["is_papila"]

def get_papila(text_filename):
    list_of_lists=[]
    with open(text_filename) as f:
        for row in f:
            inner_list=[elt.strip() for elt in row.split(" ")]
            list_of_lists.append(inner_list)
        f.close()
    papila_box=[eval(i) for i in sorted(list_of_lists, key=lambda x:x[-1], reverse=True)[0][5:-1]]
    papila_x_rect=int(papila_box[0]+((papila_box[2]-papila_box[0])/2))
    papila_y_rect=int(papila_box[1]+((papila_box[3]-papila_box[1])/2))
    x_rad=int((papila_box[2]-papila_box[0])//2)
    y_rad=int((papila_box[3]-papila_box[1])//2)
    def_rad=x_rad if x_rad < y_rad else y_rad
    return papila_x_rect, papila_y_rect, def_rad

def pil_pixel(image2_file,pixel_limit):
    """  
    
    #Objective: Normalize image pixel-wise using PIL by 
    # setting to 0 pixels with value below a threshhold;
    # copy of the normalized image is saved
    #Input: Original Image & pixel threshhold
    #Output: Resultant Image path
    

    """
    
    MyImg = Image.open(image2_file, 'r')
    pixels = MyImg.load() # creates the pixel map

    for i in range(MyImg.size[0]):    
        for j in range(MyImg.size[1]):  
            coordinate = x, y = i, j
            ## using getpixel method
            if MyImg.getpixel(coordinate) <=pixel_limit:
                pixels[i,j] = 0
            else:
                pixels[i,j] = 255

    fin = base_output_directory+"PIL/"+Path(image2_file).stem+'-PIL'+Path(image2_file).suffix        
    MyImg.save(fin)
    return fin


def mask_me(img):
    hh, ww = img.shape[:2]
    hh2 = hh // 2
    ww2 = ww // 2

    # define circles
    radius = int(im.shape[0]//mask_rad_ratio)
    yc = hh2
    xc = ww2

    # draw filled circle in white on black background as mask
    mask = np.zeros_like(img)
    mask = cv2.circle(mask, (xc,yc), radius, (255,255,255), -1)
    
    # apply mask to image
    result = cv2.bitwise_and(img, mask)

    return result


def get_me_max(image):
    img = imread(image)
    im = img_as_float(img)
    # image_max is the dilation of im with a 20*20 structuring element
    # It is used within peak_local_max function
    image_max = ndi.maximum_filter(im, size=fez, mode='constant')
    # Comparison between image_max and im to find the coordinates of local maxima
    coordinates = peak_local_max(im, min_distance=fez)
    return coordinates

def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    ## [visualization1]

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    

# Function to find the angle between two lines

# This code is contributed by mohit kumar 29.
def findAngle(M1, M2):
	
    # Store the tan value of the angle
    angle = abs((M2 - M1) / (1 + M1 * M2))

    # Calculate tan inverse of the angle
    ret = atan(angle)

    # Convert the angle from radian to degree
    val = np.rad2deg(ret)

    # Print the result
    return int(val)


def findRoy(img,papila_x_rect,papila_y_rect,def_rad):
    
    
    def_zone_1=2*(2*def_rad)
    def_zone_2=2*def_zone_1
    def_zone_3=int(1.5*def_zone_2)
    
    zone_1=img[papila_y_rect-def_zone_1:papila_y_rect+def_zone_1,papila_x_rect-def_zone_1:papila_x_rect+def_zone_1]
    zone_2=img[papila_y_rect-def_zone_2:papila_y_rect+def_zone_2,papila_x_rect-def_zone_2:papila_x_rect+def_zone_2]
    zone_3=img[papila_y_rect-def_zone_3:papila_y_rect+def_zone_3,papila_x_rect-def_zone_3:papila_x_rect+def_zone_3]

    
    # draw filled circle in white on black background as mask
    mask_zone_2 = np.zeros_like(img)
    mask_zone_2 = cv2.circle(mask_zone_2, (papila_x_rect,papila_y_rect), def_zone_1, (255,255,255), -1)#rad_zone_1
    mask_zone_2 = cv2.rectangle(mask_zone_2, (papila_x_rect-def_zone_2, papila_y_rect-def_zone_2), (papila_x_rect+def_zone_2, papila_y_rect+def_zone_2), (255,255,255), 50)
    mask_zone_2 = cv2.bitwise_not(mask_zone_2)
    mask_zone_2 = mask_zone_2[papila_y_rect-def_zone_2:papila_y_rect+def_zone_2,papila_x_rect-def_zone_2:papila_x_rect+def_zone_2]
    
    # apply mask to image
    result_zone_2 = cv2.bitwise_and(zone_2, mask_zone_2)
    
    mask_zone_3 = np.zeros_like(img)
    
    mask_zone_3 = cv2.rectangle(mask_zone_3, (papila_x_rect-def_zone_3, papila_y_rect-def_zone_3), (papila_x_rect+def_zone_3, papila_y_rect+def_zone_3), (255,255,255), 50)
    mask_zone_3 = mask_zone_3[papila_y_rect-def_zone_3:papila_y_rect+def_zone_3,papila_x_rect-def_zone_3:papila_x_rect+def_zone_3]
    
    # apply mask to image
    result_zone_3 = cv2.bitwise_and(zone_3, mask_zone_3)


    # Image classification by retinal health status
    if np.sum(result_zone_2==255)<0 and np.sum(result_zone_3==255)>0:
        return False
    elif np.sum(result_zone_2==255)>0 and np.sum(result_zone_3==255)>0:
        return True
    elif np.sum(result_zone_2==255)>0 and np.sum(result_zone_3==255)<0:
        return True
    elif np.sum(result_zone_2==255)>0 :
        return True
    elif np.sum(result_zone_2==255)<0 and np.sum(result_zone_3==255)<0:
        return False


def find_manual_nerve(img,the):

    white_set=[]
    off_me = 75
    for th in the:
        y = th[0]-int(off_me/2)
        x = th[1]-int(off_me/2)
        height = width = off_me
        roi = img[y:y+height, x:x+width]
        white_set.append([th,np.sum(roi==255)])
    sorted_white=sorted(white_set,key=itemgetter(1),reverse=True)
    
    return sorted_white[0][0][0],sorted_white[0][0][1]


def find_nasal(img,papila_x_rect,papila_y_rect,quadrants,def_rad):

    white_set=[]
    black_set=[]
    def_zone_1=2*(2*def_rad)
    def_zone_2=2*def_zone_1
    def_zone_3=int(1.5*def_zone_2)

    for k,v in quadrants.items():
        him=img[v[0][0]:v[0][1],v[0][2]:v[0][3]]
        white_set.append([k,np.sum(him==255)])
        black_set.append([k,np.sum(him==0)])
    
    sorted_black=sorted(black_set, key=itemgetter(1),reverse=True)
    sorted_white=sorted(white_set, key=itemgetter(1),reverse=False)
    

    if sorted_black[0][0] == sorted_white[0][0]:
        winner=quadrants.get(sorted_black[0][0])
        cv2.line(img, (papila_x_rect, papila_y_rect), (winner[1][0], winner[1][1]), (255,2,255), 3, cv2.LINE_AA)
        cv2.circle(img, (papila_x_rect, papila_y_rect), def_rad, (255, 0, 255), -1)#papila
        cv2.circle(img, (papila_x_rect, papila_y_rect), def_zone_1, (0,0,255),2)#zona 1#rad_zone_1
        cv2.circle(img, (papila_x_rect, papila_y_rect), def_zone_2, (255,0,0),2)#zona 1#rad_zone_1
        cv2.ellipse(img, (papila_x_rect, papila_y_rect), (int(def_zone_3*0.9),int(def_zone_3*0.7)), findAngle(winner[1][2],winner[1][3]), 0, 360, (0,255,255))#zona 3
        
        
def set_dict(papila_x_rect,papila_y_rect):
    quadrants={"quad1":[[papila_y_rect-im_offset,papila_y_rect,papila_x_rect-im_offset,papila_x_rect],[int(papila_x_rect+(papila_x_rect/2)),int(papila_y_rect+(papila_y_rect/2)),1,0]],
                    "quad2":[[papila_y_rect-im_offset,papila_y_rect,papila_x_rect,papila_x_rect+im_offset],[int(papila_x_rect-(papila_x_rect/2)),int(papila_y_rect+(papila_y_rect/2)),1,0]],
                    "quad3":[[papila_y_rect,papila_y_rect+im_offset,papila_x_rect-im_offset,papila_x_rect],[int(papila_x_rect+(papila_x_rect/2)),int(papila_y_rect-(papila_y_rect/2)),-1,0]],
                    "quad4":[[papila_y_rect,papila_y_rect+im_offset,papila_x_rect,papila_x_rect+im_offset],[int(papila_x_rect-(papila_x_rect/2)),int(papila_y_rect-(papila_y_rect/2)),-1,0]]}
    return quadrants       

def getOrientation(pts, img):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))

    return cntr

def find_me_pixels(gray,contours):
    mask = np.zeros(gray.shape,np.uint8)
    cv2.drawContours(mask,contours,0,255,-1)
    pixelpoints = np.transpose(np.nonzero(mask))
    return pixelpoints

def common_member(a, b):
    
    a_set = set([tuple(i) for i in a.tolist()])
    b_set = set([tuple(i) for i in b.tolist()])

    if (a_set & b_set):
        return list(a_set & b_set)
    else:
        return list(a_set)


def connect_me_dots(im,coordinates):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # distance-transform
    dist = cv2.distanceTransform(~gray, cv2.DIST_L1, 3)
    
    # max distance
    k = max_distance
    bw = np.uint8(dist < k)
    
    # remove extra padding created by distance-transform
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bw2 = cv2.morphologyEx(bw, cv2.MORPH_ERODE, kernel)
    
    # extra threshhold
    bw2 = cv2.dilate(bw2, np.ones((3,3), np.uint8), iterations=dilate_iter)
    ret,bw2 = cv2.threshold(bw2, 107, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # clusters
    contours, _ = cv2.findContours(bw2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        radius = int(radius)
        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h#ha de estar en un umbral (0.8<aspect_ratio<2)

        if radius > max_rad and (aspect_ratio<high_aspect and aspect_ratio>low_aspect):

            if is_papila:
                papila_x_rect,papila_y_rect,def_rad=get_papila(args["papila_loc"]) 
                quads=set_dict(papila_x_rect,papila_y_rect)
                findRoy(im,papila_x_rect,papila_y_rect,def_rad)
                find_nasal(im,papila_x_rect,papila_y_rect,quads,def_rad)
            else:
                ey,ex=find_manual_nerve(im,coordinates) 
                papila_x_rect=ex
                papila_y_rect=ey
                quads=set_dict(papila_x_rect,papila_y_rect)
                findRoy(im,papila_x_rect,papila_y_rect,int(rad_zone_1/4))
                find_nasal(im,papila_x_rect,papila_y_rect,quads,papila)
        else:
            pass
        
    mask_name=base_output_directory+'mask/'+Path(args["source"]).stem+"-mask"+Path(args["source"]).suffix
    cv2.imwrite(mask_name, im)

##Image is fistly normalized using PIL
image_file_path = pil_pixel(args["source"],pixel_limit)
coordinates = get_me_max(args["source"])
im = cv2.imread(image_file_path)
im = mask_me(im)
connect_me_dots(im,coordinates)
os.remove(image_file_path)


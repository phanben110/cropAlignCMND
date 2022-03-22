import numpy as np
import matplotlib.pyplot as plt
import cv2 
from PIL import Image 
from src.detector.detector import Detector
from src.detector.utils.image_utils import align_image 
from src.config import corner_detection 

class CropCMND(object):
    def __init__(self):
        self.cornerDetectionModel = Detector(path_to_model=corner_detection['path_to_model'],
                                               path_to_labels=corner_detection['path_to_labels'],
                                               nms_threshold=corner_detection['nms_ths'], 
                                               score_threshold=corner_detection['score_ths'])
        self.coordinateDict = dict() 
        #Define some bouding box 
        self.width = 300 
        self.height = 500 
        
        #Define bouding box image 3x4  
        self.bbImg3x4 = (16/self.width, 122/self.height, 142/self.width, 280/self.height) 

        #Define bouding box text  
        self.bbText = (150/self.width, 70/self.height, 498/self.width, 296/self.height) 
            


    def detectCorner(self, image):
        detectionBoxes, detectionClasses, categoryIndex = self.cornerDetectionModel.predict(image)

        height, width, _ = image.shape

        for i in range(len(detectionClasses)):
            label = str(categoryIndex[detectionClasses[i]]['name'])
            realYmin = int(max(1, detectionBoxes[i][0]))
            realXmin = int(max(1, detectionBoxes[i][1]))
            realYmax = int(min(height, detectionBoxes[i][2]))
            realXmax = int(min(width, detectionBoxes[i][3]))
            self.coordinateDict[label] = (realXmin, realYmin, realXmax, realYmax)

        # align image
        croppedImg = align_image(image, self.coordinateDict)
        return croppedImg 


    def predict(self, image):
        img = np.asarray(image)
        croppedImage= self.detectCorner(img)

        return croppedImage  

    def cropHeadmap(self, headmap): 
        headmap = np.asarray(headmap) 
        croppedHeadmap = align_image(headmap, self.coordinateDict)
        return croppedHeadmap

    def drawBoudingBox(self, image): 
        h,w,_ = image.shape
        cv2.rectangle(image, (int(self.bbImg3x4[0]*h),int(self.bbImg3x4[1]*w)), (int(self.bbImg3x4[2]*h), int(self.bbImg3x4[3]*w)), (255,0,0), thickness=1) 
        cv2.rectangle(image, (int(self.bbText[0]*h),int(self.bbText[1]*w)), (int(self.bbText[2]*h), int(self.bbText[3]*w)), (0,0,255), thickness=1) 
        return image 
    
if __name__ == "__main__": 
    #load image  
    pathCMND = 'images/cmnd.png'
    pathHeadmap = 'images/headmap.png'
    imgCMND = Image.open(pathCMND)
    headmapCMND = Image.open(pathHeadmap)

    #Load model cropCMND and align
    model = CropCMND() 

    imgCMND = model.predict(imgCMND) 

    #crop and align headmapCMND 
    headmapCMND = model.cropHeadmap(headmapCMND)

    #Convert image from RGB to BGR
    imgCMND = cv2.cvtColor(imgCMND, cv2.COLOR_RGB2BGR)
    headmapCMND = cv2.cvtColor(headmapCMND, cv2.COLOR_RGB2BGR)

    #draw bouding box 
    imgCMND = model.drawBoudingBox(imgCMND)
    headmapCMND = model.drawBoudingBox(headmapCMND)
    
    #imshow result
    cv2.imwrite('results/cmnd.jpg', imgCMND)
    cv2.imwrite('results/headmap.png',headmapCMND)

    cv2.imshow('CMND', imgCMND )       
    cv2.imshow('headmap', headmapCMND)
                              
    key = cv2.waitKey()           



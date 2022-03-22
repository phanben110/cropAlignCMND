from cropCMND import CropCMND 
import cv2 
from PIL import Image 
import numpy as np 

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
                                              


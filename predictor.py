import os

import numpy as np
import cv2
import torch

from PIL import Image

import matplotlib.pyplot as plt

from utilities.utils import *

def predict_with_knn(capture: cv2.VideoCapture):
    main_screen = 'Main Screen'
    mask_screen = 'Mask Screen'
    create_screen(main_screen, 640, 480)
    create_screen(mask_screen, 640, 480)
    
    subtractor = cv2.createBackgroundSubtractorKNN()

    keep_processing = True 
    while keep_processing:
        key_code = cv2.waitKey(25) & 0xFF
        keep_processing = key_code != 27
        status, frame = capture.read() 
        if status:
            mask = subtractor.apply(frame)
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contours = remove_noise(contours) 
                print('Number of persons: ', len(contours))
                # cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                if frame.shape[0] > 0 and frame.shape[1] > 0:
                    cv2.imshow(main_screen, frame)
                
                if mask.shape[0] > 0 and mask.shape[1] > 0:
                    cv2.imshow(mask_screen, mask)

    capture.release()
    
        
def predict_with_yolo(capture: cv2.VideoCapture):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    main_screen = 'Main Screen'
    create_screen(main_screen, 640, 480)
    
    keep_processing = True
    while keep_processing:
        key_code = cv2.waitKey(25) & 0xFF
        keep_processing = key_code != 27
        status, frame = capture.read() 
        if status:
            results = model(frame)
            cv2.imshow(main_screen, np.squeeze(results.render()))
            
    capture.release()            
        
            
            
    # imgs = ['https://ultralytics.com/images/zidane.jpg']
    # results = model(imgs)

    # results.print()
    # results.show()

if __name__ == '__main__':
    logger.info('Testing predictor...')
    pass
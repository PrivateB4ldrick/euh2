import argparse
import cv2
import sys
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper

vidloc = "C:\\Users\\OHPC\\Desktop\\Daaaaaaaaaaaaaarreen\\futbol\\booo.mp4"
imgloc_bp = "C:\\Users\\OHPC\\Desktop\\Daaaaaaaaaaaaaarreen\\futbol\\ball.jpg"

def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            
            frame=cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)             
            frame1=cv2.GaussianBlur(frame, (7,7), 0)            
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
            if between(cap, 0, 5000):           
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to graycsale           
                frame = cv2.GaussianBlur(frame, (7,7), 0) # Blur the image for better edge detection           
    # Morphology
                kernel = np.ones((9,9),np.uint8) 
                frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)  #saca negro          
    #            frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
             
    # 0 → 5 Sobel Edge Detection        
            if between(cap, 0000, 1000):    
                frame = cv2.Sobel(src=frame, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=13)# Sobel Edge Detection on the X axis   
                frame= cv2.bitwise_not(frame)
            if between(cap, 1000, 2000):    
                frame = cv2.Sobel(src=frame, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=13) # Sobel Edge Detection on the Y axis
            if between(cap, 2000,3000):    
                frame = cv2.Sobel(src=frame, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=15) # Combined X and Y Sobel Edge Detection
            if between(cap, 3000, 4000):    
                frame = cv2.Sobel(src=frame, ddepth=cv2.CV_64F, dx=2, dy=2, ksize=19) # Combined X and Y Sobel Edge Detection
            if between(cap, 4000, 5000):    
                frame = cv2.Sobel(src=frame, ddepth=cv2.CV_64F, dx=5, dy=5, ksize=23) # Combined X and Y Sobel Edge Detection
                
    # 0 → 5 Subtitles          
            if between(cap, 0000, 20000):  
                cv2.rectangle(frame, (0,687), (500,900), (255,255,255), 200)
            if between(cap, 0000, 1000):    
                cv2.putText(frame, "Sobel Edge Detection is used on the X", (20,610),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
                cv2.putText(frame, "axis,", (20,630),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
            if between(cap, 1000, 2000):    
                cv2.putText(frame, "Sobel Edge Detection is used on the X", (20,610),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
                cv2.putText(frame, "axis, the Y axis", (18,630),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
            if between(cap, 2000,3000):    
                cv2.putText(frame, "Sobel Edge Detection is used on the X", (20,610),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
                cv2.putText(frame, "axis, the Y axis and both axis combined.", (16,630),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
            if between(cap, 3000, 5000):    
                cv2.putText(frame, "The increase of the parameters make", (20,610),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
                cv2.putText(frame, "significant differences on the output.", (18,630),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
             
            if between(cap, 5000, 6500):    
                circles = cv2.HoughCircles (frame1,cv2.HOUGH_GRADIENT,1,100,param1=30,param2=30,minRadius=0,maxRadius=200)
            if between(cap, 6500, 8000):    
                circles = cv2.HoughCircles (frame1,cv2.HOUGH_GRADIENT,1,200,param1=30,param2=30,minRadius=0,maxRadius=150)                  
            if between(cap, 8000, 12000):    
               circles = cv2.HoughCircles (frame1,cv2.HOUGH_GRADIENT,1,250,param1=40,param2=40,minRadius=0,maxRadius=120)              
            if between(cap, 12000, 15000):    
                circles = cv2.HoughCircles (frame1,cv2.HOUGH_GRADIENT,1,600,param1=45,param2=45,minRadius=15,maxRadius=120)   
    #VER            circles = cv2.HoughCircles (frame1,cv2.HOUGH_GRADIENT,1,600,param1=42,param2=42,minRadius=40,maxRadius=90) 
                      
            if between(cap,5000,15000):
                if circles is None:
                    continue
                circles = np.uint16(np.around(circles))
                
                for i in circles[0,:]:
                    # draw the outer circle
                    cv2.circle(frame,(i[0],i[1]),i[2],(140,110,255),3)
                    # draw the center of the circle
                    cv2.circle(frame,(i[0],i[1]),2,(85,255,255),3)
                    
    # 5 → 15 Subtitles                 
            if between(cap, 5000, 20000):  
                cv2.rectangle(frame, (0,680), (500,900), (255,255,255), 200)                
            if between(cap, 5000, 8000):    
                cv2.putText(frame, "The Circle Hough Transform function also", (10,596),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
            if between(cap, 5000, 8000):
                cv2.putText(frame, "shows how the choice of the parameters ", (10,615),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
            if between(cap, 5000, 8000):
                cv2.putText(frame, "influences on a better object detection", (10,634),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
          
            if between(cap, 8000, 12000):    
                cv2.putText(frame, "due to parameters like distance between", (10,596),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
            if between(cap, 8000, 12000):
                cv2.putText(frame, "centers, precision of the shape, size ", (10,615),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
            if between(cap, 8000, 12000):
                cv2.putText(frame, "of the circle found, respectively", (10,634),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
                    
            if between(cap, 12000, 15000):    
                cv2.putText(frame, "In this case, it is shown now the", (10,596),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
            if between(cap, 12000, 15000):
                cv2.putText(frame, "best option found between the possible", (10,615),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
            if between(cap, 12000, 15000):
                cv2.putText(frame, "combinations", (10,634),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
                
    # 15 → 17 Rectangle
            if between(cap,15000,17000):
                circles = cv2.HoughCircles (frame1,cv2.HOUGH_GRADIENT,1,600,param1=45,param2=45,minRadius=15,maxRadius=120)                
                if circles is None:
                    continue
     
                circles = np.uint16(np.around(circles))  
                
                for i in circles[0,:]:
                    # draw the outer circle
                    cv2.rectangle(frame,(i[0]-i[2]-5,i[1]-i[2]-5),(i[0]+i[2]+5,i[1]+i[2]+5),(255,100,100),3)
                    # draw the center of the circle
                    cv2.circle(frame,(i[0],i[1]),2,(140,110,255),3)         
      

    #-----------------------------------------------------------------------------------------------------------------     
            if between(cap,17000,20000):   

    # roi is the object or region that we need to find CIRCLE
    #            roi = frame[i[0]-i[2]:i[0]+i[2], i[1]-i[2]:i[1]+i[2]]
                roi = cv2.imread(imgloc_bp)
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # target it's the image that we're searching for FRAME
                target = frame
                hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

    #  calculate the histogram of the object 
                roihist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0,256])

    #  standardize the histogram and apply the projection 
                cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
                dst = cv2.calcBackProject([hsvt], [0,1], roihist, [0,180,0,256], 1)

    #  convolved with the disk kernel 
                disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                cv2.filter2D(dst, -1, disc, dst)

    #  the threshold value 、 binary bitwise and operations 
                ret, thresh = cv2.threshold(dst, 80, 255, 0)
                frame = cv2.merge((thresh, thresh, thresh))
    #            frame = cv2.bitwise_and(target, thresh)
    #            frame = cv2.cvtColor(frame, cvCOLOR_HSV2RGB)                       FALTA PASAR A BLANCO Y NEGRO

    #            frame = np.vstack((target, thresh, res))
    #-----------------------------------------------------------------------------------------------------------------            
            
    #Subtitles 15-20             
            if between(cap, 15000, 20000):    
                cv2.putText(frame, "EXPLANATION", (10,596),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
                
                
    #        if between(cap, 20000, 30000):
    #            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #            lb = np.array([])
                
                
                
    #20-40
            if between(cap,40000,50000):
                circles = cv2.HoughCircles (frame1,cv2.HOUGH_GRADIENT,1,600,param1=45,param2=45,minRadius=15,maxRadius=120)                
                if circles is None:
                    continue
     
                circles = np.uint16(np.around(circles))  
                
                for i in circles[0,:]:
                    # draw the outer circle
                    cv2.rectangle(frame,(i[0]-i[2],i[1]-i[2]),(i[0]+i[2],i[1]+i[2]),(255,100,100),3)
                    # draw the center of the circle
                    cv2.circle(frame,(i[0],i[1]),2,(140,110,255),i[2])
    

            # write frame that you processed to output
            print("Trying to write frame")
            out.write(frame)
            print("Written frame")

            # (optional) display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main("C:\\Users\\OHPC\\Desktop\\Daaaaaaaaaaaaaarreen\\futbol\\booo.mp4", "C:\\Users\\OHPC\\Desktop\\Daaaaaaaaaaaaaarreen\\futbol\\output1.mp4")

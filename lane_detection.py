import cv2
from random import randrange
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

def angle_between_lines(m1, m2):
    """
    Function: Calculate the angle between two lines given their slopes.

    Parameters:
        - m1 (float): Slope of line 1
        - m2 (float): Slope of line 2
    
    Return: Angle between the two lines in degrees
    """

    # Calculates angle between the two lines
    tan_theta = abs((m2 - m1) / (1 + m1 * m2))
    theta = math.atan(tan_theta)
    return math.degrees(theta)

def detect_lines(img,threshold1 = 50,threshold2 = 150,apertureSize = 3,minLineLength=100,maxLineGap=10):
    """
    Function: Detects the lines on a given frame
    
    Parameters:
        - img: the current frame from the AUV
        - threshold1 (int): the minimum threshold for the Canny edges detection (default value is 50)
        - threshold2 (int): the maximum threshold for the Canny edges detection (default value is 150)
        - apertureSize (int): how much light gets into the camera (can be an odd integer from 3-7, default value is 3)
        - minLineLength (int): the minimum length in pixels for a detected line to actually be considered a line (default value is 100)
        - maxLineGap (int): the maximum gap in pixels between two lines for them to be considered different lines (default value is 10)

    Return: The list of lines that are detected on the given frame
    """

    # Converts image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # Uses gaussian blur to blur the image so the tiny tiles at the bottom of the pool aren't detected as lines
    blurred_image = cv2.GaussianBlur(gray, (9, 9), 0)

    #plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
    #plt.show()

    # Uses Canny edge detection to detect the lines
    edges = cv2.Canny(blurred_image, threshold1, threshold2, apertureSize) 

    # Uses Hough Lines Probabilistic transformation to detect the lines in the pool based on given parameters
    lines = cv2.HoughLinesP(
            edges,
            rho = 1,
            theta = np.pi/180,
            threshold = 100,
            minLineLength = minLineLength,
            maxLineGap = maxLineGap,
    )

    
    #plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    #plt.show()
    #print (lines)
    #be close enough, have similar slopes, be on the same side of the image
    return lines

def draw_lines(img,lines,color = (0, 255, 0)):
    """
    Function: Draws the lines that the AUV detects on the frame
    
    Parameters:
        - img: the current frame from the AUV
        - lines: a list of lines that the AUV detected
        - color: the RGB values for the color of the line
        
    Return: The image with the lines drawn on it
    """

    # Draws every line detected in the frame
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), color, 6)
    return img

def get_slopes_intercepts(lines):
    """
    Function: Calculates the slopes and x-intercepts of the lines detected in the frame
    
    Parameters:
        - lines (list): a list of lines detected on the frame
        
    Return: The lists of slopes and x-intercepts for the lines detected on the frame
    """

    # Stores the slope as the key, and the intercept as the data
    resultSet = set()

    # Initializes slope and x-intercept lists
    slopeList = []
    xInterceptList = []

    # Calculates the slopes and x-intercepts for any lines that were detected
    if len(lines) > 0:
        for line in lines:
            # Calculates slope of line
            x1, y1, x2, y2 = line[0]
            slope = (y1-y2)/(x1-x2)
            if slope == 0:
                slope = 0.001

            # Calculates the x-intercept of the line
            xIntercept = ((((1080 - y1)/slope)  )+ x1)
            roundXIntercept = round(xIntercept, 0)

            # Adds the slope and x-intercepts if they aren't already detected in the resultSet
            if not roundXIntercept in resultSet:
                resultSet.add(roundXIntercept) 
                xInterceptList.append(xIntercept)
            #    resultSet[slope][1] += 1 # keep a counter of how many lines have been iterated through and added to the one slope for averaging later
                slopeList.append(slope)

    
    # for result in resultSet:
    #     #result[0] = result[0]/result[1] # apply the dividing in the averaging
    #     xInterceptList.append(result)

    return slopeList, xInterceptList

def detect_lanes(lines):
    """
    Function: Detects all lanes made by lines detected on the frame
    
    Parameters:
        - lines (list): a list of lines detected on the image
        
    Return: A list of all lanes found in the image
    """
    
    # Gets the slopes and x-intercepts of all the lines
    slopeList, xInterceptList = get_slopes_intercepts(lines)
    #print (f"slopeList:{slopeList}")
    #print (f"xInterceptList:{xInterceptList}")

    lanes = []
    # Check if the lines intersect at the end of the screen
    # Checks if there is more than one slope/line
    if len(slopeList)> 1:
        # Performs a nested for loop to check how close the lines are to each other and which onces should be considered lanes
        for i in range(0,len(slopeList)):
            for j in range (i+1,len(slopeList)):
                # Calculates the distance between the x-intercepts of two lines  
                InterceptDist = abs(xInterceptList[i]-xInterceptList[j])
                # Sets the difference in slope to 0 if one of the slopes is 0 to prevent a divide by 0 error
                if slopeList[i] == 0 or slopeList[j] == 0:
                    slopeDiff = 0
                # Calculates the absolute value of the difference in inverse slopes of the lines
                else:
                    slopeDiff = abs(1/ slopeList[i]-1 /slopeList[j])
                
                # Arbitrary value if slope is equal to 0?
                slopeThing = 1000000
                # Sets slopeThing equal to one over the slope if the slope isn't equal to 0
                if not slopeList[i] == 0:
                    slopeThing = 1/slopeList[i]
                #print(f"DistREQ:{abs(xInterceptList[i]-xInterceptList[j])}")
                #print(f"slopeREQ:{abs(1/ slopeList[i]-1 /slopeList[j])}")
                # if statement to make sure lane is not too big (multiple lanes as one) not too different in slope (wrong side/ different lanes) and not too horizontal (other lienes reced as pool lane)

                # Does checks to determine if the two lines form a lane
                if(InterceptDist > 100 and InterceptDist< 750 and slopeDiff< 1 and abs(slopeThing) < 3 ):
                    #print(f"1/ slope:{slopeThing}")
                    # Calculates x-coordinate and y-coordinate of the "endpoint" of the lane
                    xPoint = ((slopeList[i] * xInterceptList[i]) - (slopeList[j] * xInterceptList[j]))/(slopeList[i]-slopeList[j])
                    yPoint = slopeList[i]*(xPoint - xInterceptList[i]) + 1080
                    
                    # avgSlope = (slopeList[i]+ slopeList[j])/2
                    # avgInterecept = (xInterceptList[i]+xInterceptList[j])/2
                    # Lines of the lanes
                    lane1 = [xInterceptList[i], 1080, xPoint, yPoint]
                    lane2 = [xInterceptList[j], 1080, xPoint, yPoint]
                    addedlanes = [lane1,lane2]
                    #print (f"thiasdfee:{(slopeList[i] * xInterceptList[i]) - slopeList[j] * xInterceptList[j]}")
                    lanes.append(addedlanes)


            #lanes.append(lane)

            #

            # if (yPoint> -500 and yPoint< 1080):
            #     avgInterceptX = (xInterceptList[i] + xInterceptList[j])/2
            #     lane = [xPoint.item(), avgInterceptX.item(), yPoint.item(), 1080.00]
            #     lanes.append(lane)

    return lanes

def pick_lane(frameHeight, frameWidth, lanes):
    """
    Function: Picks the lane closest to the AUV
    
    Parameters:
        - lanes (list): a list of all the lanes detected in the
    
    Return: The lane closest to the AUV"""

    maxLaneFitness = -10000000000
    maxDiff = 0
    
    # For all of the lanes, it checks the slopes of the boundary lines and the angle between them
    for addedLanes in lanes:
        center_slope_weight = 1000
        try:
            x1, y1, x2, y2 = addedLanes[0]
            slope1 = (y1-y2)/(x1-x2)
            x1, y1, x2, y2 = addedLanes[1]
            slope2 = (y1-y2)/(x1-x2)
            LineAngle = angle_between_lines(slope1, slope2)
            #print(f"I works")
            center_slope = abs((1/(((1/slope1)+(1/slope2))/2) )* center_slope_weight)
            
        except:
            center_slope = .0001
            LineAngle = .001
        diff = abs(addedLanes[0][0]  - addedLanes[1][0])
        yPoint = addedLanes[0][3]
        #print(f"yPoint:{yPoint}")
        VertDistToCenter = abs(yPoint - (frameHeight/2))
        xPoint = addedLanes[0][2]
        HortDistToCenter = abs(xPoint - (frameWidth/2))
        trueDistToCenter = np.sqrt(pow(VertDistToCenter,2)+pow(HortDistToCenter,2))
        #print (f"trueDistToCenter:{trueDistToCenter}")
        #print (f"diff:{diff}")
        #print (f"center_slope:{center_slope}")
        #laneFitness = diff - trueDistToCenter/2 + center_slope # calculate fitness, bigger = better, closer to center = better, lower centerline slope = better
        laneFitness = LineAngle #use lineangle as a analog for how close the lane is 
        print (laneFitness)
        
        if (maxLaneFitness < laneFitness and LineAngle < 50):
            maxLaneFitness = laneFitness
            pickedLane = addedLanes
            print (f"picked new lane! fitness: {laneFitness} <---------------------------------------")
    #print(f"picked: {pickedLane}")
    return pickedLane

def draw_lanes(img,lanes,color = (255, 0, 0)):
    """
    Function: Draws the lanes detected on the image
    
    Parameters:
        - img: the current frame of the AUV
        - lanes (list): a list of lanes detected on the image
        - color (tuple): the RGB color values for the lanes
        
    Return: The image with the lanes drawn on it
    """

    for addedLanes in lanes:
        color = (randrange(255),randrange(255),randrange(255))
        for lane in addedLanes:
            
            x1, y1, x2, y2 = lane
       #     print ("type(x1)")
         #  print (lane)
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 6)
    return img

def draw_Single_lane(img,lanes,color = (255, 0, 0)):
    """
    Function: Draws a single lane (used to draw the lane closest to the AUV)

    Parameters:
        - img: the current frame of the AUV
        - lanes (list): a list of the lanes detected on the image (we feed in the lane closest to the AUV)
        - color (tuple): the RGB color values for the lanes

    Return: The image with the lane drawn on it
    """
    #color = (randrange(255),randrange(255),randrange(255))
    for lane in lanes:
        
        x1, y1, x2, y2 = lane
       # print ("type(x1)")
      #  print (lane)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 6)
    return img
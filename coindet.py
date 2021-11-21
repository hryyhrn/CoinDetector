# Imports
import cv2 
import matplotlib.pyplot as plt
import numpy as np



# Input the image file, 
# reading image using OpenCV
fileName = input("Enter Image Name: ")
img = cv2.imread('D:\\Projects\\bython\\coin_detector\\img\\' + fileName) 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Converting to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Thresholding
ret, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

# Erosion with a custom structuring element
# dtype = np.uint8 (Unsigned int)
kernel = np.array([[0, 1], [1, 0]], dtype = np.uint8)
erosion = cv2.erode(thresh, kernel, iterations = 1)

# Contours
# Parent only, using RETR_EXTERNAL as a flag
contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)



coins = 0
# Checking parent contours for being circles/closed figures resembling circles
# (Number of sides > 50) 
for contour in contours:
    
    shape = cv2.approxPolyDP(contour, 0.0001*cv2.arcLength(contour, True), True)

    if len(shape) > 50:
        coins += 1



# Output
print('Number of Coins: ', coins)
plt.imshow(img)
plt.show()

import cv2
import numpy as np

# use cv to read the given image, Here im using grayscale
# which is a constant and is 0 here
# but here I'm not sure
#img = cv2.imread('red.png',cv2.IMREAD_GRAYSCALE)

# Here I think I batter read it as a color image, but I currently don't know how to use cv2.Color
img = cv2.imread('red.png',cv2.IMREAD_COLOR)

# show basic info about the image that was read
print(type(img))
print(img.shape)

'''
# show the image
cv2.imshow('img',img)
# wait 1000ms, 0 for forever
k = cv2.waitKey(0)

# save
cv2.imwrite("answer.png",img)

'''

# Convert the image to the HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# cv2.imshow('hsv',img);k = cv2.waitKey(0)


# Define the HSV range for bright orange-yellow
lower_orange_yellow = np.array([10, 100, 100])
upper_orange_yellow = np.array([15, 255, 255])

# Create a mask for the bright orange-yellow color
mask = cv2.inRange(hsv, lower_orange_yellow, upper_orange_yellow)

# Apply morphological operations to reduce noise
kernel = np.ones((5, 5), np.uint8)
mask = cv2.erode(mask, kernel, iterations=1)
mask = cv2.dilate(mask, kernel, iterations=1)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a list to store the centers of the contours
centers = []

for cnt in contours:
    # Filter out small contours that may be noise
    if cv2.contourArea(cnt) > 100:
        # Calculate moments for each contour
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            # Calculate x, y coordinates of the center
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centers.append((cx, cy))
            # Optionally, draw the center on the image
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

# Sort centers based on y-coordinate (top to bottom)
centers.sort(key=lambda x: x[1])

# Draw red lines to indicate the road
if len(centers) >= 2:
    for i in range(len(centers) - 1):
        pt1 = centers[i]
        pt2 = centers[i + 1]
        # Draw a red line between consecutive centers
        cv2.line(img, pt1, pt2, (0, 0, 255), 5)
else:
    print("Not enough orange-yellow markers detected to define the road.")


#Display the mask (optional, for debugging)
# cv2.imshow('Mask', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Show the image with the detected road
cv2.imshow('Road Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('road_detected.png', img)



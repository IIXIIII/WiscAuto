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


# Define HSV ranges for red color
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([179, 255, 255])

# Create masks for both ranges
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

# Combine masks
mask = cv2.bitwise_or(mask1, mask2)

# Apply morphological operations to reduce noise
kernel = np.ones((5, 5), np.uint8)
mask = cv2.erode(mask, kernel, iterations=2)
mask = cv2.dilate(mask, kernel, iterations=2)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize centers list
centers = []

for cnt in contours:
    if cv2.contourArea(cnt) > 100:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(cnt)
        # Optionally filter by aspect ratio or other properties
        # aspect_ratio = float(w) / h
        # if 0.8 < aspect_ratio < 1.2:

        # Draw a blue rectangle around the detected cone
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Calculate the center of the contour
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centers.append((cx, cy))
            # Mark the center with a blue circle
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)

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
    print("Not enough red markers detected to define the road.")

# Show the result
cv2.imshow('Road Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('answer.png', img)
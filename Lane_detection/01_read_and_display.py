import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor

# use cv to read the given image, Here im using grayscale
# which is a constant and is 0 here
# but here I'm not sure
#img = cv2.imread('red.png',cv2.IMREAD_GRAYSCALE)

# Here I think I batter read it as a color image, but I currently don't know how to use cv2.Color
img = cv2.imread('red.png',cv2.IMREAD_COLOR)

print(type(img))
print(img.shape)

height, width = img.shape[:2]

# Convert the image to the HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the HSV range for red color (handles the hue wrapping)
lower_red1 = np.array([0, 70, 51])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

# Create masks for both red ranges
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

# Combine the masks
mask = cv2.bitwise_or(mask1, mask2)

# ===== NEW CODE STARTS HERE =====

# Create an ROI mask that excludes the two upper corners
roi_mask = np.ones_like(mask, dtype=np.uint8) * 255  # Start with a white mask

# Define the polygons for the two upper corners
# For the left upper corner
pts_left = np.array([[0, 0], [int(width * 0.5), 0], [0, int(height * 0.5)]], np.int32)
pts_left = pts_left.reshape((-1, 1, 2))

# For the right upper corner
pts_right = np.array([[int(width * 0.5), 0], [width, 0], [width, int(height * 0.5)]], np.int32)
pts_right = pts_right.reshape((-1, 1, 2))

# Fill the upper corner polygons with black (0) to exclude them
cv2.fillPoly(roi_mask, [pts_left, pts_right], 0)

# Apply the ROI mask to the cone detection mask
mask = cv2.bitwise_and(mask, roi_mask)

# ===== NEW CODE ENDS HERE =====

# Apply morphological operations to reduce noise
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a list to store the centers of the cones
centers = []

for cnt in contours:
    if cv2.contourArea(cnt) > 100:
        # Calculate the center of the contour
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centers.append([cx, cy])
            # Optionally, mark the center on the image
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)

if len(centers) >= 2:
    centers_np = np.array(centers)
    X = centers_np[:, 0].reshape(-1, 1)  # Reshape for sklearn
    y = centers_np[:, 1]

    # Fit the first line using RANSAC
    ransac = RANSACRegressor(residual_threshold=5, random_state=0)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_

    # Draw the first line
    line_X = np.array([X.min(), X.max()])  # X values for the line
    line_y = ransac.predict(line_X.reshape(-1, 1))
    pt1 = (int(line_X[0]), int(line_y[0]))
    pt2 = (int(line_X[1]), int(line_y[1]))
    cv2.line(img, pt1, pt2, (0, 0, 255), 5)  # Red line for the first line

    # Remove inliers (cones close to the first line)
    X_outliers = X[~inlier_mask]
    y_outliers = y[~inlier_mask]

    if len(X_outliers) >= 2:
        # Fit the second line using RANSAC
        ransac2 = RANSACRegressor(residual_threshold=5, random_state=0)
        ransac2.fit(X_outliers, y_outliers)
        inlier_mask2 = ransac2.inlier_mask_

        # Draw the second line
        line_X2 = np.array([X_outliers.min(), X_outliers.max()])
        line_y2 = ransac2.predict(line_X2.reshape(-1, 1))
        pt3 = (int(line_X2[0]), int(line_y2[0]))
        pt4 = (int(line_X2[1]), int(line_y2[1]))
        cv2.line(img, pt3, pt4, (0, 0, 255), 5)  # Red line for the second line
    else:
        print("Not enough cones to fit a second line.")
else:
    print("Not enough cones detected to define the road.")

# Show the result
cv2.imshow('Road Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image
cv2.imwrite('answer_Final.png', img)

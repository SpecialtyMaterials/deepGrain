import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage import img_as_ubyte, img_as_float

nmPPx = 138
umPPx = 138/1000

img = cv2.imread("C:/Users/nickb/Desktop/projects/cobDetection/diffusionGrain/results/fullImg_testFiber1_1.png")
height, width,_ = img.shape
img2 = cv2.imread("C:/Users/nickb/Desktop/projects/cobDetection/diffusionGrain/testing/fullImg_testFiber1_1.png")

_, thresholded = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
cv2.imshow("Thresholded Result", thresholded)

binary_mask = thresholded > 0
skeleton = skeletonize(img_as_float(binary_mask))
finalSkeletonCleaned = img_as_ubyte(skeleton)

def draw_colored_contours(image, contours):
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colors = []
    for i, contour in enumerate(contours):
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)

        if w < 0.5 * width and h < 0.5 * height:
            if cv2.contourArea(approx) > 100:
                color = np.random.randint(0, 255, size=3).tolist()
                colors.append((approx, color))

                cv2.drawContours(output_image, [approx], -1, color, thickness=cv2.FILLED)
    return output_image, colors

def overlay_image(background, overlay, opacity):
    return cv2.addWeighted(background, 1 - opacity, overlay, opacity, 0)

finalSkeletonCleaned = cv2.cvtColor(finalSkeletonCleaned, cv2.COLOR_BGR2GRAY)

contours, _ = cv2.findContours(finalSkeletonCleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
colored_skeleton_cleaned, contour_colors = draw_colored_contours(finalSkeletonCleaned, contours)
cv2.imshow('Colored Final Skeleton Cleaned', colored_skeleton_cleaned)

colored_skeleton_cleaned_overlay = overlay_image(img2, colored_skeleton_cleaned, 0.15)
cv2.imshow('Colored Skeleton Cleaned Overlay', colored_skeleton_cleaned_overlay)


def click_event(event, x, y, flags, param):
    global colored_skeleton_cleaned, img2, contour_colors, colored_skeleton_cleaned_overlay

    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_color = colored_skeleton_cleaned[y, x]

        for i, (contour, color) in enumerate(contour_colors):
            if np.array_equal(pixel_color, color):

                cv2.drawContours(colored_skeleton_cleaned, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)

                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                aspect_ratio = float(w) / h if h != 0 else 0
                
                w = w*umPPx
                h = h*umPPx
                area = area*umPPx**2

                # print(f'width: {w}, height: {h}, area: {area}\n')
                contour_colors.pop(i)
                colored_skeleton_cleaned_overlay = overlay_image(img2, colored_skeleton_cleaned, 0.15)

                cv2.imshow('Colored Skeleton Cleaned Overlay', colored_skeleton_cleaned_overlay)
                cv2.imshow('Colored Final Skeleton Cleaned', colored_skeleton_cleaned)
                print(f"Contour {i} deleted.")
                break

cv2.setMouseCallback('Colored Skeleton Cleaned Overlay', click_event)

while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord('d'):
        cv2.imshow('Final Image', colored_skeleton_cleaned_overlay)
        cv2.waitKey(0)
        break
    elif key == 27: 
        break

cv2.destroyAllWindows()

contours_list = []

for contour, color in contour_colors:
    contours_list.append(contour)

heights = []
widths = []
areas = []
aspect_ratios = []

for contour in contours_list:
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    aspect_ratio = float(w) / h if h != 0 else 0
    
    w = w*umPPx
    h = h*umPPx
    area = area*umPPx**2

    heights.append(h)
    widths.append(w)
    areas.append(area)
    aspect_ratios.append(aspect_ratio)

with open("data.txt", "w") as file:
    file.write("Height\tWidth\tArea\tAspect_Ratio\n")

    for i in range(len(heights)):
        file.write(f"{heights[i]}\t{widths[i]}\t{areas[i]}\t{aspect_ratios[i]}\n")



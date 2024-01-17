import cv2 as cv
import numpy as np

#Feature Detection Algorithms:
# ORB - SIFT - FAST - BRISK - KAZE - AKAZE - harris 


#ORB

img = cv.imread('C:\\Users\\CEC\\Desktop\\im\\cv1.jpg')
img = cv.resize(img, (img.shape[1] // 6, img.shape[0] // 6))
orb = cv.ORB_create()
keypoints, descriptors = orb.detectAndCompute(img, None)
img2 = cv.drawKeypoints(img, keypoints, None,color=(0, 255, 0))
cv.imshow("orb", img2)
cv.waitKey(0)

################################################################################################################

#SIFT

img = cv.imread('C:\\Users\\CEC\\Desktop\\im\\cv1.jpg')
img = cv.resize(img, (img.shape[1] // 6, img.shape[0] // 6))
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
keypoints , descriptors = sift.detectAndCompute(gray, None)
img = cv.drawKeypoints(gray, keypoints, img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('SIFT', img)
cv.waitKey(0)

##################################################################################################################

#FAST

img = cv.imread('C:\\Users\\CEC\\Desktop\\im\\cv1.jpg')
img = cv.resize(img, (img.shape[1] // 6, img.shape[0] // 6))
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
fast = cv.FastFeatureDetector_create()
keypoints = fast.detect(gray, None)
image = cv.drawKeypoints(gray, keypoints,outImage = None,  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('FAST', image)
cv.waitKey()

#########################################################################################################################

#BRISK

img = cv.imread('C:\\Users\\CEC\\Desktop\\im\\cv1.jpg')
img = cv.resize(img, (img.shape[1] // 6, img.shape[0] // 6))
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
brisk = cv.BRISK_create()
keypoints, descriptors = brisk.detectAndCompute(img, None)
image = cv.drawKeypoints(gray, keypoints,outImage = None,  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('BRISK', image)
cv.waitKey()

#############################################################################################################################

#KAZE

img = cv.imread('C:\\Users\\CEC\\Desktop\\im\\cv1.jpg')
img = cv.resize(img, (img.shape[1] // 6, img.shape[0] // 6))
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
kaze = cv.KAZE_create()
keypoints, descriptors = kaze.detectAndCompute(img, None)
image = cv.drawKeypoints(gray, keypoints,outImage = None,  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('KAZE', image)
cv.waitKey()

########################################################################################################################

#AKAZE

img = cv.imread('C:\\Users\\CEC\\Desktop\\im\\cv1.jpg')
img = cv.resize(img, (img.shape[1] // 6, img.shape[0] // 6))
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
akaze = cv.AKAZE_create()
keypoints, descriptors = akaze.detectAndCompute(img, None)
image = cv.drawKeypoints(gray, keypoints,outImage = None,  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('AKAZE', image)
cv.waitKey()

######################################################################################################################

#harris

img = cv.imread('C:\\Users\\CEC\\Desktop\\im\\cv1.jpg')
img = cv.resize(img, (img.shape[1] // 6, img.shape[0] // 6))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
harris_result = cv.cornerHarris(gray, 2, 3, 0.05)
harris_result_dilated = cv.dilate(harris_result, None)
cv.imshow('harris-result', harris_result)
cv.imshow('harris-result-dilated', harris_result_dilated)
filtering_threshold = 0.01
img_copy = img.copy()
filter = harris_result_dilated > filtering_threshold*harris_result_dilated.max()
img_copy[filter] = (0, 0, 255)
harris_result_dilated_filtered = np.zeros(img.shape, np.uint8) 
harris_result_dilated_filtered[filter] = 255
cv.imshow('harris_result_dilated_filtered', harris_result_dilated_filtered)
cv.imshow('harris', img_copy)
k = cv.waitKey(0)

#######################################################################################################

#استخراج الكيبون من الصور ذات الاتجاهات المختلفة 
# وأيضا استخراج الكيبون من الصور مع أبجكتات أخرى

##################يرجى عند الاختبار أن يعلق  بقية الصور عند تجريب أحدها#############

#فيما يلي عدة صور للأوبجيكت بوضعيات مختلفة 

img2 = cv.imread('C:\\Users\\CEC\\Desktop\\im\\1.jpg', 0)  
img2 = cv.resize(img2, (img2.shape[1] // 6, img2.shape[0] // 6))
akaze = cv.AKAZE_create()
keypoints2, descriptors2 = akaze.detectAndCompute(img2, None)
########

img2 = cv.imread('C:\\Users\\CEC\\Desktop\\im\\2.jpg', 0)  
img2 = cv.resize(img2, (img2.shape[1] // 6, img2.shape[0] // 6))
akaze = cv.AKAZE_create()
keypoints2, descriptors2 = akaze.detectAndCompute(img2, None)
########
img2 = cv.imread('C:\\Users\\CEC\\Desktop\\im\\cv2.jpg', 0)  
img2 = cv.resize(img2, (img2.shape[1] // 6, img2.shape[0] // 6))
akaze = cv.AKAZE_create()
keypoints2, descriptors2 = akaze.detectAndCompute(img2, None)
########
img2 = cv.imread('C:\\Users\\CEC\\Desktop\\im\\cv3.jpg', 0)  
img2 = cv.resize(img2, (img2.shape[1] // 6, img2.shape[0] // 6))
akaze = cv.AKAZE_create()
keypoints2, descriptors2 = akaze.detectAndCompute(img2, None)
#######
img2 = cv.imread('C:\\Users\\CEC\\Desktop\\im\\cv4.jpg', 0)  
img2 = cv.resize(img2, (img2.shape[1] // 6, img2.shape[0] // 6))
akaze = cv.AKAZE_create()
keypoints2, descriptors2 = akaze.detectAndCompute(img2, None)

###################################################################################
#Template Matching Algorithms

#brute force - KNN - FLANN - Homography

# في هذا المثال قمت بتثبيت عدد خطوط الماتش إلى 20 خط ، وهذا ما وجدته مناسبا لالتقاط الميزات المهمة لكي لا تكون الميزات كثيرة بلا فائدة
# ولعدم رسم أكثر من خط على نفس الميزة 
# وهكذا أوضح لملاحظة الفروقات بين الخوارزميات 


#brute force 

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
img_matches = cv.drawMatches(img, keypoints, img2, keypoints2, matches[:20], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow('brute force ', img_matches)
cv.waitKey(0)
######################

#KNN

bf = cv.BFMatcher()
matches = bf.knnMatch(descriptors, descriptors2, k=2)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
img_matches = cv.drawMatches(img, keypoints, img2, keypoints2, good_matches[:20], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow('KNN', img_matches)
cv.waitKey(0) 

#####################

#FLANN

index_params = dict(algorithm=6, table_number=6,key_size=12,multi_probe_level=1)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(descriptors, descriptors2, k=2)
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
img_matches = cv.drawMatches(img, keypoints, img2, keypoints2, good_matches[:20], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)      
cv.imshow('FLANN', img_matches)
cv.waitKey(0) 

###########################

#Homography

bf = cv.BFMatcher()
matches = bf.knnMatch(descriptors, descriptors2, k=2)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0) 
h, w = gray.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv.perspectiveTransform(pts, M)
img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
img_matches = cv.drawMatches(img, keypoints, img2, keypoints2, good_matches[:20], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow('Homography', img_matches)
cv.waitKey(0)  
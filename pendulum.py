import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from tqdm import *

def rescaleFrame(frame, scale = 0.5):
    width = int(frame.shape[1] * scale) #.shape[1] refers to width of image, typecasting it into integer value
    height = int(frame.shape[0] * scale) #.shape[0] refers to height of image
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

def drawlines(img1, img2, lines, pts1, pts2):
    
    # r, c = img1.shape
    r = img1.shape[0]
    c = img1.shape[1]
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
      
    for r, pt1, pt2 in zip(lines, pts1, pts2):
          
        color = tuple(np.random.randint(0, 255, 3).tolist())
          
        x0, y0 = map(int, [0, -r[2] / r[1] ])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1] ])
          
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

def sum_of_abs_diff(pixel_vals_1, pixel_vals_2): #function to calculate the sum of absolute differences
    if pixel_vals_1.shape != pixel_vals_2.shape:
        return -1

    return np.sum(abs(pixel_vals_1 - pixel_vals_2))

def compare_blocks(y, x, block_left, right_array, block_size=5, SEARCH_BLOCK_SIZE = 56):
    # Get search range for the right image
    x_min = max(0, x - SEARCH_BLOCK_SIZE)
    x_max = min(right_array.shape[1], x + SEARCH_BLOCK_SIZE)
    first = True
    min_sad = None #sum of absolute differences
    min_index = None
    for x in range(x_min, x_max):
        block_right = right_array[y: y+block_size,
                                  x: x+block_size]
        sad = sum_of_abs_diff(block_left, block_right)
        if first:
            min_sad = sad
            min_index = (y, x)
            first = False
        else:
            if sad < min_sad:
                min_sad = sad
                min_index = (y, x)

    return min_index

def main():
    img_0 = cv.imread("pendulum/im0.png") #reading image 0
    resized_img_0 = rescaleFrame(img_0)
    gray_0 = cv.cvtColor(resized_img_0, cv.COLOR_BGR2GRAY)

    img_1 = cv.imread("pendulum/im1.png") #reading image 1
    resized_img_1 = rescaleFrame(img_1)
    gray_1 = cv.cvtColor(resized_img_1, cv.COLOR_BGR2GRAY)

    #Using ORB detector to extract the keypoints
    #creating ORB Detector
    orb = cv.ORB_create(nfeatures = 500) #by default it takes 500 as max no. of features

    key_points_1, descriptors_1 = orb.detectAndCompute(gray_0, None)
    key_points_2, descriptors_2 = orb.detectAndCompute(gray_1, None)


    # Draw matches on each image
    imgmatches_1 = cv.drawKeypoints(resized_img_0, key_points_1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow("Image 0 Keypoints", imgmatches_1)
    cv.imwrite("pendulum_image_outputs/keypoints_0.jpg", imgmatches_1)
    cv.waitKey(0)

    imgmatches_2 = cv.drawKeypoints(resized_img_1, key_points_2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow("Image 1 Keypoints", imgmatches_2)
    cv.imwrite("pendulum_image_outputs/keypoints_1.jpg", imgmatches_2)
    cv.waitKey(0)

    #Now, we match the features common to both images
    #Create a bf(Brute Force) Matcher
    # bf = cv.BFMatcher_create(cv.NORM_HAMMING) #cv.NORM_HAMMING is used for distance measurement
    bf = cv.BFMatcher()

    #compare and match descriptors from image 1 with image 2
    matching_points = bf.knnMatch(descriptors_1, descriptors_2, k=2)  #two nearnest neighbours


    #Finding out good matches, i.e., getting rid of point which aren't distinct enough
    good_matching_points = []

    # ptsLeft = [] #testing matches
    # ptsRight = [] #testing matches

    for m,n in matching_points:
        if m.distance < 0.6 * (n.distance):
            good_matching_points.append(m)
            # ptsLeft.append(key_points_1[m.trainIdx].pt) #testing matches
            # ptsRight.append(key_points_2[n.trainIdx].pt) #testing matches

    # print(good_matching_points)
    #good matching points gives a set of DObjects of the matching points in each case, we need to extract pixel coordinates of matching
    #points in both images

    # Initialize lists to store the coordinates from each image
    list_key_points_1 = []
    list_key_points_2 = []

    # for matches in good_matching_points:

    #     # Get the matching keypoints for each of the images
    #     img1_idx = matches.queryIdx
    #     img2_idx = matches.trainIdx


    #     # Get the point coordinates
    #     (x1, y1) = key_points_1[img1_idx].pt
    #     (x2, y2) = key_points_2[img2_idx].pt

    #     # Append the coordinates to each respective list
    #     list_key_points_1.append((x1, y1))
    #     list_key_points_2.append((x2, y2))

    list_key_points_1 = [key_points_1[mat.queryIdx].pt for mat in good_matching_points] 
    list_key_points_2 = [key_points_2[mat.trainIdx].pt for mat in good_matching_points]

    #choosing eight points from first image and getting their x and y coordinates
    x1_1 = list_key_points_1[0][0] #x1
    y1_1 = list_key_points_1[0][1] #y1
    x1_2 = list_key_points_1[1][0] #x2
    y1_2 = list_key_points_1[1][1] #y2
    x1_3 = list_key_points_1[2][0] #x3
    y1_3 = list_key_points_1[2][1] #y3
    x1_4 = list_key_points_1[3][0] #x4
    y1_4 = list_key_points_1[3][1] #y4
    x1_5 = list_key_points_1[4][0] #x5
    y1_5 = list_key_points_1[4][1] #y5
    x1_6 = list_key_points_1[5][0] #x6
    y1_6 = list_key_points_1[5][1] #y6
    x1_7 = list_key_points_1[6][0] #x7
    y1_7 = list_key_points_1[6][1] #y7
    x1_8 = list_key_points_1[7][0] #x8
    y1_8 = list_key_points_1[7][1] #y8

    #choosing eight points from second image and getting their x and y coordinates
    x2_1 = list_key_points_2[0][0] #x1'
    y2_1 = list_key_points_2[0][1] #y1'
    x2_2 = list_key_points_2[1][0] #x2'
    y2_2 = list_key_points_2[1][1] #y2'
    x2_3 = list_key_points_2[2][0] #x3'
    y2_3 = list_key_points_2[2][1] #y3'
    x2_4 = list_key_points_2[3][0] #x4'
    y2_4 = list_key_points_2[3][1] #y4'
    x2_5 = list_key_points_2[4][0] #x5'
    y2_5 = list_key_points_2[4][1] #y5'
    x2_6 = list_key_points_2[5][0] #x6'
    y2_6 = list_key_points_2[5][1] #y6'
    x2_7 = list_key_points_2[6][0] #x7'
    y2_7 = list_key_points_2[6][1] #y7'
    x2_8 = list_key_points_2[7][0] #x8'
    y2_8 = list_key_points_2[7][1] #y8'

    #Now we use these coordinates from both images to create a matrix A which will help us calculate the fundamental matrix
    A = np.array([
        [x1_1*x2_1, x1_1*y2_1, x1_1, y1_1*x2_1, y1_1*y2_1, y1_1, x2_1, y2_1, 1],
        [x1_2*x2_2, x1_2*y2_2, x1_2, y1_2*x2_2, y1_2*y2_2, y1_2, x2_2, y2_2, 1],
        [x1_3*x2_3, x1_3*y2_3, x1_3, y1_3*x2_3, y1_3*y2_3, y1_3, x2_3, y2_3, 1],
        [x1_4*x2_4, x1_4*y2_4, x1_4, y1_4*x2_4, y1_4*y2_4, y1_4, x2_4, y2_4, 1],
        [x1_5*x2_5, x1_5*y2_5, x1_5, y1_5*x2_5, y1_5*y2_5, y1_5, x2_5, y2_5, 1],
        [x1_6*x2_6, x1_6*y2_6, x1_6, y1_6*x2_6, y1_6*y2_6, y1_6, x2_6, y2_6, 1],
        [x1_7*x2_7, x1_7*y2_7, x1_7, y1_7*x2_7, y1_7*y2_7, y1_7, x2_7, y2_7, 1],
        [x1_8*x2_8, x1_8*y2_8, x1_8, y1_8*x2_8, y1_8*y2_8, y1_8, x2_8, y2_8, 1]])

    #taking the SVD of A matrix to get the fundamental matrix
    U, S, V = np.linalg.svd(A)
    f_values = V[-1,:] #last column values of V are the f values which form the fundamental matrix
    # f_values_array = np.array(f_values)
    Fundamental_matrix = np.reshape(f_values,(3,3)) #reshaping to 3*3 dimensions
    Fundamental_matrix = np.round(Fundamental_matrix, decimals = 3)

    # print(Fundamental_matrix)

    #Now lets calculate the essential matrix
    #for that we need the intrinsic matrix for the camera which is provided to us
    K_matrix = np.array([
        [1729.05, 0, -364.24],
        [0, 1729.05, 552.22],
        [0, 0, 1]])

    #Essential Matrix
    KT_matrix = K_matrix.transpose() # K transpose
    Essential_matrix = np.matmul(np.matmul(KT_matrix, Fundamental_matrix), K_matrix)

    # print(Essential_matrix)
    # Now we decompose the essential matrix into its respective rotational and translational matrices
    # So for that we need to take the SVD of Essential matrix
    U, S, V = np.linalg.svd(Essential_matrix)

    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]])

    #We get four combinations of the cameras' translation and Rotation matrices
    C1 = U[:,2]
    R1 = np.matmul(np.matmul(U, W), V.transpose())

    C2 = -1 * U[:,2]
    R2 = np.matmul(np.matmul(U, W), V.transpose())

    C3 = U[:,2]
    R3 = np.matmul(np.matmul(U, W.transpose()), V.transpose())

    C4 = -1 * U[:,2]
    R4 = np.matmul(np.matmul(U, W.transpose()), V.transpose())



    list_key_points_1 = np.int32(list_key_points_1)
    list_key_points_2 = np.int32(list_key_points_2)

    linesLeft = cv.computeCorrespondEpilines(list_key_points_2.reshape(-1, 1, 2), 2, Fundamental_matrix)
    linesLeft = linesLeft.reshape(-1, 3)
    img5, img6 = drawlines(gray_0, gray_1, linesLeft, list_key_points_1, list_key_points_2)


    linesRight = cv.computeCorrespondEpilines(list_key_points_1.reshape(-1, 1, 2), 1, Fundamental_matrix)
    linesRight = linesRight.reshape(-1, 3)
    
    img3, img4 = drawlines(gray_1, gray_0, linesRight, list_key_points_2, list_key_points_1)

    # plt.subplot(121), plt.imshow(img5)
    # plt.subplot(122), plt.imshow(img3)
    # plt.show()

    cv.imshow("epilines 0", img5)
    cv.imwrite("pendulum_image_outputs/epilines_0.jpg", img5)
    cv.waitKey(0)
    cv.imshow("epilines 1", img3)
    cv.imwrite("pendulum_image_outputs/epilines_1.jpg", img3)
    cv.waitKey(0)

    # rank = np.linalg.matrix_rank(Fundamental_matrix)
    # print(rank)

    #### Part 2 - Rectification ####
    h1, w1 = resized_img_0.shape[0], resized_img_0.shape[1]
    h2, w2 = resized_img_1.shape[0], resized_img_1.shape[1]

    #this function will return homography matrices for both images
    _, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(list_key_points_1), np.float32(list_key_points_2), Fundamental_matrix, imgSize = (w1, h1))

    print("H1 = ")
    print(H1)
    print("H2 = ")
    print(H2)
    #now we will use this homography matrices to warp the perspectives in both images
    img_0_rectified = cv.warpPerspective(img5, H1, (w1, h1))
    img_1_rectified = cv.warpPerspective(img3, H2, (w2, h2))

    cv.imshow("img0 rectified", img_0_rectified)
    cv.imwrite("pendulum_image_outputs/rectified_0.jpg", img_0_rectified)
    cv.waitKey(0)
    cv.imshow("img1 rectified", img_1_rectified)
    cv.imwrite("pendulum_image_outputs/rectified_1.jpg", img_1_rectified)
    cv.waitKey(0)

    #### Part 3 - Correspondence and depth ####

    #first we convert the image to arrays
    left_im_array = np.asarray(img_0_rectified)
    right_im_array = np.asarray(img_1_rectified)
    left_array = left_im_array.astype(int)
    right_array = right_im_array.astype(int)

    if left_array.shape != right_array.shape: #exception error if shape is not same
        raise "Left-Right image shape mismatch!"

    h, w, _ = left_array.shape
    disparity_map = np.zeros((h, w)) #creating a disparity map of all zeros

    BLOCK_SIZE = 20 #Block of pixels width and height
    SEARCH_BLOCK_SIZE = 56 #size of the searching block

    # Go over each pixel position
    for y in tqdm(range(BLOCK_SIZE, h-BLOCK_SIZE)):
        for x in range(BLOCK_SIZE, w-BLOCK_SIZE):
            block_left = left_array[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE] #subsection of left image as block
            min_index = compare_blocks(y, x, block_left, right_array, block_size=BLOCK_SIZE, SEARCH_BLOCK_SIZE = SEARCH_BLOCK_SIZE)
            disparity_map[y, x] = abs(min_index[1] - x)

    plt.imshow(disparity_map, cmap='hot', interpolation='nearest')
    plt.savefig('pendulum_image_outputs/depth_image.png')
    plt.show()

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
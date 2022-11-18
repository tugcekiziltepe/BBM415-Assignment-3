from stitcher import stitcher
import cv2
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

   image1 = cv2.imread("images/input2.png") # image on the middle
   image2 = cv2.imread("images/input3.png") # image on the right
   image3 = cv2.imread("images/input1.png") # image on the left

   # Call the stitcher function
   stitcher(image1, image2, image3)
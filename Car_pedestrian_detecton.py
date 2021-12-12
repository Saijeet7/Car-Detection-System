import cv2

#Our Image
img_file = 'tesla.jpg'

#Pre-trained car classifier
classifier_file = 'car_detector.xml'

#Create opencv image
img = cv2.imread(img_file)

#Convert to black and white
black_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#Display the imgae 
cv2.imshow('Car detector',black_white)


cv2.waitKey()
#  Our pre-trained
print("Code Completed")
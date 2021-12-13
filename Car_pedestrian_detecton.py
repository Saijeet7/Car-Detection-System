import cv2

#Our Image
img_file = 'car.jpg'

#Pre-trained car classifier
classifier_file = 'car_detector.xml'

#Create opencv image
img = cv2.imread(img_file)

#Convert to black and white
black_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#detect cars
cars = car_tracker.detectMultiScale(black_white)
print(cars)

#draw rectangles around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x,y),(x+w, y+h),(0, 0, 255),2)

#Display the imgae 
cv2.imshow('Car detector',img)


cv2.waitKey()
#  Our pre-trained
print("Code Completed")
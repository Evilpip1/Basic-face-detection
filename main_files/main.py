from cv2 import cv2

face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("3_people.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_detect.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    roi_gray = gray[x:x+w, y:y+h]
    roi_color = img[y:y+h, x:x+w]

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

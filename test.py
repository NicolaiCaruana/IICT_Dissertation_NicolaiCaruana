import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
from keras.models import load_model
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
threshold=0.90
cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font=cv2.FONT_HERSHEY_COMPLEX
model = load_model('MyTrainingModel.h5')

def preprocessing(img):
    img=img.astype("uint8")
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img = img/255
    return img


def get_className(classNo):
	if classNo==0:
		return "De Ashi Barai"
	elif classNo==1:
		return "Uki Goshi"


while True:
	sucess, imgOrignal=cap.read()
	faces = facedetect.detectMultiScale(imgOrignal,1.3,5)
	for x,y,w,h in faces:
		crop_img=imgOrignal[y:y+h,x:x+h]
		img=cv2.resize(crop_img, (32,32))
		img=preprocessing(img)
		img=img.reshape(1, 32, 32, 1)
		prediction=model.predict(img)
  
		predict_x=model.predict(img) 
		classes_x=np.argmax(predict_x,axis=1)  
		#classIndex=model.predict_classes(img)
  
		probabilityValue=np.amax(prediction)
		if probabilityValue>threshold:
			if classes_x==0:
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
				cv2.putText(imgOrignal, str(get_className(classes_x)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			elif classes_x==1:
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
				cv2.putText(imgOrignal, str(get_className(classes_x)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
	cv2.imshow("Result",imgOrignal)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()















import numpy as np

import cv2, os 
from PIL import Image




fL = 'datawajah'
lF = 'train'
kamera = cv2.VideoCapture(0)
wajah = cv2.CascadeClassifier('wajah.xml')
hidung = cv2.CascadeClassifier('noise.xml')

kamera.set(3,640)
kamera.set(4,480)



def Ambilgmbr():
    retV, frame = kamera.read()
    

    gelap = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceID = input("Masukkan ID : ")
    print("Wajah tetap didepan webcam. Tunggu proses pengambilan gambar ")
    recData = 1

    while True :
        
        
        faces = wajah.detectMultiScale(gelap, 1.3, 5)
        for (x, y, w, h) in faces:
            nose = hidung.detectMultiScale(gelap, 1.18, 35)
            if len (nose)>0:
                masker = False
            else :
                masker = True
            
            
                if masker:
                    frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255),3)
                    frame =  cv2.putText(frame, 'Pakai Masker', (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0),1)
                
                else:
                    frame =  cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255),3)
                    frame =  cv2.putText(frame, 'Tidak Pakai Masker', (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0),1)
                #     namaFile = 'Wajah  '+str(faceName)+' '+str(recData)+'.jpg'    
                #     cv2.imwrite(fL+'/'+namaFile, frame)
                #     recData =+ 1
                
                for (nx, ny, nw, nh ) in nose:
                    frame = cv2.rectangle(frame, (nx,ny), (nx+nw, nh+nh), (0,255,255),1)
        
            namaFile = 'Wajah'+'.'+str(faceID)+"."+str(recData)+'.jpg' 
            cv2.imwrite(fL+'/'+namaFile, frame)
            recData +=1   
            
            
        cv2.imshow('Kamera', frame)
        M = cv2.waitKey(1) & 0xFF 
        if M == 27 or M== ord('c'):
            break
        elif recData> 20:
            break
            
            
    print('Pengambilan Data Selesai')
    kamera.release()
    cv2.destroyAllWindows()
    

def getImageLabel(path):
    fL = 'datawajah'
    lF = 'train'
    kamera = cv2.VideoCapture(0)
    wajah = cv2.CascadeClassifier('wajah.xml')
    hidung = cv2.CascadeClassifier('noise.xml')

    kamera.set(3,648)
    kamera.set(4,480)
    retV, frame = kamera.read()
    gelap = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imagePaths  =[os.path.join(path,f)for f in os.listdir(path)]
    faceSample=[]
    FaceIDs = []
    for imagePath in imagePaths :
        PILImg = Image.open(imagePath).convert('L')
        imgNum = np.array(PILImg, 'uint8')
        faceID= int(os.path.split(imagePath)[-1].split(".")[1])
        faces =  wajah.detectMultiScale(imgNum)
        for (x, y, w, h) in faces :
            faceSample.append(imgNum[y:y+h, x:x+w])
            FaceIDs.append(faceID)
        return faceSample,FaceIDs
    
def train():
    fL = 'datawajah'
    lF = 'train'
    kamera = cv2.VideoCapture(0)
    

    # kamera.set(3,700)
    # kamera.set(4,520)
    retV, frame = kamera.read()
    gelap = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceDetector = cv2.CascadeClassifier('wajah.xml')
    
    
    print("Mesin melakuakan training data ")
    hidung = cv2.CascadeClassifier('noise.xml')
    fac,ids = getImageLabel(fL)
    print(ids)
    faceRecognizer.train(fac, np.array(ids))
    faceRecognizer.write(lF+'/training.xml')
    print ('Data telah ditraining',format(len(np.unique(ids))))
    

def recog():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('train/training.xml')
    
    faceCascade = cv2.CascadeClassifier('wajah.xml')
    

    font = cv2.FONT_HERSHEY_SIMPLEX

    #iniciate id counter
    id = 0

    # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['None', 'Faisal', 'Paula', 'Ilza', 'Z', 'W'] 

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:

        ret, img =cam.read()
        img = cv2.flip(img, 1) # Flip vertically

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (round(minW), round(minH)),
        )
        nose = hidung.detectMultiScale(gray, 1.18, 35)
        if len (nose)>0:
                masker = False
        else :
                masker = True

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            
            if (confidence < 100) and masker:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "Salah"
                confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 1)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        
        cv2.imshow('camera',img) 

        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27 or k == "c":
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

while True:
        print("==========================================")
        print( " ||Pilih Dengan Angka Nomor \n","||1. Ambil Data Gambar\n","||2.Train Data\n","||3.Test \n","||4.Keluar")
        print("==========================================\n")
        plh = input ("Silahkan Pilih : ")
        if plh == "1":
            Ambilgmbr()
            
        elif plh == "2":
            train()
        elif plh == "3":
            recog()
        elif plh == "4":
            break
        else :
            print("Salah pilih")
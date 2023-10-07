import cv2 
import os
import face_recognition

def takeImage(name,IDS):
    video=cv2.VideoCapture(0)

    facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    if not os.path.exists(f'datasets') :
        os.mkdir(f"datasets")


    ret=True
    while ret:

        ret,frame=video.read()

        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        cv2.rectangle(frame,(0,0),(800,50),(0,0,0),-1)
        
        for (x,y,w,h) in faces:

            x,y,w,h=x-10,y-10,w+50,h+50

            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            
            cv2.putText(frame,"Press S to save and exit",(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            if cv2.waitKey(1)==ord("s"):
                cv2.imwrite(f'datasets\\{name}_{IDS}.jpg', gray[y:y+h, x:x+w])
                print("Dataset Collection Done..................")
                ret=False

        cv2.imshow("Frame",frame)

        if cv2.waitKey(1)==27:
            break


    video.release()
    cv2.destroyAllWindows()

def testImages():
    print("Cheking Datasets........")
    path = 'datasets'
    images = []
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)

    x=-1
    try:
        for img in images:
            x+=1
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_recognition.face_encodings(img)[0]
    except:
        print(f"There is an error detecting a face in image '{myList[x]}'")
        ch=input(f"Do you want to delete the file {myList[x]}. Y/N ")
        if ch.lower()=="y":
            os.remove(f'{path}/{myList[x]}')

        return False
    
    return True

if __name__=="__main__":
    if testImages():
        print("Data is Good")

        name = input("Enter Your Name: ")
        Ids = input("Enter Your Registarion Number: ")
        takeImage(name,Ids)

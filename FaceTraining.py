import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime,date

def FaceRecog():
    path = 'datasets'
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    print(classNames)

    datetoday = date.today().strftime("%m_%d_%y")
    if not os.path.isdir('Attendance'):
        os.makedirs('Attendance')
    if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
        with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
            f.write('Name,Roll,Time')


    def findEncodings(images):
        try:
            encodeList = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(img)[0]
                encodeList.append(encode)
            return encodeList
        except:
            print("There is an error detecting a face")

    def markAttendance(name,ids):
        with open (f'Attendance/Attendance-{datetoday}.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                print(name)
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{ids},{dtString}')

    encodeListKnown = findEncodings(images)
    print('Encoding Complete')

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.40, 0.40)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            # matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)
            print(matchIndex)

            if faceDis[matchIndex]< 0.50:
                name = classNames[matchIndex].upper().split("_")[0]
                ids=classNames[matchIndex].upper().split("_")[1]
                markAttendance(name,ids)
            else: 
                name = 'Unknown'
                
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1 *2,int(x2 * 2.5), int(y2 * 3), x1 * 2 
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

        cv2.imshow('Webcam', img)

        if cv2.waitKey(1)==27 or cv2.waitKey(1)==ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__=="__main__":
    FaceRecog()
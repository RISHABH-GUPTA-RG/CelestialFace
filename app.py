from FaceTraining import FaceRecog
from datacollect import takeImage,testImages
from flask import Flask,render_template,redirect,request,Response
import cv2

app=Flask(__name__,template_folder="./templates",static_folder="./static")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/add',methods=['POST'])
def addnewuser():
    name=request.form["newusername"]
    ids=request.form["newuserid"]
    if testImages():
        takeImage(name,ids)
        testImages()
    return redirect('/')
    

@app.route('/recog')
def FacialRecog():
    FaceRecog()
    return redirect('/')

@app.route("/about")
def About():
    return render_template("about.html")

# def gen_frames(camera):  
    
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

# @app.route('/video')
# def video():
#     camera=cv2.VideoCapture(0)
#     render_template("video.html")
#     return Response(gen_frames(camera),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
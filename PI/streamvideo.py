import cv2
from flask import Flask, Response

app = Flask(__name__)

#Open the camera
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        #Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        #Yield as MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
@app.route('/video_feed')

def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    #Run at 0.0.0.0 on all network interfaces
    app.run(host='0.0.0.0', port=5000, threaded=True)


#ACCESS through http://IPADDRESS:5000/video_feed
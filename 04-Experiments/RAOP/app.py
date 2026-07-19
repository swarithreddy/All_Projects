from flask import Flask, render_template, request, session, redirect, url_for
from geopy.distance import geodesic
import cv2
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__, template_folder='templates1')
app.secret_key = 'your_secret_key'

# Face recognition setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
orb = cv2.ORB_create()
known_faces_dir = 'known_faces'
known_faces_data = []

# Classroom GPS location
CLASSROOM_LOCATION = (17.338094233780147, 78.54301348765571)
ALLOWED_DISTANCE_METERS = 20

# Load known faces
for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith(('.jpg', '.png')):
        name = os.path.splitext(filename)[0]
        img_path = os.path.join(known_faces_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            continue
        x, y, w, h = faces[0]
        face_roi = cv2.resize(gray[y:y+h, x:x+w], (150, 150))
        kp, des = orb.detectAndCompute(face_roi, None)
        if des is not None:
            known_faces_data.append({'name': name, 'des': des, 'kp': kp})

@app.route('/', methods=['GET'])
def login():
    session.clear()
    return render_template('login_timetable.html')

@app.route('/verify', methods=['POST'])
def verify():
    subject = request.form.get('subject')
    session['subject'] = subject
    session['attempts'] = 0
    return render_template('verify.html', subject=subject)

@app.route('/verify_face', methods=['POST'])
def verify_face():
    subject = session.get('subject')
    session['attempts'] += 1
    cap = cv2.VideoCapture(0)
    match_found = False
    matched_name = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face = cv2.resize(gray[y:y+h, x:x+w], (150, 150))
            kp2, des2 = orb.detectAndCompute(face, None)
            if des2 is None:
                continue
            best_distance = float('inf')
            for known in known_faces_data:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(known['des'], des2)
                if matches:
                    avg_distance = sum([m.distance for m in matches]) / len(matches)
                    if avg_distance < best_distance:
                        best_distance = avg_distance
                        matched_name = known['name']
            if best_distance < 60:
                match_found = True
                break

        if match_found or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not match_found:
        if session['attempts'] >= 3:
            return redirect(url_for('login'))
        return render_template('unsuccessful.html', subject=subject, remaining=3 - session['attempts'])

    # Location verification
    latitude = float(request.form.get('latitude', 0.0))
    longitude = float(request.form.get('longitude', 0.0))
    scanned_location = (latitude, longitude)
    distance = geodesic(CLASSROOM_LOCATION, scanned_location).meters

    if distance <= ALLOWED_DISTANCE_METERS:
        session['matched_name'] = matched_name  # Store the matched name for later use
        return render_template('scan_qr.html', subject=subject)
    return render_template('failure.html')

@app.route('/scan_qr', methods=['POST'])
def scan_qr():
    cap = cv2.VideoCapture(0)
    qr_detected = False
    detector = cv2.QRCodeDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and decode QR codes using OpenCV
        data, bbox, _ = detector.detectAndDecode(frame)
        if data:  # If any QR code is detected
            qr_detected = True
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if qr_detected:
        # Store attendance in Excel
        student_name = session.get('matched_name', 'Unknown')
        subject = session.get('subject', 'Unknown')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Prepare data for Excel
        new_entry = {
            'Student Name': student_name,
            'Subject': subject,
            'Timestamp': timestamp
        }

        # Load or create the Excel file
        excel_file = 'attendance.xlsx'
        try:
            df = pd.read_excel(excel_file)
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        except FileNotFoundError:
            df = pd.DataFrame([new_entry])
        
        # Save to Excel
        df.to_excel(excel_file, index=False)

        return render_template('success.html')
    return render_template('failure.html')

if __name__ == '__main__':
    app.run(debug=True)
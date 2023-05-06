from flask import Flask, render_template, request,send_file
import cv2
import face_recognition
import os
import io
import pandas as pd
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app=Flask(__name__,static_folder='static')

# Load known face encodings from static folder
known_faces = []
known_names = []
for filename in os.listdir('static/images'):
    image = face_recognition.load_image_file(os.path.join('static/images', filename))
    face_encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(face_encoding)
    known_names.append(os.path.splitext(filename)[0])

# Initialize attendance DataFrame
attendance = pd.DataFrame(columns=['Name', 'Attendance', 'Time'])
print(cv2.__version__)
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route('/live_attendance', methods=['GET', 'POST'])
def live_attendance():
    global attendance  # declare attendance as global
    
    if request.method == 'POST':
        face_count_live=0
        cap = cv2.VideoCapture(0)
        while True:
            # Capture frame from camera
            ret, frame = cap.read()

            # Find faces in frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            # Loop over face encodings to see if they match any known faces
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                name = "Unknown"

                # Check if there is a match
                if True in matches:
                    match_index = matches.index(True)
                    name = known_names[match_index]
                    if name not in attendance['Name'].values:
                        # Add attendance record with current time
                        now = datetime.datetime.now()
                        attendance = attendance.append({'Name': name, 'Attendance': 'Present', 'Time': now}, ignore_index=True)
                        face_count_live += 1  # increment face count by 1

                # Draw a box around the face and label it
                top, right, bottom, left = face_recognition.face_locations(frame)[0]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)

            # Display the resulting image with face count
            cv2.putText(frame, f"Face count: {face_count_live}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow('Face Recognition Attendance System', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release camera and close window
        cap.release()
        cv2.destroyAllWindows()

        # Render attendance information on the web app
        return render_template('live_attendance.html', attendance=attendance)
    else:
        return render_template('live_attendance.html')



# Upload video file and perform face recognition
@app.route('/recorded_video', methods=['GET', 'POST'])
def recorded_video():
    global attendance  # declare attendance as global
    if request.method == 'POST':
        face_count=0
        file = request.files['video']
        # Save uploaded file to static folder
        filename = file.filename
        file.save(os.path.join('static', filename))
        # Read video file and perform face recognition
        cap = cv2.VideoCapture(os.path.join('static', filename))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert color from BGR to RGB for face_recognition library
            rgb_frame = frame[:, :, ::-1]
            # Detect faces in the frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            for face_encoding in face_encodings:
                # Compare face encoding with known faces
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                if True in matches:
                    match_index = matches.index(True)
                    name = known_names[match_index]
                    if name not in attendance['Name'].values:
                        now = datetime.datetime.now()
                        attendance = attendance.append({'Name': name, 'Attendance': 'Present', 'Time': now}, ignore_index=True)
                        face_count += 1  # increment face count by 1
                # Draw a box around the face and label it
                top, right, bottom, left = face_locations[0]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
            # Display the resulting image with face count
            cv2.putText(frame, f"Face count: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)   
            cv2.imshow('Recorded Video', frame)
            # Exit loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release capture and destroy windows
        cap.release()
        cv2.destroyAllWindows()
        os.remove(os.path.join('static', filename))
        # Render attendance information on the web app
        return render_template('live_attendance.html', attendance=attendance)
    else:
        return render_template('live_attendance.html')

@app.route('/download_attendance')
def download_attendance():
    csv_string = attendance.to_csv(index=False)
    return send_file(
        # Use BytesIO to treat the string as a file-like object
        io.BytesIO(csv_string.encode('utf-8')),
        mimetype='text/csv',
        attachment_filename='attendance.csv',
        as_attachment=True
    )

if __name__=='__main__':
    app.run(debug=True)
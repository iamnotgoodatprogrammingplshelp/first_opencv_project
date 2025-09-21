import face_recognition
image = face_recognition.load_image_file("/Users/user/Desktop/open_cv/camera facial detection/dad.jpg")
face_locations = face_recognition.face_locations(image)
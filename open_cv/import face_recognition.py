import face_recognition
import numpy as np
import cv2
import pyscreenshot as pys



while True:
    img= pys.grab()
    img_np=np.array(img)
    known_image = face_recognition.load_image_file("/Users/user/Desktop/open_cv/facial recognition/obama.jpg")
    unknown_image = face_recognition.load_image_file("/Users/user/Desktop/open_cv/facial recognition/uknown.jpg")
    biden_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
    results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
    forcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', forcc, 8, (1920,1080))

    #frame= cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Screen', img_np)
    out.write(img_np)


    if cv2.waitKey(20) & 0xFF==ord('q'):
        break

out.release()
cv2.destroyAllWindows()
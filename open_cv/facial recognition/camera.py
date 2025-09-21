import face_recognition
import cv2
import numpy as np
import os
import sys

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True
        
    def load_known_faces(self, faces_folder):
        """Load known faces from a folder containing images"""
        if not os.path.exists(faces_folder):
            print(f"Warning: Folder {faces_folder} does not exist")
            return False
            
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        loaded_faces = 0
        
        for filename in os.listdir(faces_folder):
            if filename.lower().endswith(supported_formats):
                try:
                    # Load image
                    image_path = os.path.join(faces_folder, filename)
                    image = face_recognition.load_image_file(image_path)
                    
                    # Get face encodings
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if face_encodings:
                        # Use the first face found in the image
                        face_encoding = face_encodings[0]
                        
                        # Use filename (without extension) as the person's name
                        name = os.path.splitext(filename)[0]
                        
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(name)
                        loaded_faces += 1
                        print(f"Loaded face: {name}")
                    else:
                        print(f"No face found in {filename}")
                        
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
                    
        print(f"Total faces loaded: {loaded_faces}")
        return loaded_faces > 0
    
    def add_known_face(self, image_path, name):
        """Add a single known face"""
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
                print(f"Added face: {name}")
                return True
            else:
                print(f"No face found in {image_path}")
                return False
                
        except Exception as e:
            print(f"Error adding face {name}: {str(e)}")
            return False
    
    def recognize_faces(self, frame):
        """Recognize faces in a frame"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert BGR to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        
        if self.face_locations:
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
        else:
            self.face_encodings = []
        
        self.face_names = []
        for face_encoding in self.face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            
            if True in matches:
                # Find the best match
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                    name = f"{name} ({confidence:.2f})"
            
            self.face_names.append(name)
    
    def draw_results(self, frame):
        """Draw face recognition results on frame"""
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Choose color based on recognition
            color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)
            
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def run_webcam(self):
        """Run face recognition on webcam feed"""
        if not self.known_face_encodings:
            print("No known faces loaded. Please add some faces first.")
            return
            
        # Initialize webcam
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting face recognition. Press 'q' to quit, 's' to save screenshot")
        
        try:
            while True:
                ret, frame = video_capture.read()
                
                if not ret:
                    print("Error: Could not read frame from webcam")
                    break
                
                # Only process every other frame for better performance
                if self.process_this_frame:
                    self.recognize_faces(frame)
                
                self.process_this_frame = not self.process_this_frame
                
                # Draw results
                frame = self.draw_results(frame)
                
                # Add instructions
                cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Face Recognition', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    filename = f"screenshot_{np.random.randint(1000, 9999)}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved as {filename}")
                    
        except KeyboardInterrupt:
            print("\nStopping face recognition...")
        
        finally:
            video_capture.release()
            cv2.destroyAllWindows()

def main():
    # Create face recognition system
    fr_system = FaceRecognitionSystem()
    
    # Option 1: Load faces from a folder
    # fr_system.load_known_faces("known_faces")
    
    # Option 2: Add individual faces (update these paths)
    # fr_system.add_known_face("path/to/person1.jpg", "Person 1")
    # fr_system.add_known_face("path/to/person2.jpg", "Person 2")
    
    # Example with some common paths (update these)
    faces_to_add = [
        ("obama.jpg", "Barack Obama"),
        ("biden.jpg", "Joe Biden"),
        # Add more faces here
    ]
    
    for image_path, name in faces_to_add:
        if os.path.exists(image_path):
            fr_system.add_known_face(image_path, name)
        else:
            print(f"Image not found: {image_path}")
    
    # Run the webcam face recognition
    fr_system.run_webcam()

if __name__ == "__main__":
    main()
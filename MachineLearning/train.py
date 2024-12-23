import imageio
import cv2
import face_recognition
import time
import os


def detect_and_save_faces(frame_cv2, frame_number, output_folder='captured_faces'):
    # Detect face locations in the frame
    face_locations = face_recognition.face_locations(frame_cv2)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    face_encodings = []
    # Iterate through each detected face and save it
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Crop the face from the frame
        face_image = frame_cv2[top:bottom, left:right]

        # Save the cropped face
        face_filename = f'{output_folder}/face_{frame_number}_{i + 1}.jpg'
        cv2.imwrite(face_filename, face_image)
        print(f'Saved {face_filename}')

        # Get the face encoding
        encoding = face_recognition.face_encodings(frame_cv2, [(top, right, bottom, left)])[0]
        face_encodings.append(encoding)

    return face_encodings, len(face_locations)


# Compare two face encodings
def compare_faces(reference_encoding, face_encodings, tolerance=0.6):
    for i, encoding in enumerate(face_encodings):
        distance = face_recognition.face_distance([reference_encoding], encoding)[0]
        print(f'Face {i + 1} distance: {distance}')
        if distance < tolerance:
            print(f"Face {i + 1} matches the reference image!")
        else:
            print(f"Face {i + 1} does not match the reference image.")


# Main video capture logic
start_time = time.time()
cap = imageio.get_reader('<video0>')

# Load and encode the reference image
reference_image_path = 'D:\MachineLearning\captured_faces/face_0_1.jpg'
reference_image = face_recognition.load_image_file(reference_image_path)
reference_encoding = face_recognition.face_encodings(reference_image)[0]

for frame_number, frame in enumerate(cap):
    # Convert the frame to BGR format (for OpenCV)
    frame_cv2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Detect and save faces, and get their embeddings
    face_encodings, faces_detected = detect_and_save_faces(frame_cv2, frame_number)
    print(f'Detected {faces_detected} face(s) in frame {frame_number}')

    # Compare detected faces with the reference image
    if faces_detected > 0:
        compare_faces(reference_encoding, face_encodings)

    # Display the frame with detected faces (if any)
    for (top, right, bottom, left) in face_recognition.face_locations(frame_cv2):
        cv2.rectangle(frame_cv2, (left - 25, top - 100), (right + 25, bottom + 25), (0, 255, 0), 2)
        face = frame_cv2[top - 100:bottom + 25, left - 25:right + 25]
        image_filename = f'captured_frame_{frame_number}.jpg'
        cv2.imwrite(image_filename, face)
        print(f'Saved {image_filename}')

    capture_time = time.time()
    elapsed_time = capture_time - start_time
    print(f'Elapsed time: {elapsed_time}')

    # Break after the first frame is processed if faces are detected
    if faces_detected > 0:
        break

cv2.destroyAllWindows()

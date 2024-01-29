import os
import face_recognition
import shutil
import numpy as np

def find_images(directory):
    """查找目录中的图像文件"""
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    return image_files

def group_images_by_face(root_directory):
    known_face_encodings = []
    known_face_names = []

    for current_directory, subdirectories, files in os.walk(root_directory):
        print(f"Processing directory: {current_directory}")
        image_files = find_images(current_directory)

        for image_file in image_files:
            try:
                image_path = os.path.join(current_directory, image_file)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                    name = None

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]
                    else:
                        name = f"person_{len(known_face_encodings)+1}"
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(name)

                    output_directory = os.path.join(root_directory, name)
                    if not os.path.exists(output_directory):
                        os.makedirs(output_directory)

                    output_path = os.path.join(output_directory, image_file)
                    shutil.copy(image_path, output_path)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

if __name__ == "__main__":
    target_directory = "目录"  # 确保这里是你的目标目录路径
    group_images_by_face(target_directory)

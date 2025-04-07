import os
import cv2

def color_to_lineart(color_dir, line_dir):
    os.makedirs(line_dir, exist_ok=True)
    files = sorted(os.listdir(color_dir))
    for i, file in enumerate(files):
        color_path = os.path.join(color_dir, file)
        line_path = os.path.join(line_dir, file.replace(".jpg", ".png"))

        img = cv2.imread(color_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        lineart = cv2.bitwise_not(edges)
        cv2.imwrite(line_path, lineart)

color_to_lineart("anime_dataset/color", "anime_dataset/line")

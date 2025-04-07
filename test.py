import os
import cv2
import urllib.request
import numpy as np
from PIL import Image
from tqdm import tqdm

os.makedirs("dataset/color", exist_ok=True)
os.makedirs("dataset/line", exist_ok=True)

image_urls = []
for i in range(20):
    url = f"https://picsum.photos/256/256?random={i}"
    image_urls.append(url)

def make_lineart(img, low=50, high=150):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, low, high)
    lineart = cv2.bitwise_not(edges)
    return lineart


def enhanced_lineart(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(img, kernel)
    blurred = cv2.GaussianBlur(dilated, (3, 3), 0)
    return blurred


def xdog(img, k=1.6, sigma=0.5, p=21, epsilon=-0.1, phi=10):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(img, (p, p), sigma)
    g2 = cv2.GaussianBlur(img, (p, p), sigma*k)
    diff = g1 - g2
    diff = diff / (np.max(diff) + 1e-6)
    xdog = 1.0 + np.tanh(phi*(diff + epsilon))
    xdog = (xdog*255).clip(0, 255).astype(np.uint8)
    return xdog


for i, url in enumerate(tqdm(image_urls)):
    path_color = f"dataset/color/img_{i:02}.png"
    path_line  = f"dataset/line/img_{i:02}.png"

    urllib.request.urlretrieve(url, path_color)
    img = cv2.imread(path_color)
    lineart = make_lineart(img, 30, 80)
    # lineart = enhanced_lineart(img)
    # lineart = xdog(img)
    cv2.imwrite(path_line, lineart)

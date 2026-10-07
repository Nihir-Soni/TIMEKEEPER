import cv2
import numpy as np
import random

def apply_blur(img, severity=1.0):
    # severity mapped to kernel size (3 to 11)
    k_size = int(3 + 2 * int(severity * 4))
    return cv2.GaussianBlur(img, (k_size, k_size), 0)

def apply_noise(img, severity=1.0):
    noise = np.random.normal(0, 15 * severity, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def apply_jpeg_compression(img, severity=1.0):
    # severity 0.0 -> quality 90, severity 1.0 -> quality 10
    quality = int(90 - (severity * 80))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)

def apply_fading(img, severity=1.0):
    # Reduce contrast, shift brightness
    alpha = 1.0 - (0.4 * severity)
    beta = 50 * severity
    faded = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return faded

def apply_scratches(img, severity=1.0):
    scratched = img.copy()
    h, w = img.shape[:2]
    num_scratches = int(severity * 20)
    for _ in range(num_scratches):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = x1 + random.randint(-50, 50), y1 + random.randint(-150, 150)
        thickness = random.randint(1, 3)
        color = random.choice([(255, 255, 255), (0, 0, 0), (200, 200, 200)])
        cv2.line(scratched, (x1, y1), (x2, y2), color, thickness)
    return scratched

def apply_missing_regions(img, severity=1.0):
    damaged = img.copy()
    h, w = img.shape[:2]
    num_holes = int(severity * 5)
    for _ in range(num_holes):
        x, y = random.randint(0, w), random.randint(0, h)
        r = random.randint(10, int(50 * severity) + 10)
        cv2.circle(damaged, (x, y), r, (255, 255, 255), -1)  # white hole simulating torn photo
    return damaged

def apply_resolution_degradation(img, severity=1.0):
    h, w = img.shape[:2]
    scale = 1.0 - (0.7 * severity) # Down to 30% resolution
    small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return restored

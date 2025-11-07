import json
import os.path
import cv2
import sys
import json
import easyocr
from ultralytics import YOLO
import numpy
model = YOLO("yolo11n.pt")

def detect_shot_cuts(video_path):
    """
    Detects Hard Cuts in the Videos using Histogram Difference.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"The video cannot be opened from : {video_path}")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return 0

    previous_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    previous_hist = cv2.calcHist([previous_gray],[0],None,[256],[0,256])
    shot_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        diff = cv2.compareHist(hist,previous_hist,cv2.HISTCMP_CORREL)
        threshold = 0.7
        if diff < threshold:
            shot_counter += 1
        previous_hist = hist

    cap.release()
    return shot_counter

def ocr_text_ratio(video_path, min_confidence=0.5):
    """
    Detects text presence in every nth (sample_rate) frame in the video using EasyOCR

    """

    sample_rate = 10 # skip some frames for faster calculation
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"The video cannot be opened from : {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return 0.0

    frames_with_texts, sampled_frames, frame_index = 0,0,0
    reader = easyocr.Reader(['en'])

    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_index % sample_rate == 0:
            sampled_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = reader.readtext(gray)
            for _,text,conf in results:
                if conf >= min_confidence and len(text.strip()) >= 3:   # check only words with min len of 5
                    frames_with_texts += 1
                    break

        frame_index += 1

    cap.release()
    return frames_with_texts/sampled_frames if sampled_frames > 0 else 0.0

def person_object_ratio(video_path,sample_rate=10):
    """
    Find person vs other objects ratio using YOLO 11n
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"The video cannot be opened from : {video_path}")

    person_frame, object_frame, frame_index = 0,0,0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_index % sample_rate == 0:
            res = model(frame, verbose=False)
            classes = res[0].boxes.cls.cpu().numpy()
            person = (classes == 0).any()
            others = (classes != 0).any() and (classes.size > 0)
            if person: person_frame += 1
            if others: object_frame += 1
        frame_index += 1
    cap.release()
    return person_frame/(object_frame + 1e-6)

def video_extractor(video_path):
    print("Extracting shot cuts .....")
    shot_cuts = detect_shot_cuts(video_path)
    print("___Number of hard cuts:",shot_cuts)

    print("Analyzing OCR .....")
    text_present_ratio = round(ocr_text_ratio(video_path,0.4),2)
    print("___Text_present_ratio:", text_present_ratio)

    print("Detecting objects with YOLO .....")
    person_vs_object_ratio = round(person_object_ratio(video_path,10), 2)
    print("___Person vs object ratio:",person_vs_object_ratio)

    features = {
        "shot_cuts": shot_cuts,
        "text_present_ratio": text_present_ratio,
        "person_vs_object_ratio": person_vs_object_ratio
    }
    return features

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python shot_cuts.py <path to video>")
        sys.exit(1)
    video_path = sys.argv[1]

    try:
        features = video_extractor(video_path)
        output_path = os.path.splitext(video_path)[0] + "_features.json"
        with open(output_path, 'w') as f:
            json.dump(features,f,indent=2)
    except Exception as e:
         print(f"error : {e}")
         sys.exit(1)

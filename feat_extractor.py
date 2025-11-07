import cv2
import sys
import easyocr

def detect_shot_cuts(video_path):
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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"The video cannot be opened from : {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return 0.0

    frames_with_texts = 0
    reader = easyocr.Reader(['en'])

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray)
        for _,text,conf in results:
            if conf >= min_confidence and len(text.strip()) >= 5:   # check only words with min len of 5
                frames_with_texts += 1
                break

    cap.release()
    return frames_with_texts/total_frames

def video_extractor(video_path):
    print("Extracting shot cuts .....")
    print(detect_shot_cuts(video_path))

    print("Analyzing OCR .....")
    print(ocr_text_ratio(video_path,0.5))

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python shot_cuts.py <path to video>")
        sys.exit(1)

    video_path = sys.argv[1]

    try:
        video_extractor(video_path)
    except Exception as e:
         print(f"error : {e}")
         sys.exit(1)

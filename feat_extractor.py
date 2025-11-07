import cv2
import sys
import easyocr
import pytesseract

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

def ocr_text_ratio(video_path):

    """ test pytesseract and easyocr performance """

    cap = cv2.VideoCapture(video_path)
    reader = easyocr.Reader(['en'])
    for i in range(50):
        ret, frame = cap.read()
        if ret:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # x, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(gray)
            text1 = reader.readtext(gray)
            print(f"text for {i} :", text)
            for (bbox, text2, confidence) in text1:
                print(f"Text eocr {i}: {text2}, Confidence: {confidence:.2f}")

            cv2.imshow(f"gray {i}", gray)
            cv2.waitKey(0)
            cv2.destroyWindow(f"gray {i}")
            # plt.plot(thresh)
            # cv2.imshow(f"thresh {i}",thresh)
            # cv2.waitKey(0)
            # cv2.destroyWindow(f"thresh {i}")
    cap.release()

def video_extractor(video_path):
    print("Extracting shot cuts .....")
    shot_cuts = detect_shot_cuts(video_path)
    return shot_cuts

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python shot_cuts.py <path to video>")
        sys.exit(1)

    video_path = sys.argv[1]

    try:
        features = video_extractor(video_path)
        print(features)
    except Exception as e:
         print(f"error : {e}")
         sys.exit(1)

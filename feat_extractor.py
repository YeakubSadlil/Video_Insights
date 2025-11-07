import cv2
import sys

def detect_shot_cuts(video_path):
    pass

def video_extractor(video_path):
    print("Extracting shot cuts")
    shot_cuts = detect_shot_cuts(video_path)

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python shot_cuts.py <path to video>")
        sys.exit(1)

    video_path = sys.argv[1]

    try:
        features = video_extractor(video_path)
    except Exception as e:
         print(f"error : {e}")
         sys.exit(1)

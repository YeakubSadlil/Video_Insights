
---

# Video Feature Extractor
The python script analyze a video and extract features including **hard shot cuts**, **text presence** and **person-object ratio**.

## Installation
```bash
git clone https://github.com/YeakubSadlil/Video_Insights.git

cd Video_Insights

pip install -r requirements.txt
```
## Usage
Provide your local video path
```python
python3 feat_extractor.py <path_to_video>
```
### Example:
```python
python3 feat_extractor.py video.mp4
```
It also saves a `JSON` file with extracted features

**Notes**
- Sampling is used to skip some frame to speed up precessing
- A hard cut is detected when histogram correlation coefficient difference < 0.7(threshold)
- Text Presence Ratio (by EasyOCR) = number of frames with text / total frames
- Only counted text with min_confidence and min length of 3
- Dominance ratio = number of frames with person / number of frames with other objects
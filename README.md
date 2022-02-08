# Tennis Tracker
---
## Example Output
![Alt text](images/out.jpg?raw=true "Example output")

---

## Installation
Install the requirements using the requirements or configure them yourself.

```bash
pip install -r requirements.txt
```

## How to use

```bash
python run.py -o output/out.mov -v videos/RollingTennisBall.mov
```

## TODO
- Test script that places a tennis ball template in random spots
- Finish C++ implementation
- Track Ball over multiple frames and show trajectory
- Work on more robust method than hsv thresholding/contour fitting 
- Create augmentations to image to test robustness
- Add argument to adjust size of frames to process on for speed up

import argparse
import cv2
from detect import BallDetect, RegionDetect, Writer, VideoProcess, Visualize

def run(opt):
    vp = VideoProcess(opt.video_path)
    vw = Writer(opt.output_path, width=vp.width, height=vp.height, fps = vp.fps)
    rt = RegionDetect()
    bt = BallDetect()
    visualizer = Visualize()
    i = 0
    while True: 
        ret, c_img, f_img = vp.getImage()
        if not ret:
            break 
        bt.detectBall(f_img)
        rt.outlineRegion(f_img)
        rt.ballRegion(bt)
        visualizer.outlineBallBB(c_img, bt)
        visualizer.outlineBall(c_img, bt)
        c_img = visualizer.outlineRegion(c_img, rt)
        vw.write(c_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_path", type=str, default="videos/RollingTennisBall.mov")
    parser.add_argument("-o", "--output_path", type=str, default="output/out.mov")
    opt = parser.parse_args() 
    run(opt)
    
from fastai.vision.all import *
import cv2
import numpy as np

BASE_DIR = Path.cwd()

video_file = BASE_DIR / 'videos/challenge_video.mp4'
vid_out = BASE_DIR / 'videos/challenge_video_result.mp4'

def get_overlayed_image(img, msk):
    return cv2.addWeighted(img,0.8,msk,0.2,0)

def get_y(fn):
    # return get_msk(fn, p2c)
    pass

def main():
    model = load_learner(BASE_DIR / 'road_segmentation/models/initial.pkl', cpu=False)
    cap = cv2.VideoCapture(str(video_file))
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    vid_writer = cv2.VideoWriter(str(vid_out), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 16,
                                          (int(w), int(h)))
    print(f"Shape of images ({h}, {w})")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = cv2.resize(frame, (256, 256))
                mask = model.predict(frame)
                mask = mask[2][1].detach().numpy()
                mask_colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
                mask_colored[mask > 0.5] = [0, 255, 0]
                mask_colored = cv2.resize(mask_colored, (int(w), int(h)))
                result = get_overlayed_image(frame, mask_colored)
                cv2.imshow("Result", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                vid_writer.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        vid_writer.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
            

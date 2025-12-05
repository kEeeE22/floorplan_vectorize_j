import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line

class cleanFloorplanExtractor:
    def __init__(self, target_width, target_height, tolerance=45):
        self.target_width = target_width
        self.target_height = target_height
        self.tolerance = tolerance

    def process_img(self, img):
        resized_segment = cv2.resize(
                    img,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_NEAREST
                )

        img_rgb = cv2.cvtColor(resized_segment, cv2.COLOR_BGR2RGB)

            # ===== MASK CỬA SỔ THEO MÀU CHÍNH XÁC =====
        target = np.array([255, 60, 128])
        wall_target = np.array([255,255,255])

        diff = np.abs(img_rgb - target)

        window_mask = np.where(np.max(diff, axis=2) < self.tolerance, 255, 0).astype(np.uint8)

        wall_diff = np.abs(img_rgb - wall_target)

        wall_mask = np.where(np.max(wall_diff, axis=2) < self.tolerance, 255, 0).astype(np.uint8)

        return window_mask, wall_mask
    
    def get_openings_skel(self, openings_mask, kernel=3, iters=1):
        kernel_ = np.ones((kernel,kernel), np.uint8)
        window_mask = cv2.dilate(openings_mask, kernel_, iterations=iters)

        skel = skeletonize(window_mask // 255).astype(np.uint8) * 255

        window_lines = probabilistic_hough_line(
            skel,
            threshold=5,
            line_length=5,
            line_gap=2
        )

        win_segments = []
        for (x1,y1), (x2,y2) in window_lines:
            win_segments.append([[x1,y1], [x2,y2]])
        # debug = openings_mask.copy()
        # debug = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)
        # for (x1, y1), (x2, y2) in window_lines:
        #     cv2.line(debug, (x1, y1), (x2, y2), (0,255,0), 2)
        # cv2_imshow(debug)
        # print(debug.shape)
        win1 = [np.array(s, dtype=np.int32) for s in win_segments]
        return win1
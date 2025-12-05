import cv2
import numpy as np
from skimage.morphology import skeletonize
from rdp import rdp
from skan import Skeleton

class rawFloorplanExtractor:
    def __init__(self, contour_threshold=60, rdp_epsilon=2.0, min_length=10):
        self.threshold = contour_threshold
        self.epsilon = rdp_epsilon
        self.min_length = min_length

    def get_raw_contours(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_img = np.ones_like(img) * 255
        for c in contours:
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            cv2.drawContours(contour_img, [c], -1, color, -1)

        return binary, contour_img

    def skeleton_to_vectors(self, binary):
        skel_img = skeletonize(binary > 0)
        skel = Skeleton(skel_img)

        vector_polylines = []
        for i in range(skel.n_paths):
            raw_coords = skel.path_coordinates(i)
            if len(raw_coords) < self.min_length:
                continue
            xy = np.column_stack((raw_coords[:, 1], raw_coords[:, 0]))
            simplified = rdp(xy, epsilon=self.epsilon)
            vector_polylines.append(simplified)

        return skel_img, vector_polylines
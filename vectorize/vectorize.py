import cv2

from vectorize.graph_process.build_graph import build_labeled_graph
from vectorize.graph_process.graph_optim import GraphOptimizer
from vectorize.raw_img_process import rawFloorplanExtractor
from vectorize.clean_img_process import cleanFloorplanExtractor
from .merge.wall_check import wall_line_check

def vectorize(raw_path, clean_path):

    #get wall from raw
    raw = cv2.imread(raw_path)
    print('Raw shape:')
    print(raw.shape)
    raw_extractor = rawFloorplanExtractor()
    # binary, contour_img = raw_extractor.get_raw_contours(raw)
    # skeleton_image, vectors = raw_extractor.skeleton_to_vectors(binary)

    #get window from clean
    clean = cv2.imread(clean_path)
    clean_extractor = cleanFloorplanExtractor(
        target_width=raw.shape[1],
        target_height=raw.shape[0],
        tolerance=45
    )
    window_mask, wall_mask = clean_extractor.process_img(clean)
    
    #vectors
    win = clean_extractor.get_openings_skel(window_mask, kernel=1, iters=1)

    skeleton_image, vectors = raw_extractor.skeleton_to_vectors(wall_mask)
    # filtered_vectors = wall_line_check(
    #     raw_vectors=vectors, clean_wall_mask=wall_mask, threshold=0.2,
    #     search_tolerance=10, num_samples=5)
    
    #graph
    G = build_labeled_graph(vectors, win)

    optimizer = GraphOptimizer(G)
    optimized_G = optimizer.optimize()

    return optimized_G
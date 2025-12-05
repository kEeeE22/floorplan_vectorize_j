import cv2
import numpy as np

def open_window_edges(
    img_rgb,
    window_color=(255, 60, 128),
    wall_color=(255, 255, 255),
    erode_kernel=3,
    dilate_kernel=3,
    erode_iter=1,
    dilate_iter=1
):
    """
    Làm sạch vùng cửa sổ (màu hồng) bị dính vào tường trắng bằng erosion-dilation.

    Args:
        img_rgb (np.ndarray): Ảnh RGB đầu vào.
        window_color (tuple): Màu vùng cửa sổ (default = (255,60,128)).
        wall_color (tuple): Màu tường (default = trắng).
        erode_kernel (int): Kích thước kernel khi co vùng (xóa viền hồng nhỏ).
        dilate_kernel (int): Kích thước kernel khi giãn vùng (phục hồi vùng chính).
        erode_iter (int): Số lần erosion.
        dilate_iter (int): Số lần dilation.
    Returns:
        np.ndarray: Ảnh RGB sau khi làm sạch viền cửa sổ.
    """

    # --- B1. Tạo mask vùng cửa sổ ---
    mask_window = cv2.inRange(img_rgb, np.array(window_color), np.array(window_color))

    # --- B2. Co vùng (erosion) để loại bỏ pixel thừa ---
    kernel_erode = np.ones((erode_kernel, erode_kernel), np.uint8)
    mask_eroded = cv2.erode(mask_window, kernel_erode, iterations=erode_iter)

    # --- B3. Giãn nhẹ vùng trở lại (dilation) ---
    kernel_dilate = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    mask_clean = cv2.dilate(mask_eroded, kernel_dilate, iterations=dilate_iter)

    # --- B4. Tô lại những pixel hồng bị loại bỏ bằng màu tường ---
    diff = cv2.subtract(mask_window, mask_clean)
    clean_img = img_rgb.copy()
    clean_img[diff > 0] = wall_color

    return clean_img

def close_window_edges(
    img_rgb,
    window_color=(255, 60, 128),
    wall_color=(255, 255, 255),
    dilate_kernel=3,
    erode_kernel=3,
    dilate_iter=1,
    erode_iter=1
):
    """
    Làm sạch vùng cửa sổ (màu hồng) bị dính vào tường trắng
    bằng phép morphological closing (giãn trước, co sau).

    Args:
        img_rgb (np.ndarray): Ảnh RGB đầu vào.
        window_color (tuple): Màu vùng cửa sổ (default = (255,60,128)).
        wall_color (tuple): Màu tường (default = trắng).
        dilate_kernel (int): Kích thước kernel khi giãn vùng.
        erode_kernel (int): Kích thước kernel khi co vùng.
        dilate_iter (int): Số lần dilation.
        erode_iter (int): Số lần erosion.
    Returns:
        np.ndarray: Ảnh RGB sau khi làm sạch viền cửa sổ.
    """

    # --- B1. Tạo mask vùng cửa sổ ---
    mask_window = cv2.inRange(img_rgb, np.array(window_color), np.array(window_color))

    # --- B2. Giãn vùng trước (dilation) để lấp khe nhỏ ---
    kernel_dilate = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    mask_dilated = cv2.dilate(mask_window, kernel_dilate, iterations=dilate_iter)

    # --- B3. Co vùng lại (erosion) để khôi phục kích thước ---
    kernel_erode = np.ones((erode_kernel, erode_kernel), np.uint8)
    mask_clean = cv2.erode(mask_dilated, kernel_erode, iterations=erode_iter)

    # --- B4. Tô lại pixel hồng bị loại bỏ bằng màu tường ---
    diff = cv2.subtract(mask_window, mask_clean)
    clean_img = img_rgb.copy()
    clean_img[diff > 0] = wall_color

    return clean_img

def clean_opening_to_wall(img_rgb, pink_color=(255,60,128), wall_color=(255,255,255), min_area=200):
    mask = cv2.inRange(img_rgb, np.array(pink_color), np.array(pink_color))
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    clean_img = img_rgb.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)
        if cv2.contourArea(cnt) < min_area:
            cv2.drawContours(clean_img, [cnt], -1, wall_color, -1)
    black_mask = cv2.inRange(clean_img, np.array([0,0,0]), np.array([60,60,60]))
    contours_black, _ = cv2.findContours(black_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_black:
        area = cv2.contourArea(cnt)
        if cv2.contourArea(cnt) < 200:

            cv2.drawContours(clean_img, [cnt], -1, wall_color, -1)
    return clean_img
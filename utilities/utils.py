import cv2

from .log import logger

def create_screen(screen_name, width, height):
	cv2.namedWindow(screen_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(screen_name, width	, height)

def read_image(image_path):
	return cv2.imread(image_path, cv2.IMREAD_COLOR)

def remove_noise(contours):
	biggest_contour = max(contours, key=cv2.contourArea)
	biggest_area = cv2.contourArea(biggest_contour)
	return [cnt for cnt in contours if cv2.contourArea(cnt) / biggest_area > 0.1]

if __name__ == '__main__':
    logger.info('Testing utils...')
import click

from predictor import *
from utilities.utils import *
    
@click.command()
@click.option('--video_path', help='Path to video', default='data/public_walk.mp4')
@click.option('--method', help='The detection method', type=click.Choice(['opencv', 'yolo']), default='opencv')
def process(video_path, method):
    capture = cv2.VideoCapture(video_path)
    
    if method == 'opencv':
        predict_with_knn(capture)
    elif method == 'yolo':
        predict_with_yolo(capture)
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    logger.info('... [ Processing ] ...')
    process()
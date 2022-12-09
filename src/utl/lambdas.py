from const.path import CTX
import cv2 as cv
from PIL import Image
# lambda_list = [lambda x: x+3, lambda x: x*5]
# 리스트 활용 시 하나의 람다 함수 = 하나의 인덱스
# print(lambda_list[0](5))
# print(lambda_list[1](5))
def MosaicLambda(*params):
    cmd = params[0]
    target = params[1]
    if cmd == 'IMAGE_READ_FOR_CV':
        return (lambda x: cv.imread(CTX+x))(target)
    elif cmd == 'IMAGE_READ_FOR_PLT':
        return cv.cvtColor((lambda x: cv.imread(CTX+x))(target), cv.COLOR_BGR2RGB)
    elif cmd == 'GRAY_SCALE': # GRAYSCALE 변환 공식
        return (lambda x: x[:, :, 0] * 0.114 + x[:, :, 1] * 0.587 + x[:, :, 2] * 0.229)(target)
    elif cmd == 'IMAGE_FROM_ARRAY':
        return (lambda x: Image.fromarray(x))(target)
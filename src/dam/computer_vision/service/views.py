from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import copy

from src.utl.lambdas import MosaicLambda
from mosaic.services import ImageToNumberArray, \
    Canny, Hough, Haar, mosaic, mosaics
from const.path import CTX

class MenuController(object):

    @staticmethod
    def menu_0(*params):
         print(params[0])

    @staticmethod
    def menu_1(*params): # 원본
        print(params[0])
        img =  MosaicLambda('IMAGE_READ_FOR_CV', params[1])
        print(f'cv2 버전 {cv.__version__}')  # cv2 버전 4.6.0
        print(f' Shape is {img.shape}')
        cv.imshow('Original', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    @staticmethod
    def menu_2(*params): # 그레이
        print(params[0])
        arr = ImageToNumberArray(params[1])
        img = MosaicLambda('GRAY_SCALE', arr)
        plt.imshow(MosaicLambda('IMAGE_FROM_ARRAY', img))
        plt.show()

    @staticmethod
    def menu_3(*params): # 엣지
        print(params[0])
        ### 디스크에서 읽는 경우 ###
        # img = cv.imread('./data/roi.jpg', 0)
        # img = cv.imread(img, 0)
        ### 메모리에서 읽는 경우 ###
        img = ImageToNumberArray(params[1])
        print(f'img type : {type(img)}')
        # img = GaussianBlur(img, 1, 1) cv.Canny() 를 사용하지 않는 경우 필요
        # img = Canny(img, 50, 150) cv.Canny() 를 사용하지 않는 경우 필요
        edges = cv.Canny(np.array(img), 100, 200)
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    @staticmethod
    def menu_4(*params): # 직선
        print(params[0])
        img = ImageToNumberArray(params[1])
        edges = cv.Canny(img, 100, 200) # (image, threshold 1=100, threshold 2=200)
        dst = Hough(edges)
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(dst, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    @staticmethod
    def menu_5(*params): # 모자이크
        print(params[0])
        cat = cv.imread(CTX + params[1])
        mos = mosaic(cat, (10, 120, 230, 350), 10)
        # (20, 10, 200, 200) : (모자이크 시작위치(좌상단 0,0), 모자이크 크기)
        # cv.imwrite(CTX + cat - computer_vision.png, mos)

        cv.imshow('CAT MOSAIC', mos)
        cv.waitKey(0)
        cv.destroyAllWindows()

    @staticmethod
    def menu_6(*params):
        print(params[0])
        param = params[1]  # girl.jpg
        girl_original = MosaicLambda('IMAGE_READ_FOR_PLT', param)
        girl_gray = MosaicLambda('GRAY_SCALE', girl_original)
        girl_canny = Canny(girl_original)
        girl_hough = Hough(girl_canny)
        girl_clone = copy.deepcopy(girl_original)
        rect = Haar(girl_clone)
        girl_mosaic = mosaic(girl_original, rect, 10)

        plt.subplot(161), plt.imshow(girl_original, cmap='gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(162), plt.imshow(girl_gray, cmap='gray')
        plt.title('Gray'), plt.xticks([]), plt.yticks([])
        plt.subplot(163), plt.imshow(girl_canny, cmap='gray')
        plt.title('Edge'), plt.xticks([]), plt.yticks([])
        plt.subplot(164), plt.imshow(girl_hough, cmap='gray')
        plt.title('Hough'), plt.xticks([]), plt.yticks([])
        plt.subplot(165), plt.imshow(girl_clone, cmap='gray')
        plt.title('HAAR'), plt.xticks([]), plt.yticks([])
        plt.subplot(166), plt.imshow(girl_mosaic, cmap='gray')
        plt.title('MOSAIC'), plt.xticks([]), plt.yticks([])
        plt.show()
        pass

    @staticmethod
    def menu_7(*params): # 모녀 모자이크
        print(params[0])
        girl_with_mom = MosaicLambda('IMAGE_READ_FOR_CV', params[1])
        girl_with_mom = cv.cvtColor(girl_with_mom, cv.COLOR_BGR2RGB)
        mos = mosaics(girl_with_mom, 10)
        plt.subplot(121), plt.imshow(girl_with_mom, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(mos, cmap='gray')
        plt.title('Mosaic Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    @staticmethod
    def menu_8(*params): # 선택 모자이크
        print(params[0])
        img = MosaicLambda('IMAGE_READ_FOR_CV', params[1])
        # 선택영역 x,y값
        x_pos,y_pos,width,height = cv.selectROI("location", img, False)
        print("x position, y position :",x_pos, y_pos)
        print("width, height : ",width, height)
        mosaic_loc = img[y_pos:y_pos+height, x_pos:x_pos+width]
        mosaic_loc = cv.blur(mosaic_loc,(50,50))
        mosaic = img
        mosaic[y_pos:y_pos+height, x_pos:x_pos+width] = mosaic_loc
        cv.destroyAllWindows()
        cv.imshow("complete", mosaic)
        cv.waitKey(0)
        cv.destroyAllWindows()

    @staticmethod
    def menu_9(*params):
        print(params[0])
        img = MosaicLambda('IMAGE_READ_FOR_CV', params[1])
        # 비율로 이미지 크기 조정
        img2 = cv.resize(img, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
        img_cvt = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        cv.imshow("img_cvt", img_cvt)
        cv.waitKey(0)
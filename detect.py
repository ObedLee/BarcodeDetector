from __future__ import print_function
import argparse
import numpy as np
import cv2
import os
import glob
import time


def detectBarcode(img, fname):
    # 개별 작성
    cv2.imshow(fname, img)

    # 소벨 연산자를 사용하여 수평 및 수직 방향으로 경계 강도를 계산
    sharpx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
    sharpx = cv2.convertScaleAbs(sharpx)
    sharpy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)
    sharpy = cv2.convertScaleAbs(sharpy)

    # 수평 방향의 경계 강도 영상에 수직 방향의 경계 강도 영상을 빼 후보 영역을 검출
    dstx = cv2.subtract(sharpx, sharpy)
    # 가우시안 블러 적용
    dstx = cv2.GaussianBlur(dstx, (15, 13), 0)
    # 임계화 수행
    th, dstx = cv2.threshold(dstx, 100, 200, cv2.THRESH_BINARY)

    # 모폴로지 변환(닫힘 연산) 적용, 직사각형
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (57, 13))
    dstx = cv2.morphologyEx(dstx, cv2.MORPH_CLOSE, kernel)
    # 3번 반복하여 침식과 팽창 연산을 적용, 정사각형
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    dstx = cv2.erode(dstx, kernel, iterations=3)
    dstx = cv2.dilate(dstx, kernel, iterations=3)

    # 연결요소를 찾아 가장 큰 연결 요소만
    (contours, hierarchy) = cv2.findContours(dstx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        contour_x = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    else:
        contour_x = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # 수직 방향 동일 진행
    dsty = cv2.subtract(sharpy, sharpx)
    dsty = cv2.GaussianBlur(dsty, (13, 15), 0)
    th, dsty = cv2.threshold(dsty, 100, 200, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,57))
    dsty = cv2.morphologyEx(dsty, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,13))
    dsty = cv2.erode(dsty, kernel, iterations=3)
    dsty = cv2.dilate(dsty, kernel, iterations=3)


    (contours, hierarchy) = cv2.findContours(dsty, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        contour_y = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    else:
        contour_y = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # 수평 방향과 수직 방향을 비교하여 연결 요소의 크기가 더 큰 방향의 값을 최종 바코드 영역으로 판단
    if len(contour_x) > len(contour_y):
        rect = cv2.minAreaRect(contour_x)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # ref.data의 값고 비교 시 좌표 순서 안 맞는게 있어서 코드 추가
        if box[2][1] < box[0][1]:
            temp = box[2][1]
            box[2][1] = box[0][1]
            box[0][1] = temp
        if box[2][0] < box[0][0]:
            temp = box[2][0]
            box[2][0] = box[0][0]
            box[0][0] = temp
    else:
        rect = cv2.minAreaRect(contour_y)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if box[2][1] < box[0][1]:
            temp = box[2][1]
            box[2][1] = box[0][1]
            box[0][1] = temp
        if box[2][0] < box[0][0]:
            temp = box[2][0]
            box[2][0] = box[0][0]
            box[0][0] = temp

    points = np.concatenate([box[0], box[2]])

    return points


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required = True, help = "path to the dataset folder")
    ap.add_argument("-r", "--detectset", required = True, help = "path to the detectset folder")
    ap.add_argument("-f", "--detect", required = True, help = "path to the detect file")
    args = vars(ap.parse_args())
    
    dataset = args["dataset"]
    detectset = args["detectset"]
    detectfile = args["detect"]

    # 결과 영상 저장 폴더 존재 여부 확인
    if not os.path.isdir(detectset):
        os.mkdir(detectset)

    # 결과 영상 표시 여부
    verbose = False

    # 검출 결과 위치 저장을 위한 파일 생성
    f = open(detectfile, "wt", encoding="UTF-8")  # UT-8로 인코딩

    # 바코드 영상에 대한 바코드 영역 검출
    filelist = glob.glob(dataset + "/*.jpg") + glob.glob(dataset + "/*.JPG") + glob.glob(dataset + "/*.jpeg")+glob.glob(dataset + "/*.JPEG")

    start = time.time()  # 시작 시간 저장
    for imagePath in filelist:
        print(imagePath, '처리중...')

        # 영상을 불러오고 그레이 스케일 영상으로 변환
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 바코드 검출
        points = detectBarcode(gray,imagePath)

        # 바코드 영역 표시
        detectimg = cv2.rectangle(image, (points[0], points[1]), (points[2], points[3]), (0, 255, 0), 2)  # 이미지에 사각형 그리기

        # 결과 영상 저장
        loc1 = imagePath.rfind("/")
        loc2 = imagePath.rfind(".")
        fname = 'result/' + imagePath[loc1 + 1: loc2] + '_res.jpg'
        cv2.imwrite(fname, detectimg)

        # 검출한 결과 위치 저장
        f.write(imagePath[loc1 + 1: loc2])
        f.write("\t")
        f.write(str(points[0]))
        f.write("\t")
        f.write(str(points[1]))
        f.write("\t")
        f.write(str(points[2]))
        f.write("\t")
        f.write(str(points[3]))
        f.write("\n")

        if verbose:
            cv2.imshow("image", image)
            cv2.waitKey(0)

    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
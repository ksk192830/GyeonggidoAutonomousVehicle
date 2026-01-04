import cv2
import numpy as np

# 절대경로 이미지
IMG_PATH = "/home/sg/caps_ws/convenient/img/image102.jpg"

image = cv2.imread(IMG_PATH)
if image is None:
    print(f"Failed to load image: {IMG_PATH}")
    exit(1)

# 화면에 그릴 이미지 (원본 보존용)
display_img = image.copy()

points = []   # [(x1, y1), (x2, y1)]
stage = 0     # 0: 첫 점 대기, 1: 첫 점 선택 후 Enter 대기,
              # 2: 두 번째 점 대기, 3: 측정 완료


def mouse_callback(event, x, y, flags, param):
    global points, stage, display_img

    if event == cv2.EVENT_LBUTTONDOWN:
        # 1) 첫 번째 점: x,y 자유
        if stage == 0:
            points = [(x, y)]
            stage = 1

            # 화면 초기화 후 빨간 가로선 + 점 그리기
            display_img = image.copy()
            h, w = display_img.shape[:2]
            y1 = y
            cv2.line(display_img, (0, y1), (w - 1, y1), (0, 0, 255), 2)
            cv2.circle(display_img, (x, y1), 4, (0, 0, 255), -1)

            print(f"첫 번째 점: ({x}, {y})")
            print("확정 할 거면 Enter 입력")

        # 2) 두 번째 점: y는 고정, x만 자유
        elif stage == 2 and points:
            x1, y1 = points[0]
            x2 = x
            y2 = y1  # y 고정

            points.append((x2, y2))
            stage = 3

            # 두 번째 점 그리기
            display_img = image.copy()
            h, w = display_img.shape[:2]
            cv2.line(display_img, (0, y1), (w - 1, y1), (0, 0, 255), 2)
            cv2.circle(display_img, (x1, y1), 4, (0, 255, 0), -1)
            cv2.circle(display_img, (x2, y2), 4, (255, 0, 0), -1)

            pixel_count = abs(x2 - x1)
            print(f"두 번째 점 (y locked): ({x2}, {y2})")
            print(f"Pixel count between points: {pixel_count}")
            print(f"다시 할 거면 Enter")


cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_callback)

print("Left-click to select the first point.")
print("After first point, press Enter to move to second point.")
print("Press ESC to exit.")

while True:
    cv2.imshow("image", display_img)
    key = cv2.waitKey(20) & 0xFF

    # ESC → 종료
    if key == 27:
        break

    # Enter
    if key in (13, 10):
        # 첫 점 선택 후 → 두 번째 점 선택 모드로
        if stage == 1:
            stage = 2
            print("Now left-click to select the second point (y is fixed).")

        # 측정 끝난 후 → 초기화해서 다시 측정
        elif stage == 3:
            points = []
            stage = 0
            display_img = image.copy()
            print("Reset. Left-click to select a new first point.")

cv2.destroyAllWindows()


points = []  # will store [(x1, y1), (x2, y1)]
stage = 0    # 0: waiting first point, 1: first selected, waiting Enter,
             # 2: waiting second point, 3: measurement done


def mouse_callback(event, x, y, flags, param):
    global points, stage

    if event == cv2.EVENT_LBUTTONDOWN:
        # first point (free x, y)
        if stage == 0:
            points = [(x, y)]

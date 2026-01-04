import cv2

# 여기에 절대경로 직접 넣어두기
IMAGE_PATH = "/home/sg/caps_ws/convenient/img/image102.jpg"

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img = param
        h, w = img.shape[:2]

        rel_x = x / w
        rel_y = y / h

        print(f"Absolute: ({x}, {y})  |  Relative: ({rel_x:.4f}, {rel_y:.4f})")

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("이미지를 불러올 수 없습니다. IMAGE_PATH를 확인하세요.")
        return

    window_name = "Image"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse, img)

    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC 키
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

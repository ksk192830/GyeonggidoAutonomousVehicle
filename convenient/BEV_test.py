import cv2
import numpy as np
import os

def bev_transform(image):
    h, w = image.shape[:2]

    persp_src = [0.25, 1.0, 0.75, 1.0, 0.6, 0.5, 0.4, 0.5]
    persp_dst = [0.25, 1.0, 0.75, 1.0, 0.75, 0.0, 0.25, 0.0]

    src_pts = np.float32([
        [persp_src[0] * w, persp_src[1] * h],
        [persp_src[2] * w, persp_src[3] * h],
        [persp_src[4] * w, persp_src[5] * h],
        [persp_src[6] * w, persp_src[7] * h],
    ])
    dst_pts = np.float32([
        [persp_dst[0] * w, persp_dst[1] * h],
        [persp_dst[2] * w, persp_dst[3] * h],
        [persp_dst[4] * w, persp_dst[5] * h],
        [persp_dst[6] * w, persp_dst[7] * h],
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    bev = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)
    return bev


def main():

    # 원본 이미지 절대경로
    img_path = "/home/sg/caps_ws/convenient/img/image102.jpg"

    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ 이미지 로드 실패: {img_path}")
        return

    bev = bev_transform(image)

    # ---- BEV 결과 저장 경로 설정 ----
    base, ext = os.path.splitext(img_path)
    save_path = f"{base}_BEV{ext}"

    cv2.imwrite(save_path, bev)
    print(f"✅ BEV 이미지 저장됨: {save_path}")

    # 화면 표시
    cv2.imshow("Original", image)
    cv2.imshow("BEV", bev)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Erosion — 강의 개념에 맞춘 독립 실행 예제.
설치: python -m pip install numpy matplotlib
실행: python 02_erosion.py
강의자료의 이미지를 복사하지 않고 새 원본 이미지를 생성합니다.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 영상은 0=검은 배경, 1=흰 물체입니다. 출력 영상의 해상도는 유지됩니다.
# SE의 1은 검사할 위치, 0은 무시할 위치입니다.
# 0인 위치에서 검은 배경을 요구하는 것이 아닙니다.
# 중심 대칭이며 가로/세로 크기가 홀수인 SE를 사용합니다.

def erosion(image, se):
    # 침식: SE의 모든 활성 위치가 물체 내부에 들어가는 Fit일 때만 1입니다.
    # 일부만 겹치는 경우와 전혀 겹치지 않는 경우는 모두 0으로 남습니다.
    height, width = image.shape
    kh, kw = se.shape
    pad_y, pad_x = kh // 2, kw // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)),
                    mode="constant", constant_values=0)
    result = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            region = padded[y:y + kh, x:x + kw]
            pixels = region[se == 1]
            # SE가 0인 곳은 검사하지 않습니다. 전체 배열의 일치 검사가 아닙니다.
            if np.all(pixels == 1):   # 검사 위치가 모두 1일 때 참: 논리 AND
                result[y, x] = 1

    # 영상 밖을 0으로 보므로 화면 경계에 붙은 물체도 침식됩니다.
    return result

def make_image():
    # 큰 물체 두 개를 폭 3픽셀의 가는 연결부로 이어 줍니다.
    image = np.zeros((220, 300), dtype=np.uint8)
    image[60:160, 40:115] = 1
    image[60:160, 175:250] = 1
    image[108:111, 115:175] = 1
    # 작은 흰 점은 5x5 구조 요소가 들어갈 수 없으므로 침식으로 사라집니다.
    for y, x in [(30, 35), (180, 145), (35, 220), (185, 260)]:
        image[y:y + 3, x:x + 3] = 1
    return image

def show_image(ax, image, title):
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def show_se(ax, se, title):
    # 구조 요소를 확대 표시합니다. 영상 패널과 표시 배율은 다릅니다.
    ax.imshow(se, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    h, w = se.shape
    for y in range(h):
        for x in range(w):
            ax.text(x, y, str(se[y, x]), ha="center", va="center",
                    color="black" if se[y, x] else "white", fontsize=9)
    # 빨간 테두리는 SE의 중심(origin)입니다. 여기를 검사할 픽셀에 맞춥니다.
    ax.add_patch(plt.Rectangle((w//2-0.5, h//2-0.5), 1, 1,
                              fill=False, edgecolor="red", linewidth=2))
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.6)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    ax.set_title(title + "\nRed = origin (enlarged view)")


def main():
    original = make_image()
    output = Path(__file__).resolve().parent / "results" / "erosion"
    output.mkdir(parents=True, exist_ok=True)
    plt.imsave(output / "original.png", original, cmap="gray", vmin=0, vmax=1)
    se = np.ones((5, 5), dtype=np.uint8)  # 강의의 정사각형 침식 예에 대응
    result = erosion(original, se)
    # 원본과 결과는 나란히 놓고, 오른쪽에 실제 사용한 SE를 확대 표시합니다.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [3, 3, 1.7]})
    show_image(axes[0], original, "Original")
    show_image(axes[1], result, "Erosion")
    show_se(axes[2], se, "Structuring element")
    plt.imsave(output / "result.png", result, cmap="gray", vmin=0, vmax=1)
    print("흰 픽셀:", int(original.sum()), "->", int(result.sum()))
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    print("정사각형 SE가 완전히 들어가는 Fit 위치만 남습니다. 물체가 줄고 가는 연결부와 작은 흰 점이 사라집니다.")
    print("저장 위치:", output)
    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Dilation — 강의 개념에 맞춘 독립 실행 예제.
설치: python -m pip install numpy matplotlib
실행: python 01_dilation.py
강의자료의 이미지를 복사하지 않고 새 원본 이미지를 생성합니다.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 영상은 0=검은 배경, 1=흰 물체입니다. 출력 영상의 해상도는 유지됩니다.
# SE의 1은 검사할 위치, 0은 무시할 위치입니다.
# 0인 위치에서 검은 배경을 요구하는 것이 아닙니다.
# 중심 대칭이며 가로/세로 크기가 홀수인 SE를 사용합니다.

def dilation(image, se):
    # 팽창: 구조 요소가 물체에 하나라도 닿는 Hit이면 중심 위치를 1로 설정합니다.
    # Fit(전부 겹침)도 Hit 조건을 만족합니다. Miss(전혀 겹치지 않음)는 0입니다.
    # 대칭 SE에서는 반사해도 모양이 같으므로 아래 이웃 검사로 구현할 수 있습니다.
    if not np.array_equal(se, se[::-1, ::-1]):
        raise ValueError("이 실습 함수는 중심 대칭 구조 요소를 사용합니다.")
    height, width = image.shape
    kh, kw = se.shape
    pad_y, pad_x = kh // 2, kw // 2

    # 가장자리에서도 동일한 크기의 이웃을 검사하도록 검은색(0) 패딩을 추가합니다.
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)),
                    mode="constant", constant_values=0)
    result = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            # 패딩 때문에 이 작은 영역의 중심은 원본의 (y, x)에 대응합니다.
            region = padded[y:y + kh, x:x + kw]
            pixels = region[se == 1]  # SE가 1인 위치만 골라 검사합니다.
            if np.any(pixels == 1):   # 하나라도 1이면 참: 논리 OR
                result[y, x] = 1

    # 원본 크기의 배열에만 기록했으므로 패딩을 별도로 잘라낼 필요가 없습니다.
    return result

def make_image():
    # 가는 H자 획, 끊어진 부분, 가까이 있는 두 물체를 만듭니다.
    image = np.zeros((220, 300), dtype=np.uint8)
    image[45:170, 48:51] = 1
    image[45:170, 115:118] = 1
    image[105:108, 48:118] = 1
    image[83:86, 48:51] = 0       # 3픽셀 단절: 팽창 후 연결되는지 관찰
    image[70:140, 185:215] = 1
    image[70:140, 218:248] = 1    # 두 사각형 사이의 간격은 3픽셀
    image[185:187, 150:152] = 1   # 흰 점 잡음도 팽창으로 커진다는 비교 대상
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
    output = Path(__file__).resolve().parent / "results" / "dilation"
    output.mkdir(parents=True, exist_ok=True)
    plt.imsave(output / "original.png", original, cmap="gray", vmin=0, vmax=1)
    # 같은 원본에 네 가지 SE를 사용해야 '모양'에 따른 효과를 비교할 수 있습니다.
    cross = np.zeros((5, 5), dtype=np.uint8)
    cross[2, :] = 1
    cross[:, 2] = 1
    elements = {"Vertical": np.ones((5, 1), dtype=np.uint8),
                "Horizontal": np.ones((1, 5), dtype=np.uint8),
                "Square": np.ones((5, 5), dtype=np.uint8),
                "Cross": cross}
    fig, axes = plt.subplots(4, 3, figsize=(12, 11),
                             gridspec_kw={"width_ratios": [3, 3, 1.7]})
    for row, (label, se) in enumerate(elements.items()):
        result = dilation(original, se)
        show_image(axes[row, 0], original, "Original")
        show_image(axes[row, 1], result, "Dilation: " + label)
        show_se(axes[row, 2], se, label + " SE")
        plt.imsave(output / (label.lower() + "_result.png"), result,
                   cmap="gray", vmin=0, vmax=1)
        print(label, "흰 픽셀:", int(original.sum()), "->", int(result.sum()))
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    print("수직 SE는 수직 획의 단절, 수평 SE는 사각형 사이의 수평 간격을 연결합니다. 정사각형과 십자형의 확장 모양도 비교하세요.")
    print("저장 위치:", output)
    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()

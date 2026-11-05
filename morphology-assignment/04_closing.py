# -*- coding: utf-8 -*-
"""Closing — 강의 개념에 맞춘 독립 실행 예제.
설치: python -m pip install numpy matplotlib
실행: python 04_closing.py
강의자료의 이미지를 복사하지 않고 새 원본 이미지를 생성합니다.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 영상은 0=검은 배경, 1=흰 물체입니다. 출력 영상의 해상도는 유지됩니다.
# SE의 1은 검사할 위치, 0은 무시할 위치입니다.
# 0인 위치에서 검은 배경을 요구하는 것이 아닙니다.
# 중심 대칭이며 가로/세로 크기가 홀수인 SE를 사용합니다.

def disk_se(radius):
    # 반지름 r의 원판: 중심에서 거리가 r 이하인 칸만 1입니다.
    # 바깥 사각형의 크기는 (2r+1)x(2r+1)이지만 모서리는 검사하지 않습니다.
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x*x + y*y <= radius*radius).astype(np.uint8)


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

def closing(image, se):
    # 같은 SE로 팽창한 다음 침식합니다.
    # 중간 팽창이 화면 밖에서 잘리지 않도록 합성 연산 전체에 여백을 둡니다.
    # 이 예제의 물체는 경계에서 떨어져 있지만, 일반적인 경계 처리도 명시합니다.
    py, px = se.shape[0] // 2, se.shape[1] // 2
    height, width = image.shape
    extended = np.pad(image, ((py, py), (px, px)), constant_values=0)
    intermediate_extended = dilation(extended, se)
    result_extended = erosion(intermediate_extended, se)
    # 팽창과 침식을 모두 끝낸 다음 원래 영상 크기로 자릅니다.
    result = result_extended[py:py + height, px:px + width]
    intermediate = intermediate_extended[py:py + height, px:px + width]
    return result, intermediate

def make_image():
    # 흰 물체 내부의 크기가 다른 검은 원과 좁은 균열을 새로 만듭니다.
    image = np.zeros((220, 300), dtype=np.uint8)
    image[30:190, 30:270] = 1
    y, x = np.ogrid[:220, :300]
    # 원판형 SE(반지름 4)보다 작은 구멍과 큰 구멍을 함께 둡니다.
    for cy, cx, r in [(65,65,2), (105,65,3), (155,85,3),
                       (65,145,3), (155,160,2), (110,220,18)]:
        image[(y-cy)**2 + (x-cx)**2 <= r*r] = 0
    image[30:190, 115:118] = 0  # 폭 3픽셀의 균열
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
    output = Path(__file__).resolve().parent / "results" / "closing"
    output.mkdir(parents=True, exist_ok=True)
    plt.imsave(output / "original.png", original, cmap="gray", vmin=0, vmax=1)
    se = disk_se(radius=4)  # 반지름 4픽셀, 9x9 배열의 원판형 SE
    result, intermediate = closing(original, se)
    # 원본과 결과는 나란히 놓고, 오른쪽에 실제 사용한 SE를 확대 표시합니다.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [3, 3, 1.7]})
    show_image(axes[0], original, "Original")
    show_image(axes[1], result, "Closing")
    show_se(axes[2], se, "Structuring element")
    plt.imsave(output / "result.png", result, cmap="gray", vmin=0, vmax=1)
    print("흰 픽셀:", int(original.sum()), "->", int(result.sum()))
    # 두 단계의 순서와 중간 결과를 별도로 보여줍니다. SE는 두 단계에서 같습니다.
    process_fig, process_axes = plt.subplots(1, 3, figsize=(12, 4))
    show_image(process_axes[0], original, "Original")
    show_image(process_axes[1], intermediate, "After Dilation")
    show_image(process_axes[2], result, "Closing")
    process_fig.tight_layout()
    process_fig.savefig(output / "process.png", dpi=180)
    plt.imsave(output / "intermediate.png", intermediate, cmap="gray", vmin=0, vmax=1)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    print("같은 원판형 SE로 팽창 후 침식합니다. 작은 검은 구멍과 균열은 채워지고 큰 구멍은 남습니다. 바깥 모서리도 일부 변할 수 있습니다.")
    print("저장 위치:", output)
    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()

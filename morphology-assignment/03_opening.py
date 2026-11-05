# -*- coding: utf-8 -*-
"""Opening — 강의 개념에 맞춘 독립 실행 예제.
설치: python -m pip install numpy matplotlib
실행: python 03_opening.py
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

def opening(image, se):
    # 같은 SE로 침식한 다음 팽창합니다. 두 연산의 순서를 바꾸면 닫힘입니다.
    intermediate = erosion(image, se)
    result = dilation(intermediate, se)
    # 중간 결과도 반환하여 큰 물체가 작아졌다가 다시 커지는 과정을 확인합니다.
    return result, intermediate

def make_image():
    # 강의의 '원과 선 중에서 원을 남기는 예'에 대응하는 새로운 합성 영상입니다.
    # 원판형 SE가 들어가는 큰 원은 남고, 폭 3픽셀 선은 침식에서 사라집니다.
    image = np.zeros((220, 300), dtype=np.uint8)
    y, x = np.ogrid[:220, :300]
    for cy, cx, r in [(80, 75, 34), (140, 220, 43)]:
        image[(y-cy)**2 + (x-cx)**2 <= r*r] = 1
    image[35:185, 143:146] = 1  # 가는 수직선
    image[185:188, 25:130] = 1  # 가는 수평선
    image[79:82, 75:145] = 1   # 원에 붙은 가느다란 돌출부
    rng = np.random.default_rng(42)
    for _ in range(45):
        cy = int(rng.choice([15, 202]))
        cx = int(rng.integers(12, 285))
        image[cy:cy+2, cx:cx+2] = 1  # 작은 흰 점
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
    output = Path(__file__).resolve().parent / "results" / "opening"
    output.mkdir(parents=True, exist_ok=True)
    plt.imsave(output / "original.png", original, cmap="gray", vmin=0, vmax=1)
    se = disk_se(radius=4)  # 반지름 4픽셀, 9x9 배열의 원판형 SE
    result, intermediate = opening(original, se)
    # 원본과 결과는 나란히 놓고, 오른쪽에 실제 사용한 SE를 확대 표시합니다.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [3, 3, 1.7]})
    show_image(axes[0], original, "Original")
    show_image(axes[1], result, "Opening")
    show_se(axes[2], se, "Structuring element")
    plt.imsave(output / "result.png", result, cmap="gray", vmin=0, vmax=1)
    print("흰 픽셀:", int(original.sum()), "->", int(result.sum()))
    # 두 단계의 순서와 중간 결과를 별도로 보여줍니다. SE는 두 단계에서 같습니다.
    process_fig, process_axes = plt.subplots(1, 3, figsize=(12, 4))
    show_image(process_axes[0], original, "Original")
    show_image(process_axes[1], intermediate, "After Erosion")
    show_image(process_axes[2], result, "Opening")
    process_fig.tight_layout()
    process_fig.savefig(output / "process.png", dpi=180)
    plt.imsave(output / "intermediate.png", intermediate, cmap="gray", vmin=0, vmax=1)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    print("같은 원판형 SE로 침식 후 팽창합니다. 가는 선과 작은 흰 점은 제거되고 큰 원은 대체로 유지됩니다. 완전한 원상 복원은 아닙니다.")
    print("저장 위치:", output)
    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()

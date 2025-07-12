"""
python quick_vis.py --img "C:\\path\\image.jpg" --min_area 25 --max_area 400
검출 좌표마다 초록 ● 표시 → annotated_<원본>.jpg 저장
"""
import argparse, cv2
from pathlib import Path
from label import detect_spots, fill_big_white

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--min_area", type=int, default=250)
    ap.add_argument("--max_area", type=int, default=400)
    ap.add_argument("--big_area", type=int, default=400)
    args = ap.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise RuntimeError("이미지 불러오기 실패")

    # 1) 카운트용: 원본 복사본으로 spot 검출
    pts, _, _ = detect_spots(img.copy(), args.min_area, args.max_area)

    # 2) 시각화: 큰 덩어리 → 내부 초록 ● + 작은 spot ●
    fill_big_white(img, args.big_area)
    for x, y in pts.astype(int):
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

    out = Path(args.img).with_name("annotated_" + Path(args.img).name)
    cv2.imwrite(str(out), img)
    print(f"→ {out} 저장, 검출 점 개수: {len(pts)}")

if __name__ == "__main__":
    main()

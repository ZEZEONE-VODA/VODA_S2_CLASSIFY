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
    ap.add_argument("--min_area", type=int, default=100)   # ← 기본값도 실제 면적 단위로
    ap.add_argument("--max_area", type=int, default=400)
    ap.add_argument("--big_area", type=int, default=400)
    args = ap.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise RuntimeError("이미지 불러오기 실패")

    # ── 1) spot 검출 ──────────────────────────────────────
    # detect_spots 안에서는 sigma 로만 걸러지므로 면적 필터를 직접 적용
    pts, _, _ = detect_spots(
        img.copy(),
        min_area=args.min_area,
        max_area=args.max_area,
        min_threshold=100,       # ← 필요 시 인자 추가
        max_threshold=500
    )

    # ── 2) 시각화: 큰 덩어리 + 작은 spot 표식 ─────────────
    fill_big_white(img, args.big_area)
    for x, y in pts.astype(int):
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

    out = Path(args.img).with_name("annotated_" + Path(args.img).name)
    cv2.imwrite(str(out), img)
    print(f"→ {out} 저장, 검출 점 개수: {len(pts)}")

if __name__ == "__main__":
    main()

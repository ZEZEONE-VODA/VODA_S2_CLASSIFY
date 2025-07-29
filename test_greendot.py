# test_greendot.py
import argparse
import json
from pathlib import Path
import cv2

from analyser import analyse_bgr  # ← 당신 모듈 파일명 맞춰서 수정

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def annotate(img, result, font_scale=0.6, thickness=2):
    lines = [
        f"Label: {result['label']}",
        f"max_cluster: {result['max_cluster']}",
        f"uniformity: {result['uniformity']:.3f}",
        f"n_spots: {result['n_spots']}",
    ]
    y = 25
    for t in lines:
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
        y += 22
    return img

def iter_images(path: Path):
    if path.is_dir():
        for p in path.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                yield p
    elif path.is_file() and path.suffix.lower() in IMG_EXTS:
        yield path

def process_one(img_path: Path, out_dir: Path, save_json: bool, kwargs):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[!] 읽기 실패: {img_path}")
        return

    res = analyse_bgr(img, **kwargs)
    annotated = annotate(res["annotated"].copy(), res)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / f"{img_path.stem}_annotated.png"
    cv2.imwrite(str(out_img), annotated)
    print(f"[+] 이미지 저장: {out_img}")

    if save_json:
        out_json = out_dir / f"{img_path.stem}_result.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in res.items() if k != "annotated"},
                      f, ensure_ascii=False, indent=2)
        print(f"[+] JSON 저장:  {out_json}")

def main():
    ap = argparse.ArgumentParser(
        description="Green-dot 분포 테스트 러너 (로컬 전용)")
    ap.add_argument("input", nargs="?", default=".",
                    help="이미지 파일 또는 폴더 (기본: 현재 폴더)")
    ap.add_argument("-o", "--out", default="results", help="출력 폴더")
    ap.add_argument("--json", action="store_true", help="JSON 결과 저장")

    # ---- 분석 변수들 ----
    ap.add_argument("--max-clu-thr", type=int, default=15)
    ap.add_argument("--min-area", type=int, default=130)
    ap.add_argument("--max-area", type=int, default=400)
    ap.add_argument("--min-th", type=int, default=100, dest="min_threshold")
    ap.add_argument("--max-th", type=int, default=500, dest="max_threshold")
    ap.add_argument("--eps", type=float, default=30.0)
    ap.add_argument("--min-samples", type=int, default=6)
    ap.add_argument("--uni-thr", type=float, default=0.89)
    ap.add_argument("--big-area", type=int, default=200)

    args = ap.parse_args()

    kwargs = dict(
        max_clu_thr=args.max_clu_thr,
        min_area=args.min_area,
        max_area=args.max_area,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        eps=args.eps,
        min_samples=args.min_samples,
        uni_thr=args.uni_thr,
        big_area=args.big_area,
    )

    in_path = Path(args.input)
    out_dir = Path(args.out)

    imgs = list(iter_images(in_path))
    if not imgs:
        print("[!] 처리할 이미지가 없습니다.")
        return

    for p in imgs:
        process_one(p, out_dir, args.json, kwargs)

if __name__ == "__main__":
    main()

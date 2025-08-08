#!/usr/bin/env python3
# test_greendot.py
# Green-dot 분포 일괄 테스트 러너 (진행률 바 + A/B별 통계 + 결과 저장)

import argparse
import json
from pathlib import Path
from collections import defaultdict   # ### NEW
import cv2
from tqdm import tqdm                 # 진행률 바
from analyser import analyse_bgr      # ← 모듈 경로/함수명 맞게 수정

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ────────────────────────── 유틸 ──────────────────────────
def annotate(img, result, font_scale: float = 0.6, thickness: int = 2):
    """분석 결과를 이미지 좌상단에 텍스트로 오버레이"""
    lines = [
        f"Label: {result['label']}",
        f"max_cluster: {result['max_cluster']}",      # 클러스터 크기
        f"uniformity: {result['uniformity']:.3f}",
        f"n_spots: {result['n_spots']}",
    ]
    y = 25
    for t in lines:
        cv2.putText(
            img,
            t,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += 22
    return img


def iter_images(path: Path):
    """(재귀 없이) 폴더 또는 단일 파일에서 이미지 경로를 순회"""
    if path.is_dir():
        for p in sorted(path.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                yield p
    elif path.is_file() and path.suffix.lower() in IMG_EXTS:
        yield path


def process_one(img_path: Path, out_dir: Path, save_json: bool, kwargs: dict):
    """이미지 하나를 분석하고 결과 파일을 저장, 통계용 dict 반환"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[!] 읽기 실패: {img_path}")
        return None

    res = analyse_bgr(img, **kwargs)
    annotated = annotate(res["annotated"].copy(), res)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / f"{img_path.stem}_annotated.png"
    cv2.imwrite(str(out_img), annotated)

    if save_json:
        out_json = out_dir / f"{img_path.stem}_result.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in res.items() if k != "annotated"},
                f,
                ensure_ascii=False,
                indent=2,
            )

    return {
        "name": img_path.name,
        "label": res["label"],
        "uniformity": res["uniformity"],
        "max_cluster": res["max_cluster"],      # ### NEW
    }


# ────────────────────────── 메인 ──────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Green-dot 분포 일괄 테스트 러너 (로컬 전용)"
    )
    ap.add_argument(
        "input",
        nargs="?",
        default=".",
        help="이미지 파일 또는 폴더 (기본: 현재 폴더)",
    )
    ap.add_argument("--json", action="store_true", help="이미지별 JSON 저장")

    # 분석 파라미터
    ap.add_argument("--max-clu-thr", type=int, default=20)
    ap.add_argument("--min-area", type=int, default=200)
    ap.add_argument("--max-area", type=int, default=2000)
    ap.add_argument("--min-th", type=int, default=100, dest="min_threshold")
    ap.add_argument("--max-th", type=int, default=500, dest="max_threshold")
    ap.add_argument("--eps", type=float, default=55.0)
    ap.add_argument("--min-samples", type=int, default=6)
    ap.add_argument("--uni-thr", type=float, default=0.90)
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

    in_path = Path(args.input).resolve()
    out_dir = Path("output3")
    imgs = list(iter_images(in_path))

    if not imgs:
        print("[!] 처리할 이미지가 없습니다.")
        return

    # ----- 분석 루프 (진행률 표시) -----
    results = []
    for p in tqdm(imgs, desc="Analysing", unit="img"):
        r = process_one(p, out_dir, args.json, kwargs)
        if r:
            results.append(r)

    if not results:
        print("[!] 분석 결과가 없습니다.")
        return

    # ----- 통계/요약 (A/B별) -----    ### NEW/CHG
    stats = defaultdict(lambda: {"uni_sum": 0.0, "clu_sum": 0, "count": 0})
    for r in results:
        lab = r["label"]
        stats[lab]["uni_sum"] += r["uniformity"]
        stats[lab]["clu_sum"] += r["max_cluster"]
        stats[lab]["count"]   += 1

    # 평균 계산
    for lab in stats:
        n = stats[lab]["count"]
        stats[lab]["uni_avg"] = stats[lab]["uni_sum"] / n if n else 0
        stats[lab]["clu_avg"] = stats[lab]["clu_sum"] / n if n else 0

    # ----- 출력 -----               ### NEW/CHG
    print("\n========= 결과 요약 =========")
    for r in results:
        print(
            f"{r['name']:<25} | {r['label']} "
            f"| uni={r['uniformity']:.3f} | max_clu={r['max_cluster']}"
        )
    print("------------------------------------------")
    print(
        f"총 {len(results)}개  |  "
        f"A: {stats['A']['count']}  "
        f"B: {stats['B']['count']}"
    )
    print(
        f"A 평균 → uniformity={stats['A']['uni_avg']:.3f} , "
        f"max_cluster={stats['A']['clu_avg']:.2f}"
    )
    print(
        f"B 평균 → uniformity={stats['B']['uni_avg']:.3f} , "
        f"max_cluster={stats['B']['clu_avg']:.2f}"
    )

    # summary.json 저장                ### NEW/CHG
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total": len(results),
                "A": {
                    "count": stats["A"]["count"],
                    "average_uniformity": stats["A"]["uni_avg"],
                    "average_max_cluster": stats["A"]["clu_avg"],
                },
                "B": {
                    "count": stats["B"]["count"],
                    "average_uniformity": stats["B"]["uni_avg"],
                    "average_max_cluster": stats["B"]["clu_avg"],
                },
                "images": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[+] summary.json 저장: {summary_path}")


if __name__ == "__main__":
    main()

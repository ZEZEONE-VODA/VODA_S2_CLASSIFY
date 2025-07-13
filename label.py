# pip install opencv-python numpy pandas scikit-learn scipy
import argparse, shutil
from pathlib import Path
import cv2, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.stats import entropy

DOT_R   = 5                      # fill_big_white 에서 쓰는 초록 점 반경(px)
MIN_GAP = DOT_R * 2 + 1          # 겹침 허용 안 함(≥ 11 px)

# ────────────── 1) spot 검출 ──────────────
def detect_spots(img_bgr, min_area=300, max_area=2000,
                 min_threshold=150, max_threshold=500):
    """SimpleBlobDetector 기반 spot 좌표·면적 반환"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea, p.minArea, p.maxArea = True, min_area, max_area
    p.minThreshold, p.maxThreshold       = min_threshold, max_threshold
    p.filterByColor, p.blobColor         = True, 255
    p.filterByCircularity = p.filterByInertia = p.filterByConvexity = False

    kps  = cv2.SimpleBlobDetector_create(p).detect(gray)
    pts  = np.array([kp.pt for kp in kps], dtype=np.float32)
    area = np.array([kp.size**2 * np.pi / 4 for kp in kps], dtype=np.float32)
    return pts, area, gray / 255.0

# ────────────── 2) green-circle(=raw) 판별 ──────────────
def has_green_circle(img_bgr, min_radius=50, max_radius=300, min_votes=30):
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    blur = cv2.GaussianBlur(mask, (9, 9), 2)

    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=100, param2=min_votes,
        minRadius=min_radius, maxRadius=max_radius
    )
    return circles is not None

# ────────────── 3) 큰 흰 덩어리 내부를 작은 초록 ●로 채우기 ──────────────
def fill_big_white(
        img_bgr,
        min_area=1_000, max_area=20_000, thresh_val=100,
        dot_radius=5, dot_step=14, morph_ksize=9,
        return_pts=False):                 # ★ 추가

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bin_ = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, k)

    filled = []                            # ★ 찍은 점을 담을 리스트
    cnts, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        mask = np.zeros_like(gray, np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)

        h, w = mask.shape
        for y in range(0, h, dot_step):
            for x in range(0, w, dot_step):
                if mask[y, x]:
                    cv2.circle(img_bgr, (x, y), dot_radius, (0, 255, 0), -1)
                    if return_pts:         # ★ 좌표 저장
                        filled.append((x, y))

    if return_pts:
        return np.array(filled, np.float32)
    # return None 생략 시 기본 None 반환


# ────────────── 4) 지표 계산 함수들 ──────────────
def nn_cv(pts):
    if len(pts) < 2:
        return float("nan")
    dists = (
        NearestNeighbors(n_neighbors=2)
        .fit(pts)
        .kneighbors(pts)[0][:, 1]
    )
    return float(dists.std() / dists.mean())

def cluster_max(pts, eps=30, min_samples=2):
    if len(pts) < min_samples:
        return 0
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    if np.all(labels == -1):
        return 0
    sizes = np.bincount(labels[labels != -1])
    return int(sizes.max())

def grid_uniformity(pts, h, w, n=12):
    if len(pts) == 0:
        return 0.0
    gx, gy = w / n, h / n
    hist = np.zeros(n * n, int)
    for x, y in pts:
        ix, iy = min(int(x // gx), n - 1), min(int(y // gy), n - 1)
        hist[iy * n + ix] += 1
    return entropy(hist, base=np.e) / np.log(n * n)

def has_overlap(pts, min_gap=MIN_GAP):
    if len(pts) < 2:
        return False
    nn = NearestNeighbors(n_neighbors=2).fit(pts)
    d  = nn.kneighbors(pts, return_distance=True)[0][:, 1]
    return np.any(d < min_gap)

# ────────────── analyse() ──────────────
def analyse(path: Path, args):
    img  = cv2.imread(str(path))
    # 1) 원래 스팟 검출
    pts, _, _ = detect_spots(
        img,
        args.min_area, args.max_area,
        args.min_threshold, args.max_threshold
    )

    # 2) 시각화용 흰 덩어리 채우기 점까지 받아오기
    vis = img.copy()
    filled_pts = fill_big_white(vis, args.big_area, return_pts=True)

    # 3) pts + filled_pts 합치기
    if filled_pts is not None and filled_pts.size:
        all_pts = np.vstack([pts, filled_pts])
    else:
        all_pts = pts

    # 4) overlap / cluster_max / uniformity 계산
    overlap    = has_overlap(all_pts)
    max_clu_sz = cluster_max(all_pts, eps=args.eps, min_samples=args.min_samples)
    uni_val    = grid_uniformity(all_pts, *img.shape[:2])

    # 5) 진단용 거리 지표
    n_spots = len(all_pts)
    if n_spots > 1:
        dists = (
            NearestNeighbors(n_neighbors=2)
            .fit(all_pts)
            .kneighbors(all_pts)[0][:, 1]
        )
        min_nn_dist = float(dists.min())
        nn_cv_val   = float(dists.std() / dists.mean())
    else:
        min_nn_dist = float("nan")
        nn_cv_val   = float("nan")

    # 6) 클러스터 개수 계산 (빈 배열 또는 소수 클러스터 건너뜀)
    if all_pts.ndim != 2 or all_pts.shape[0] < args.min_samples:
        n_clusters = 0
    else:
        labels = DBSCAN(eps=args.eps, min_samples=args.min_samples) \
                   .fit_predict(all_pts)
        # -1 은 노이즈 레이블이므로 제외
        n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))

    # 7) A/B 판정
    label = "A" if (
        max_clu_sz < args.max_clu_thr
        and uni_val    >= args.uni_thr
    ) else "B"

    # 8) 시각화 : 원래 스팟만 그려 주기
    for x, y in pts.astype(int):
        cv2.circle(vis, (x, y), DOT_R, (0, 255, 0), -1)

    # 9) 결과 dict 반환 (CSV에 모든 컬럼으로)
    return {
        "file":         path.name,
        "overlap":      overlap,
        "max_cluster":  max_clu_sz,
        "uniformity":   uni_val,
        "n_spots":      n_spots,
        "min_nn_dist":  min_nn_dist,
        "nn_cv":        nn_cv_val,
        "n_clusters":   n_clusters,
        "label":        label,
        "vis":          vis,
        "path":         path
    }


# ────────────── main() ──────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir",  required=True)
    ap.add_argument("--dest_dir", required=True)
    ap.add_argument("--out_csv",  default="result.csv")

    ap.add_argument("--threshold",    type=float, default=0.55)
    ap.add_argument("--ratio",        type=float, default=0.40)
    ap.add_argument("--eps",          type=float, default=30)
    ap.add_argument("--min_samples",  type=int,   default=3)
    ap.add_argument("--uni_thr",      type=float, default=0.85)
    # ★ 새 옵션
    ap.add_argument("--max_clu_thr",  type=int,   default=6)

    ap.add_argument("--min_area",      type=int, default=16)
    ap.add_argument("--max_area",      type=int, default=400)
    ap.add_argument("--min_threshold", type=int, default=150)
    ap.add_argument("--max_threshold", type=int, default=500)
    ap.add_argument("--factor",        type=float, default=4.0)

    ap.add_argument("--big_area", type=int, default=400)
    ap.add_argument("--min_radius", type=int, default=30)
    ap.add_argument("--max_radius", type=int, default=300)
    ap.add_argument("--min_votes",  type=int, default=30)
    args = ap.parse_args()

    imgs = sorted(Path(args.img_dir).glob("*.png")) + \
           sorted(Path(args.img_dir).glob("*.jpg"))
    if not imgs:
        raise RuntimeError("이미지가 없습니다.")

    for sub in ("A", "B", "A_raw", "B_raw"):
        d = Path(args.dest_dir) / sub
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in imgs:
        r = analyse(p, args)
        rows.append(r)

        # 저장·카피는 여기서 바로
        shutil.copy2(r["path"], Path(args.dest_dir) / r["label"])
        raw_dir = Path(args.dest_dir) / (r["label"] + "_raw")
        raw_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(raw_dir / f"annotated_{r['file']}"), r["vis"])

    # ★ 동일 임계값 사용
    hard_B = [r for r in rows if r["max_cluster"] >= args.max_clu_thr]
    cands  = [r for r in rows if r["max_cluster"] <  args.max_clu_thr]

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()

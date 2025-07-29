"""
green‑dot 분포 분석 로직 모듈
(한 파일로 몰아넣어 재사용·테스트 편의 ↑)
"""
from pathlib import Path
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.stats import entropy

# ───────── 공통 상수 ─────────
DOT_R   = 5
MIN_GAP = DOT_R * 2 + 1   # 11px 이하이면 overlap 간주

# ───────── 1) spot 검출 ─────────
def detect_spots(img_bgr, min_area=10, max_area=2000,
                 min_threshold=100, max_threshold=500):
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

# ───────── 3) 큰 흰 덩어리 내부 채우기 ─────────
def fill_big_white(
    img_bgr, min_area=10, max_area=20_000, thresh_val=200,
    dot_radius=5, morph_ksize=9, return_pts=False
):
    # ---- 튜닝 파라미터(필요시 숫자만 바꿔 쓰세요) ----
    AREA_PER_DOT = 100   # 면적당 점 1개 기준(작게 하면 더 많이 찍힘)
    MIN_STEP     = dot_radius * 2  # 점 간격 최소치

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bin_ = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, k)

    filled = []
    cnts, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue

        # --- 면적 기반 점 개수 계산 ---
        n_dots = max(1, int(area // AREA_PER_DOT))

        # 마스크 생성
        mask = np.zeros_like(gray, np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)

        x0, y0, w0, h0 = cv2.boundingRect(c)

        # 목표 개수에 맞춰 스텝 계산(대략적)
        est_step = int(np.sqrt(area / n_dots))
        step = max(MIN_STEP, est_step)

        placed = 0
        off = step // 2
        for y in range(y0 + off, y0 + h0, step):
            for x in range(x0 + off, x0 + w0, step):
                if mask[y, x]:
                    cv2.circle(img_bgr, (x, y), dot_radius, (0, 255, 0), -1)
                    if return_pts:
                        filled.append((x, y))
                    placed += 1
                    if placed >= n_dots:  # 목표 개수 채우면 중단
                        break
            if placed >= n_dots:
                break

    if return_pts:
        filled = list(set(map(tuple, filled)))          # 좌표 중복 제거
        return np.array(filled, np.float32).reshape(-1, 2)


# ───────── 유틸 지표 함수 ─────────
def has_overlap(pts, min_gap=MIN_GAP):
    if len(pts) < 2:
        return False
    d = (
        NearestNeighbors(n_neighbors=2)
        .fit(pts)
        .kneighbors(pts)[0][:, 1]
    )
    return bool(np.any(d < min_gap))

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
    return float(entropy(hist, base=np.e) / np.log(n * n))

# ───────── 메인 분석 함수 ─────────
def analyse_bgr(
    img_bgr,
    *,
    max_clu_thr: int = 15,
    min_area: int = 10,
    max_area: int = 400,
    min_threshold: int = 100,
    max_threshold: int = 500,
    eps: float = 30.0,
    min_samples: int = 6,
    uni_thr: float = 0.89,
    big_area: int = 200,
    dot_radius: int = 5
):
        # ── 반경·간격을 런타임 계산 ─────────────────────
    DOT_R   = dot_radius
    MIN_GAP = DOT_R * 2 + 1
    pts, _, _ = detect_spots(img_bgr, min_area, max_area,
                             min_threshold, max_threshold)

    vis = img_bgr.copy()
    filled_pts = fill_big_white(
        vis,
        min_area=big_area,       # ← 꼭 키워드로!
        max_area=20_000,
        thresh_val=200,
        dot_radius=dot_radius,
        morph_ksize=9,
        return_pts=True
    )

        # 2) 아직 안 그려진 blob 점도 그리기
    drawn_set = set()  # (x,y) 정수 좌표 저장용
    if filled_pts is not None and filled_pts.size:
        for x, y in filled_pts.astype(int):
            drawn_set.add((x, y))

    for x, y in pts.astype(int):
        if (x, y) not in drawn_set:                 # 중복 방지
            cv2.circle(vis, (x, y), DOT_R, (0, 255, 0), -1)
            drawn_set.add((x, y))

    # 3) 실제 그려진 점 좌표 배열
    drawn_pts = np.array(list(drawn_set), dtype=np.float32)

    all_pts = np.vstack([pts, filled_pts]) if filled_pts.size else pts
        # --- 추가(선택): 고유 좌표만 카운트 ---
    uniq_all = np.unique(all_pts.astype(int), axis=0)
    n_spots  = int(len(uniq_all))

    overlap = has_overlap(all_pts, min_gap=MIN_GAP)
    max_clu_sz = cluster_max(all_pts, eps=eps, min_samples=min_samples)
    uni_val    = grid_uniformity(all_pts, *img_bgr.shape[:2])

    n_spots = int(len(all_pts))
    if n_spots > 1:
        dists = (
            NearestNeighbors(n_neighbors=2)
            .fit(all_pts)
            .kneighbors(all_pts)[0][:, 1]
        )
        min_nn_dist = float(dists.min())
        nn_cv_val   = float(dists.std() / dists.mean())
    else:
        min_nn_dist = nn_cv_val = float("nan")

    if all_pts.shape[0] >= min_samples:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(all_pts)
        n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    else:
        n_clusters = 0

    label = "A" if (max_clu_sz < max_clu_thr and uni_val >= uni_thr) else "B"

    return {
        "label":        label,
        "overlap":      bool(overlap),
        "max_cluster":  int(max_clu_sz),
        "uniformity":   float(uni_val),
        "n_spots":      n_spots,
        "min_nn_dist":  min_nn_dist,
        "nn_cv":        nn_cv_val,
        "n_clusters":   n_clusters,
        "annotated":    vis,  # 시각화 BGR
    }

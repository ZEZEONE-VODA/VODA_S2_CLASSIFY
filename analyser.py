"""
green-dot 분포 분석 로직 (non-overlap 보장 버전)
"""
from pathlib import Path
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.stats import entropy
from scipy.spatial import cKDTree   # 거리 빠른 계산용

# ───────── 공통 상수 ─────────
DOT_R   = 3                # 시각화·충돌 반경(px)
MIN_GAP = DOT_R * 2 + 1         # 이 거리 미만이면 겹침으로 간주

# ───────── 1) spot 검출 ─────────
def detect_spots(img_bgr,
                 min_area=130, max_area=2000,
                 min_threshold=100, max_threshold=500):
    """SimpleBlobDetector 로 흰점 검출(겹침 차단 포함)"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea, p.minArea, p.maxArea  = True, min_area, max_area
    p.filterByColor, p.blobColor          = True, 255
    p.minThreshold, p.maxThreshold        = min_threshold, max_threshold
    p.filterByCircularity = p.filterByInertia = p.filterByConvexity = False
    p.minDistBetweenBlobs = float(MIN_GAP)  # ← 핵심: 서로 MIN_GAP 이상 떨어진 키포인트만

    detector = cv2.SimpleBlobDetector_create(p)
    kps      = detector.detect(gray)

    pts  = np.array([kp.pt for kp in kps], dtype=np.float32)
    area = np.array([kp.size**2 * np.pi / 4 for kp in kps], dtype=np.float32)
    return pts, area, gray / 255.0


# ───────── 2) 큰 흰 덩어리 내부 채우기 ─────────
def fill_big_white(img_bgr,
                   existing_pts,
                   *,
                   min_area=130, max_area=20_000,
                   thresh_val=200,
                   dot_radius=DOT_R,
                   morph_ksize=9,
                   return_pts=False):
    """
    • 흰 덩어리 내부를 일정 간격 점으로 채움
    • 이미 찍힌 existing_pts 와 MIN_GAP 이상 떨어진 지점만 추가
    """
    # --- 설정값 ---
    AREA_PER_DOT = 100            # 면적당 점 1개 목표
    MIN_STEP     = dot_radius*2   # 점 간 최소 간격

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bin_ = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, k)

    tree = cKDTree(existing_pts) if len(existing_pts) else None
    filled = []

    cnts, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue

        # 예상 점 개수 및 간격 계산
        n_dots  = max(1, int(area // AREA_PER_DOT))
        est_step = int(np.sqrt(area / n_dots))
        step     = max(MIN_STEP, est_step)
        off      = step // 2

        mask = np.zeros_like(gray, np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        x0, y0, w0, h0 = cv2.boundingRect(c)

        placed = 0
        for y in range(y0 + off, y0 + h0, step):
            for x in range(x0 + off, x0 + w0, step):
                if not mask[y, x]:
                    continue
                # ── 거리 체크 ──
                if tree is not None and tree.query((x, y), k=1)[0] < MIN_GAP:
                    continue
                cv2.circle(img_bgr, (x, y), dot_radius, (0, 255, 0), -1)
                if return_pts:
                    filled.append((x, y))
                placed += 1
                # KDTree 업데이트
                if tree is None:
                    tree = cKDTree(np.array([[x, y]], dtype=np.float32))
                else:
                    tree = cKDTree(np.vstack([tree.data, (x, y)]))
                if placed >= n_dots:
                    break
            if placed >= n_dots:
                break

    if return_pts:
        return np.array(filled, dtype=np.float32).reshape(-1, 2)


# ───────── 유틸 지표 함수 ─────────
def has_overlap(pts, min_gap=MIN_GAP):
    if len(pts) < 2:
        return False
    d = NearestNeighbors(n_neighbors=2).fit(pts).kneighbors(pts)[0][:, 1]
    return bool(np.any(d < min_gap))

def cluster_max(pts, eps=50, min_samples=3):
    if len(pts) < min_samples:
        return 0
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    if np.all(labels == -1):
        return 0
    sizes = np.bincount(labels[labels != -1])
    return int(sizes.max())

def grid_uniformity(pts, h, w, n=20):
    """n×n 그리드 (직사각형) 기반 엔트로피. 필요 시 수정."""
    if len(pts) == 0:
        return 0.0
    gx, gy = w / n, h / n
    hist = np.zeros(n * n, int)
    for x, y in pts:
        ix, iy = min(int(x // gx), n-1), min(int(y // gy), n-1)
        hist[iy * n + ix] += 1
    return float(entropy(hist, base=np.e) / np.log(n * n))


# ───────── 메인 분석 함수 ─────────
def analyse_bgr(img_bgr,
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
                dot_radius: int = DOT_R):
    """BGR 이미지 → 분석 dict (겹침 0 보장)"""

    global DOT_R, MIN_GAP
    DOT_R   = dot_radius
    MIN_GAP = DOT_R * 2 + 1

    # 1) 기존 blob 추출 (겹침 차단 적용됨)
    pts, _, _ = detect_spots(img_bgr, min_area, max_area,
                             min_threshold, max_threshold)
    pts = pts.reshape(-1, 2)

    # 2) 큰 흰 덩어리 채우기 (distance check 포함)
    vis = img_bgr.copy()
    filled_pts = fill_big_white(vis,
                                existing_pts=pts,
                                min_area=big_area,
                                max_area=20_000,
                                dot_radius=dot_radius,
                                return_pts=True)
    if filled_pts is None:
        filled_pts = np.empty((0, 2), dtype=np.float32)
    else:
        filled_pts = filled_pts.reshape(-1, 2)

    # 3) 최종 좌표 병합 + MIN_GAP 재검증
    all_pts_raw = np.vstack([pts, filled_pts])
    if len(all_pts_raw):
        keep = []
        tree = cKDTree(np.empty((0, 2)))
        for p in all_pts_raw:
            if len(tree.data) == 0 or tree.query(p, k=1)[0] >= MIN_GAP:
                keep.append(p)
                tree = cKDTree(np.vstack([tree.data, p]))
        all_pts = np.array(keep, dtype=np.float32)
    else:
        all_pts = all_pts_raw

    # 4) 시각화 (중복 없는 좌표만)
    for x, y in all_pts.astype(int):
        cv2.circle(vis, (x, y), DOT_R, (0, 255, 0), -1)

    # ─── 지표 계산 ───
    n_spots     = int(len(all_pts))
    overlap     = has_overlap(all_pts, MIN_GAP)
    max_clu_sz  = cluster_max(all_pts, eps=eps, min_samples=min_samples)
    uni_val     = grid_uniformity(all_pts, *img_bgr.shape[:2])

    if n_spots > 1:
        dists = NearestNeighbors(n_neighbors=2).fit(all_pts).kneighbors(all_pts)[0][:, 1]
        min_nn_dist = float(dists.min())
        nn_cv_val   = float(dists.std() / dists.mean())
    else:
        min_nn_dist = nn_cv_val = float("nan")

    if n_spots >= min_samples:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(all_pts)
        n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    else:
        n_clusters = 0

    label = "A" if (max_clu_sz < max_clu_thr and uni_val >= uni_thr) else "B"

    return {
        "label":       label,
        "overlap":     bool(overlap),
        "max_cluster": int(max_clu_sz),
        "uniformity":  float(uni_val),
        "n_spots":     n_spots,
        "min_nn_dist": min_nn_dist,
        "nn_cv":       nn_cv_val,
        "n_clusters":  n_clusters,
        "annotated":   vis,
    }

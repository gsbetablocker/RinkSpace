"""Re-read the short, motion-blurred board pan at the peak of Jin S07's zoom."""

from argparse import ArgumentParser

import cv2
import numpy as np

from flow_depths import bind
from lane_end_evidence import WORK, foreground
from lane_review import evaluate
from repair_motion import project
from review import camera_state, native_for, source_frames
from shared import PREPARED, arrays, read, save, write
from source_bundle import flow, native_run, relative


def measure():
    # This board crop and the coarse initial pan are specific to source 57.62–58.12 seconds.
    # Mutual printed-feature matches seed the search. Each admitted pair additionally
    # passes reverse optical flow, patch correlation and a separate wide template search.
    n = "jin_s07"
    s = read(PREPARED / "setup.json")["shots"][n]
    parent = relative(WORK / "delivery")
    b = camera_state(parent)
    native = native_for(parent, "jin")
    pts = b[n + "_pts"]
    tt = np.flatnonzero((pts / 12800 >= 57.62) & (pts / 12800 <= 58.12))
    gray = []
    masks = []
    images = []
    cv2.setNumThreads(1)
    for j, (pt, img) in zip(tt, source_frames(s["source"], pts[tt]), strict=True):
        gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        images.append(img)
        m = foreground(s, None, j, pt, exclude_uncertain_region=False)
        mask = np.ones((1080, 1920), np.uint8) * 255 if m is not None else np.zeros((1080, 1920), np.uint8)
        if m is not None:
            mask[cv2.dilate(m, np.ones((25, 25), np.uint8)) > 0] = 0
        mask[:350] = 0
        mask[616:] = 0
        mask[320:650, 958:1242] = 0
        mask[:, :15] = 0
        mask[:, -15:] = 0
        masks.append(mask)
    sift = cv2.SIFT_create(nfeatures=600, contrastThreshold=0.004, edgeThreshold=10)
    skey = [sift.detectAndCompute(g, m) for g, m in zip(gray, masks)]
    bf = cv2.BFMatcher()
    arr = []
    for j in range(1, len(tt)):
        A, B = gray[j - 1], gray[j]
        (k0, d0), (k1, d1) = skey[j - 1 : j + 1]
        H = np.array([[1.0, 0.0, -43.0], [0.0, 1.0, 0.0]])
        if d0 is not None and d1 is not None and min(len(d0), len(d1)) >= 3:

            def best(d, e):
                return {
                    (m.queryIdx, m.trainIdx)
                    for m, l in bf.knnMatch(d, e, k=2)
                    if m.distance < 0.81 * l.distance
                }

            pairs = best(d0, d1) & {(v, u) for u, v in best(d1, d0)}
            if len(pairs) >= 4:
                u, v = zip(*pairs)
                sa = np.float32([k0[i].pt for i in u])
                sb = np.float32([k1[i].pt for i in v])
                HH, h = cv2.estimateAffinePartial2D(sa, sb, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if HH is not None and h.sum() >= 4:
                    H = HH
        x = cv2.goodFeaturesToTrack(A, 250, 0.002, 10, mask=masks[j - 1], blockSize=5)
        if x is None:
            continue
        x = x.reshape(-1, 2)
        cv2.cornerSubPix(A, x, (4, 4), (-1, -1), (3, 16, 0.025))
        init = (x @ H[:, :2].T + H[:, 2]).astype("f4")
        opt = dict(
            winSize=(23, 23), maxLevel=3, criteria=(3, 32, 0.012), flags=cv2.OPTFLOW_USE_INITIAL_FLOW
        )
        y, ok, _ = cv2.calcOpticalFlowPyrLK(A, B, x, init.copy(), **opt)
        back, ok2, _ = cv2.calcOpticalFlowPyrLK(B, A, y, x.copy(), **opt)
        good = (
            (ok.ravel() > 0)
            & (ok2.ravel() > 0)
            & (np.linalg.norm(x - back, axis=1) < 0.65)
            & (np.linalg.norm(y - init, axis=1) < 22.0)
        )
        idx = np.flatnonzero(good & (y > [12, 12]).all(1) & (y < [1907, 1067]).all(1))
        idx = idx[masks[j][np.rint(y[idx, 1]).astype(int), np.rint(y[idx, 0]).astype(int)] > 0]
        for v in idx:
            l = cv2.getRectSubPix(A, (19, 19), tuple(x[v])).astype(float)
            r = cv2.getRectSubPix(B, (19, 19), tuple(y[v])).astype(float)
            if min(l.std(), r.std()) < 2.1:
                continue
            cor = np.corrcoef(l.ravel(), r.ravel())[0, 1]
            if cor > 0.94:
                xx, yy = np.rint(x[v]).astype(int)
                u0 = max(0, xx - 105)
                u1 = min(1920, xx + 81)
                v0 = max(0, yy - 32)
                v1 = min(1080, yy + 33)
                tpl = A[yy - 10 : yy + 11, xx - 10 : xx + 11]
                z = cv2.matchTemplate(B[v0:v1, u0:u1], tpl, cv2.TM_CCOEFF_NORMED)
                _, mx, _, pk = cv2.minMaxLoc(z)
                if mx > 0.93 and np.linalg.norm(y[v] - [u0 + pk[0] + 10, v0 + pk[1] + 10]) < 2.0:
                    arr.append((tt[j], x[v], y[v], cor))
        if abs(pts[tt[j]] / 12800 - 57.86) < 0.0001:
            tile = images[j].copy()
            for row in [row for row in arr if row[0] == tt[j]]:
                cv2.arrowedLine(
                    tile,
                    np.rint(row[1]).astype(int),
                    np.rint(row[2]).astype(int),
                    (50, 175, 80),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imwrite(
                str(WORK / "delivery/inspection/jin_s07_board_pairs.jpg"), cv2.resize(tile, (960, 540))
            )
    i, x, y, cor = map(np.asarray, zip(*arr))
    w, err = bind(
        native, s, b[n + "_rigs"], b[n + "_controls"], b[n + "_times"], i - 1, x, np.zeros(len(i), int)
    )
    good = np.isfinite(w).all(1)
    i, x, y, w, cor = i[good], x[good], y[good], w[good], cor[good]
    de = project(native, s, b, n, i, w) - project(native, s, b, n, i - 1, w) - (y - x)
    save(
        WORK / "delivery/inspection/jin_s07_board_diagnostics.npz",
        index=i,
        left_uv=x,
        uv=y,
        world=w,
        correlation=cor,
        model_error=de,
        pts=pts,
        source_sha256=s["source_sha256"],
    )
    for j in np.unique(i):
        if 57.80 - 1e-5 <= pts[j] / 12800 <= 57.94 + 1e-5:
            sel = i == j
            print(
                pts[j] / 12800,
                sel.sum(),
                round(np.ptp(x[sel, 0])),
                round(float(np.median(np.linalg.norm(de[sel], axis=1))), 2),
                np.round(np.median(de[sel], axis=0), 1).tolist(),
                round(float(np.median(np.linalg.norm((y - x)[sel], axis=1))), 2),
            )


def fit():
    n = "jin_s07"
    parent = relative(WORK / "delivery")
    b = camera_state(parent)
    s = read(PREPARED / "setup.json")["shots"][n]
    native = native_for(parent, "jin")
    cert = arrays(WORK / "delivery/inspection/jin_s07_board_diagnostics.npz")
    i = cert["index"]
    u = cert["left_uv"]
    v = cert["uv"]
    assert cert["source_sha256"].item() == s["source_sha256"]
    held = (np.floor(u[:, 0] / 85).astype(int) + np.floor(u[:, 1] / 55).astype(int)) % 4 == 0
    w, err = bind(
        native, s, b[n + "_rigs"], b[n + "_controls"], b[n + "_times"], i - 1, u, np.zeros(len(i), int)
    )
    ok = np.isfinite(w).all(1)
    d = np.linalg.norm(cert["model_error"], axis=1)
    supported = []
    for j in np.unique(i):
        take = (i == j) & ~held & ok
        if np.count_nonzero(take) >= 5 and np.ptp(u[take, 0]) > 120 and np.median(d[take]) > 1.6:
            supported.append(int(j))
    old = arrays(WORK / "texture" / (n + ".npz"))
    q = arrays(WORK / "texture/quality" / (n + ".npz"))
    vals = dict(
        index=i,
        lag=np.ones(len(i), int),
        left_uv=u,
        uv=v,
        world=w,
        inverse_error=err,
        valid=ok,
        quality=cert["correlation"],
        held=held,
    )
    for k, x in vals.items():
        old[k] = np.concatenate((old[k], x.astype(old[k].dtype)))
    td = WORK / "zoom_texture"
    save(td / (n + ".npz"), **old)
    q["keep"] = np.r_[q["keep"], ok]
    q["transient"] = np.unique(np.r_[q["transient"], supported])
    save(td / "quality" / (n + ".npz"), **q)
    save(WORK / "detail/zoom" / (n + ".npz"), transient=np.array(supported, int), pts=cert["pts"])
    write(
        td / "source_manifest.json",
        dict(
            source=str(WORK / "delivery/inspection/jin_s07_board_diagnostics.npz"),
            source_sha256=s["source_sha256"],
            pairs=len(i),
            held=int(held.sum()),
            supported_pts=cert["pts"][supported].tolist(),
            spatial_partition="(floor(x/85)+floor(y/55)) % 4 == 0",
        ),
    )
    for stage, gain in [("zoom_local", 1.6), ("zoom_track", 3.2)]:
        folder = WORK / stage / n
        fl = flow(gain)
        fl[fl.index("--texture-dir") + 1] = td
        native_run(
            folder,
            parent,
            [
                "--mode",
                "camera",
                "--shots",
                n,
                "--stride",
                "2",
                "--detail-dir",
                WORK / "detail/zoom",
                "--local-detail",
                "--unit-weights",
                "--huber-delta",
                "5.2",
                "--station-seed-dir",
                WORK / "evaluation/release",
                *fl,
            ],
        )
        evaluate(relative(folder), n, WORK / "evaluation" / stage)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["measure", "fit"])
    if parser.parse_args().action == "measure":
        measure()
    else:
        fit()

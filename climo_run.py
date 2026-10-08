#!/usr/bin/env python3
"""
CloudScope Radar climatology - score one month of the MRMS archive, hour by hour, with the same
classifier the live page uses.

    python climo_run.py 2023-06            # one month, hourly
    python climo_run.py 2023-06 --days 2   # first two days only, for a quick test

Archive (NOAA open data on AWS, from 14 Oct 2020 - MRMS v12; NOAA advises against mixing in the
earlier v11 data):
    https://noaa-mrms-pds.s3.amazonaws.com/CONUS/<Product>_<level>/YYYYMMDD/
        MRMS_<Product>_<level>_YYYYMMDD-HHMMSS.grib2.gz

Measured 7 Oct 2026: every product the classifier reads is archived for the whole period,
including NLDN CG (1-14 KB a file) and the 3 km 3-D level Steiner needs. Archived files are
CONUS-wide and grow over the period, ~7 MB per sample moment in 2020 and ~12 MB in 2026, so:
  - LayerCompositeReflectivity_High and EchoTop_18 are skipped - the classifier never reads them
  - Model_0degC_Height (8.5 MB, hourly) is fetched every 6 hours and reused in between

What is the SAME as live: the rules, the classes, Steiner on the 3 km level, the elevated-echo
and decayed-anvil logic, NLDN CG for lightning. Also the SAME as live: satellite cloud tops can raise a cumulus class over echo (never add cloud,
never lower a class, only within 3 km above the radar echo top), and detached-anvil history is
steered by GOES band-14 anvil-level winds (band 8, then 32 kt in every direction, as fallbacks).

What DIFFERS, by necessity:
  - satellite cloud-top HEIGHT comes from band 13 itself - the brightness temperature placed on the
    MRMS freezing level and a 6.5 C/km lapse rate - for all six years. The 2 km height product
    (ACHA2KMC) only begins 23 Mar 2023; using it from then on would put a method change, and a
    step in the statistics, in the middle of the record. A +/-1 km height error moves the parallax
    correction ~0.7 km, inside both the 2.5 km match radius and the 3 km consistency margin.
  - GOES-16 before 7 Apr 2025, GOES-19 from then on, chosen by DATE: GOES-19 files from Oct 2024
    to Apr 2025 come from its checkout at 89.5 W, not the operational 75.2 W slot
  - lightning is NLDN CG only (GLM would add ~1 TB over six years) - so intracloud-only
    lightning is missed and the lightning rule is undercounted
  - isotherm heights come from the MRMS model freezing level plus a lapse rate (no historical
    BUFKIT), so the +5 and -20 C heights are estimates
  - anvil exception part (a) uses the 0 C slice rather than echo bases (no full volume)
  - detached-anvil history is the previous HOUR, not the previous 5-minute frames

Output: out/YYYYMM.npz - every sample's per-pad status and rule bits, and per-cell counts of
violating / watch-or-worse by LOCAL hour, for the merge step.
"""

import argparse
import calendar
import concurrent.futures as cf
import datetime
import logging
import multiprocessing as mp
import os
import re
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import requests

import mrms_classify as mc

BUCKET = "https://noaa-mrms-pds.s3.amazonaws.com"
LOCAL = ZoneInfo("America/New_York")
UA = {"User-Agent": "CloudScope-climatology/1.0"}

# key -> archived product directory. The keys match mrms_classify.PRODUCTS.
ARCHIVE = {
    "comp":  "MergedReflectivityQCComposite_00.50",
    "r0":    "Reflectivity_0C_00.50",
    "r5":    "Reflectivity_-5C_00.50",
    "r10":   "Reflectivity_-10C_00.50",
    "r15":   "Reflectivity_-15C_00.50",
    "r20":   "Reflectivity_-20C_00.50",
    "super": "LayerCompositeReflectivity_Super_00.50",
    "hmax":  "HeightCompositeReflectivity_00.50",
    "vii":   "VII_00.50",
    "cg":    "NLDN_CG_030min_AvgDensity_00.00",
}
UNUSED = ("high", "et18")                       # fetched live for display only
STEINER_DIR, STEINER_KM = "MergedReflectivityQC_03.00", 3.0
GOES_SWITCH = datetime.datetime(2025, 4, 7)
GOES_BUCKETS = ("https://noaa-goes16.s3.amazonaws.com", "https://noaa-goes19.s3.amazonaws.com")
SAT_TOL_MIN, WIND_TOL_MIN = 12, 30
_goes_listings = {}

# Per pad and sample, what the "what if" page needs to re-apply any standoff or threshold without
# re-scoring the radar: the distance (nmi) to the nearest cell of each rule's cloud mask, and the
# MRR the exceptions test. Distances are rounded UP to 0.1 nmi, capped at 25.4 (255 = farther or
# none) - rounding up keeps "within r" exact for any r that is a multiple of 0.1 nmi, because the
# standoff footprint and this distance use the same grid metric. MRR is dBZ x 2 (negative -> 0).
FEAT_DIST = ["cuthru", "cu10", "cu20", "att", "det", "warm", "thick", "ltg"]
FEAT_NAMES = FEAT_DIST + ["mrr1"]
FREEZING_DIR = "Model_0degC_Height_00.50"
FREEZING_EVERY_H = 6
MATCH_MIN = 3.0                                 # a file must lie within this of the sample time
JOB_BUDGET_S = 5.5 * 3600                       # stop cleanly before GitHub's 6 h job limit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(processName)s %(message)s")
_listings = {}


def _session():
    """Retries broken connections AND throttling/server errors, with backoff.

    S3 routinely drops idle keep-alive connections ("RemoteDisconnected ... Retrying" in the
    log) - harmless, retried on a fresh connection. The first version did not retry HTTP status
    errors at all, so a 503 SlowDown under load from 20 parallel jobs would have skipped the hour
    instead of waiting and trying again.
    """
    from urllib3.util.retry import Retry
    retry = Retry(total=8, connect=6, read=6, status=6, backoff_factor=0.8,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(["GET"]), raise_on_status=False)
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(max_retries=retry, pool_maxsize=16))
    s.headers.update(UA)
    return s


def day_listing(sess, product, day):
    """[(datetime, url)] for one product-day, cached for the process."""
    ck = (product, day)
    if ck in _listings:
        return _listings[ck]
    out, token = [], None
    while True:
        url = (f"{BUCKET}/?list-type=2&prefix=CONUS/{product}/{day:%Y%m%d}/"
               + (f"&continuation-token={requests.utils.quote(token)}" if token else ""))
        r = sess.get(url, timeout=90)
        r.raise_for_status()
        for key in re.findall(r"<Key>([^<]+)</Key>", r.text):
            m = re.search(r"_(\d{8}-\d{6})\.grib2\.gz$", key)
            if m:
                out.append((datetime.datetime.strptime(m.group(1), "%Y%m%d-%H%M%S"), f"{BUCKET}/{key}"))
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if not m:
            break
        token = m.group(1)
    out.sort()
    _listings[ck] = out
    return out


def nearest(sess, product, t, tol_min):
    """URL of the archived file nearest t, looking into the neighbouring day near midnight."""
    days = {t.date()}
    if t.hour == 0 and t.minute < tol_min + 1:
        days.add((t - datetime.timedelta(days=1)).date())
    if t.hour == 23 and t.minute > 59 - tol_min - 1:
        days.add((t + datetime.timedelta(days=1)).date())
    best, gap = None, None
    for d in days:
        for ft, url in day_listing(sess, product, datetime.datetime.combine(d, datetime.time())):
            g = abs((ft - t).total_seconds()) / 60.0
            if g <= tol_min and (gap is None or g < gap):
                best, gap = url, g
    return best


def goes_bucket(t):
    return GOES_BUCKETS[0] if t < GOES_SWITCH else GOES_BUCKETS[1]


def goes_nearest(sess, t, product, match, tol_min):
    """(bucket, key) of the archived ABI file nearest t, from the operational GOES-East of the day."""
    base = goes_bucket(t)
    best = None
    for h in (-1, 0, 1):
        tt = t + datetime.timedelta(hours=h)
        prefix = f"ABI-L2-{product}/{tt:%Y}/{tt.timetuple().tm_yday:03d}/{tt:%H}/"
        ck = (base, prefix)
        if ck not in _goes_listings:
            keys, token = [], None
            try:
                while True:
                    url = (f"{base}/?list-type=2&prefix={prefix}"
                           + (f"&continuation-token={requests.utils.quote(token)}" if token else ""))
                    r = sess.get(url, timeout=90)
                    r.raise_for_status()
                    keys += re.findall(r"<Key>([^<]+)</Key>", r.text)
                    m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
                    if not m:
                        break
                    token = m.group(1)
            except Exception:
                keys = []
            _goes_listings[ck] = keys
        for k in _goes_listings[ck]:
            if match not in k:
                continue
            m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", k)
            if not m:
                continue
            y, d, H, M, S = map(int, m.groups())
            st = datetime.datetime(y, 1, 1, H, M, S) + datetime.timedelta(days=d - 1)
            g = abs((st - t).total_seconds()) / 60.0
            if g <= tol_min and (best is None or g < best[0]):
                best = (g, k)
    return (base, best[1]) if best else None


def sat_tops_bt(sess, t, la, lo, z0_m):
    """Parallax-corrected band-13 cloud-top temperature and a height DERIVED from it, on the MRMS
    grid. Only pixels colder than 0 C are placed - nothing warmer could raise a cumulus class.
    Returns the grids (all-NaN when there is no cold cloud), or None when no file was found."""
    from scipy.spatial import cKDTree
    hit = goes_nearest(sess, t, "CMIPC", "C13_", SAT_TOL_MIN)
    if not hit:
        return None
    base, key = hit
    r = sess.get(f"{base}/{key}", timeout=120)
    r.raise_for_status()
    ds, tmp = mc._nc_open(r.content)
    try:
        lat, lon, sl, geom = mc._fixed_grid(ds)
        v = ds.variables["CMI"][sl]
        v = np.asarray(v.filled(np.nan) if np.ma.isMaskedArray(v) else v, float)
    finally:
        ds.close()
        if tmp:
            os.remove(tmp)
    LO, LA = np.meshgrid(lo, la)
    out = {"bt_c": np.full(LA.shape, np.nan, np.float32), "top_m": np.full(LA.shape, np.nan, np.float32)}
    bt = v - 273.15
    cold = np.isfinite(bt) & np.isfinite(lat) & (bt < 0.0)
    if not cold.any():
        return out
    h = z0_m + (-bt[cold]) / mc.LAPSE_C_PER_KM * 1000.0
    dla, dlo = mc.parallax_shift(lat[cold], lon[cold], h, geom)
    tla, tlo = lat[cold] - dla, lon[cold] - dlo
    kx = 111.32 * np.cos(np.radians(float(np.mean(la))))
    xy = lambda a, b: np.column_stack([((b - lo[0]) * kx).ravel(), ((a - la[0]) * 111.32).ravel()])
    d, i = cKDTree(xy(tla, tlo)).query(xy(LA, LO), distance_upper_bound=mc.SAT_MATCH_KM)
    ok = np.isfinite(d)
    out["bt_c"].ravel()[ok] = bt[cold][i[ok]]
    out["top_m"].ravel()[ok] = h[i[ok]]
    return out


def anvil_wind(sess, t):
    """Anvil-level vector-mean wind near the Cape at time t - the live rules (band 14, band 8 as
    the fallback, at least 5 vectors above 400 hPa within 150 km), on the archived files."""
    for band in ("C14_", "C08_"):
        hit = goes_nearest(sess, t, "DMWC", band, WIND_TOL_MIN)
        if not hit:
            continue
        base, key = hit
        try:
            r = sess.get(f"{base}/{key}", timeout=90)
            r.raise_for_status()
            ds, tmp = mc._nc_open(r.content)
            try:
                g = lambda n: np.asarray(ds.variables[n][:], float)
                lat, lon, spd, dirn, p = g("lat"), g("lon"), g("wind_speed"), g("wind_direction"), g("pressure")
                dqf = g("DQF") if "DQF" in ds.variables else np.zeros(lat.shape)
            finally:
                ds.close()
                if tmp:
                    os.remove(tmp)
        except Exception:
            continue
        k = np.cos(np.radians(mc.CAPE_LAT))
        near = ((dqf == 0) & (p <= mc.WIND_TOP_HPA) & (np.hypot((lat - mc.CAPE_LAT) * 111.32,
                (lon - mc.CAPE_LON) * 111.32 * k) <= mc.WIND_RADIUS_KM))
        if int(near.sum()) < mc.WIND_MIN_VECTORS:
            continue
        u = -spd[near] * np.sin(np.radians(dirn[near])) * 3.6
        v = -spd[near] * np.cos(np.radians(dirn[near])) * 3.6
        return {"u_kmh": float(u.mean()), "v_kmh": float(v.mean()),
                "unc_kmh": max(mc.WIND_MIN_UNC_KT * 1.852, float(np.sqrt(u.var() + v.var())))}
    return None


def fetch(sess, url, fill):
    return mc.decode_crop(mc.fetch_url(sess, url), fill)


def score_hours(hours):
    """Score a contiguous run of hourly sample times, in order. Returns a dict of arrays."""
    sess = _session()
    pads = list(mc.SITES)
    rec = {"t": [], "status": [], "red": [], "yel": [], "sat": [], "wind": [], "satup": [], "feat": []}
    acc = None
    z0, z0_t, prev_att, prev_t = None, None, None, None
    skipped = []
    t_start = time.monotonic()
    # 4, not 8: each decode briefly holds a full CONUS grid (~200 MB at peak), and with four
    # processes per job 8 threads apiece could hold ~6 GB at once on a 16 GB runner.
    pool = cf.ThreadPoolExecutor(max_workers=4)
    for t in hours:
        if time.monotonic() - t_start > JOB_BUDGET_S:
            skipped.append((t.isoformat(), "job time budget spent"))
            continue
        try:
            urls = {k: nearest(sess, p, t, MATCH_MIN) for k, p in ARCHIVE.items()}
            gone = [k for k in mc.ESSENTIAL if urls.get(k) is None]
            if urls.get("cg") is None:
                gone.append("cg")
            if gone:
                skipped.append((t.isoformat(), f"missing {gone}"))
                continue
            futs = {k: pool.submit(fetch, sess, u, mc.PRODUCTS[k][1]) for k, u in urls.items() if u}
            wfut = pool.submit(anvil_wind, sess, t)
            su = nearest(sess, STEINER_DIR, t, MATCH_MIN)
            sfut = pool.submit(fetch, sess, su, -999.0) if su else None
            F, la, lo = {}, None, None
            for k, f in futs.items():
                fld, fla, flo, _ = f.result()
                F[k] = fld
                if la is None:
                    la, lo = fla, flo
            for k in list(mc.PRODUCTS):
                if k not in F:
                    F[k] = np.full((la.size, lo.size), mc.PRODUCTS[k][1], np.float32)
            if z0 is None or (t - z0_t).total_seconds() >= FREEZING_EVERY_H * 3600:
                zu = nearest(sess, FREEZING_DIR, t, 45)
                if zu:
                    z0, _, _, _ = fetch(sess, zu, None)
                    z0_t = t
            zz = z0 if z0 is not None else np.full((la.size, lo.size), 4800.0, np.float32)
            sep = None
            if sfut is not None:
                lvl, _, _, _ = sfut.result()
                sep = mc.separation_lite(lvl, STEINER_KM, la, lo)
            wind = wfut.result()
            history = None
            if prev_att is not None and (t - prev_t) <= datetime.timedelta(hours=1, minutes=5):
                from scipy.ndimage import shift as _shift
                dlat_km, dlon_km = mc._spacing(la, lo)
                hrs = (t - prev_t).total_seconds() / 3600.0
                if wind is not None:          # moved downwind, grown by the wind's uncertainty
                    moved = _shift(prev_att.astype(np.float32),
                                   (-wind["v_kmh"] * hrs / dlat_km, wind["u_kmh"] * hrs / dlon_km),
                                   order=0, cval=0.0) > 0.5
                    r_nm = wind["unc_kmh"] * hrs / 1.852
                else:                         # no winds: the all-directions allowance
                    moved = prev_att
                    r_nm = mc.DRIFT_KMH * hrs / 1.852
                history = mc.maximum_filter(moved.astype(np.uint8),
                                            footprint=mc._disc(max(r_nm, 0.5), dlat_km, dlon_km)) > 0
            try:
                sat = sat_tops_bt(sess, t, la, lo, float(np.nanmean(zz)))
            except Exception as e:
                sat = None
                logging.warning(f"{t:%Y-%m-%d %HZ} satellite: {type(e).__name__}: {e}")
            status, red, yel, cls, top, diag = mc.evaluate(F, zz, la, lo, history, None, sep, None, sat)
            prev_att, prev_t = (cls == 6), t

            from scipy.ndimage import distance_transform_edt as _edt
            dlat_km, dlon_km = mc._spacing(la, lo)
            ji = [(int(np.argmin(np.abs(la - mc.SITES[n_][0]))), int(np.argmin(np.abs(lo - mc.SITES[n_][1]))))
                  for n_ in pads]
            fv = np.full((len(pads), len(FEAT_NAMES)), 255, np.uint8)
            for q, key in enumerate(FEAT_DIST):
                msk = diag["feat"][key]
                if msk.any():
                    dist = _edt(~msk, sampling=(dlat_km, dlon_km)) / 1.852
                    for p_, (j_, i_) in enumerate(ji):
                        v_ = dist[j_, i_]
                        fv[p_, q] = 255 if v_ > 25.4 else int(np.ceil(v_ * 10 - 1e-6))
            mr = diag["feat"]["mrr1"]
            for p_, (j_, i_) in enumerate(ji):
                fv[p_, -1] = int(np.clip(np.round(max(float(mr[j_, i_]), 0.0) * 2), 0, 254))
            rec["feat"].append(fv)

            st, rb, yb = [], [], []
            for name in pads:
                plat, plon = mc.SITES[name]
                j = int(np.argmin(np.abs(la - plat))); i = int(np.argmin(np.abs(lo - plon)))
                st.append(int(status[j, i]))
                rb.append(sum(1 << b for b, k in enumerate(mc.RULE_KEYS) if red[k][j, i]))
                yb.append(sum(1 << b for b, k in enumerate(mc.RULE_KEYS) if yel[k][j, i] and not red[k][j, i]))
            rec["sat"].append(sat is not None); rec["wind"].append(wind is not None)
            rec["satup"].append(int(diag["sat_up"].sum()))
            rec["t"].append(t.isoformat()); rec["status"].append(st)
            rec["red"].append(rb); rec["yel"].append(yb)

            if acc is None:
                shape = (24,) + status.shape
                acc = {"n": np.zeros(24, np.int32), "red": np.zeros(shape, np.uint16),
                       "watch": np.zeros(shape, np.uint16), "shape": status.shape}
            lh = t.replace(tzinfo=datetime.timezone.utc).astimezone(LOCAL).hour
            acc["n"][lh] += 1
            acc["red"][lh] += (status == 2)
            acc["watch"][lh] += (status >= 1)
        except Exception as e:
            skipped.append((t.isoformat(), f"{type(e).__name__}: {e}"))
            logging.warning(f"{t:%Y-%m-%d %HZ}: {type(e).__name__}: {e}")
    pool.shutdown()
    return {"rec": rec, "acc": acc, "skipped": skipped}


def month_hours(ym, days=None):
    y, m = map(int, ym.split("-"))
    first = datetime.datetime(y, m, 1)
    n = calendar.monthrange(y, m)[1] if days is None else min(days, calendar.monthrange(y, m)[1])
    start = max(first, datetime.datetime(2020, 10, 14))       # v12 archive begins here
    end = first + datetime.timedelta(days=n)
    out, t = [], start
    while t < end:
        out.append(t)
        t += datetime.timedelta(hours=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("month", help="YYYY-MM")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--procs", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--out", default="out")
    a, _ = ap.parse_known_args()
    hours = month_hours(a.month, a.days)
    if not hours:
        logging.info("no hours in range"); return
    # contiguous chunks so the detached-anvil history still runs in order within each
    k = max(1, min(a.procs, len(hours)))
    chunks = [list(c) for c in np.array_split(np.array(hours, dtype=object), k) if len(c)]
    t0 = time.monotonic()
    if len(chunks) == 1:
        parts = [score_hours(chunks[0])]
    else:
        with mp.get_context("spawn").Pool(len(chunks)) as pool:
            parts = pool.map(score_hours, chunks)
    rec = {"t": [], "status": [], "red": [], "yel": [], "sat": [], "wind": [], "satup": [], "feat": []}
    acc, skipped = None, []
    for p in parts:
        for key in rec:
            rec[key] += p["rec"][key]
        skipped += p["skipped"]
        if p["acc"] is not None:
            if acc is None:
                acc = {k: v.copy() if hasattr(v, "copy") else v for k, v in p["acc"].items()}
            else:
                for key in ("n", "red", "watch"):
                    acc[key] += p["acc"][key]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, a.month.replace("-", "") + ".npz")
    np.savez_compressed(
        path,
        t=np.array(rec["t"]), status=np.array(rec["status"], np.int8).reshape(-1, len(mc.SITES)),
        red=np.array(rec["red"], np.uint8).reshape(-1, len(mc.SITES)),
        yel=np.array(rec["yel"], np.uint8).reshape(-1, len(mc.SITES)),
        pads=np.array(list(mc.SITES)), rule_keys=np.array(mc.RULE_KEYS),
        map_n=(acc["n"] if acc else np.zeros(24, np.int32)),
        map_red=(acc["red"] if acc else np.zeros((24, 1, 1), np.uint16)),
        map_watch=(acc["watch"] if acc else np.zeros((24, 1, 1), np.uint16)),
        skipped=np.array([f"{a_} {b_}" for a_, b_ in skipped]),
        sat=np.array(rec["sat"], bool), wind=np.array(rec["wind"], bool),
        satup=np.array(rec["satup"], np.int32),
        feat=(np.stack(rec["feat"]) if rec["feat"] else np.zeros((0, len(mc.SITES), len(FEAT_NAMES)), np.uint8)),
        feat_names=np.array(FEAT_NAMES),
        classifier=np.array(mc.VIEWER_VERSION_EXPECTED), hours_planned=np.array(len(hours)))
    logging.info(f"{a.month}: {len(rec['t'])} of {len(hours)} hours scored, {len(skipped)} skipped, "
                 f"{time.monotonic() - t0:.0f} s -> {path}")


if __name__ == "__main__":
    main()

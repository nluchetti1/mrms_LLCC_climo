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
and decayed-anvil logic, NLDN CG for lightning. What DIFFERS, by necessity:
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
FREEZING_DIR = "Model_0degC_Height_00.50"
FREEZING_EVERY_H = 6
MATCH_MIN = 3.0                                 # a file must lie within this of the sample time
JOB_BUDGET_S = 5.5 * 3600                       # stop cleanly before GitHub's 6 h job limit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(processName)s %(message)s")
_listings = {}


def _session():
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(max_retries=4, pool_maxsize=16))
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


def fetch(sess, url, fill):
    return mc.decode_crop(mc.fetch_url(sess, url), fill)


def score_hours(hours):
    """Score a contiguous run of hourly sample times, in order. Returns a dict of arrays."""
    sess = _session()
    pads = list(mc.SITES)
    rec = {"t": [], "status": [], "red": [], "yel": []}
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
            history = None
            if prev_att is not None and (t - prev_t) <= datetime.timedelta(hours=1, minutes=5):
                dlat_km, dlon_km = mc._spacing(la, lo)
                r_nm = mc.DRIFT_KMH * (t - prev_t).total_seconds() / 3600.0 / 1.852
                history = mc.maximum_filter(prev_att.astype(np.uint8),
                                            footprint=mc._disc(r_nm, dlat_km, dlon_km)) > 0
            status, red, yel, cls, top, diag = mc.evaluate(F, zz, la, lo, history, None, sep)
            prev_att, prev_t = (cls == 6), t

            st, rb, yb = [], [], []
            for name in pads:
                plat, plon = mc.SITES[name]
                j = int(np.argmin(np.abs(la - plat))); i = int(np.argmin(np.abs(lo - plon)))
                st.append(int(status[j, i]))
                rb.append(sum(1 << b for b, k in enumerate(mc.RULE_KEYS) if red[k][j, i]))
                yb.append(sum(1 << b for b, k in enumerate(mc.RULE_KEYS) if yel[k][j, i] and not red[k][j, i]))
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
    rec = {"t": [], "status": [], "red": [], "yel": []}
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
        classifier=np.array(mc.VIEWER_VERSION_EXPECTED), hours_planned=np.array(len(hours)))
    logging.info(f"{a.month}: {len(rec['t'])} of {len(hours)} hours scored, {len(skipped)} skipped, "
                 f"{time.monotonic() - t0:.0f} s -> {path}")


if __name__ == "__main__":
    main()

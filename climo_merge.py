#!/usr/bin/env python3
"""
Merge the month files from climo_run.py into the published climatology.

    python climo_merge.py --inp out --site site

For each pad, by day of year and LOCAL hour (America/New_York, so DST is handled):
    violating %        share of samples that were red
    watch-or-worse %   share that were yellow or red
    samples            how many hourly samples back the figure - about 6 years x 15 days = 90
    rule shares        of the violating samples, the share in which each rule fired

Day of year is smoothed with a +/-7 day window (circular across New Year), so each figure rests on
~90 samples and carries roughly +/-5 points of sampling noise at 50%. Feb 29 is folded into Feb 28.

Maps: violating % per grid cell for each calendar month and local hour.

Files written to the site folder:
    climo.json        metadata - period, classifier version, pads, rules, coverage
    climo_pads.bin    per pad, uint8 planes of 365 x 24: violating, watch, samples, then one per rule
    climo_maps.bin    gzip of uint8 [12 months][24 hours][ny][nx] violating %, 255 = no samples
    basemap.png       coastline and pads on the same Mercator extent as the live page
"""

import argparse
import datetime
import glob
import gzip
import json
import os
import shutil
from zoneinfo import ZoneInfo

import numpy as np

LOCAL = ZoneInfo("America/New_York")
WINDOW_D = 7


def doy_index(d):
    """0..364 on a non-leap calendar; Feb 29 folds into Feb 28."""
    if d.month == 2 and d.day == 29:
        return 58
    return datetime.date(2001, d.month, d.day).timetuple().tm_yday - 1


def circular_window_sum(a):
    """Sum over +/-WINDOW_D days along axis 0, wrapping across New Year."""
    out = np.zeros_like(a, dtype=np.float64)
    for s in range(-WINDOW_D, WINDOW_D + 1):
        out += np.roll(a, s, axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="out")
    ap.add_argument("--site", default="site")
    ap.add_argument("--web", default="web")
    a, _ = ap.parse_known_args()
    files = sorted(glob.glob(os.path.join(a.inp, "*.npz")))
    if not files:
        raise SystemExit("no month files found")
    os.makedirs(a.site, exist_ok=True)

    pads = rule_keys = None
    n_pad = red_pad = watch_pad = rules_pad = None
    map_n = np.zeros((12, 24), np.int64)
    map_red = None
    coverage, classifiers, years = {}, set(), set()
    first = last = None
    for f in files:
        z = np.load(f, allow_pickle=False)
        if pads is None:
            pads = [str(p) for p in z["pads"]]
            rule_keys = [str(r) for r in z["rule_keys"]]
            P, R = len(pads), len(rule_keys)
            n_pad = np.zeros((P, 365, 24), np.int64)
            red_pad = np.zeros_like(n_pad); watch_pad = np.zeros_like(n_pad)
            rules_pad = np.zeros((P, R, 365, 24), np.int64)
        classifiers.add(str(z["classifier"]))
        ym = os.path.basename(f)[:6]
        coverage[ym] = {"scored": int(len(z["t"])), "planned": int(z["hours_planned"]),
                        "skipped": int(len(z["skipped"]))}
        if "sat" in z.files:
            coverage[ym]["sat"] = int(z["sat"].sum()); coverage[ym]["wind"] = int(z["wind"].sum())
            coverage[ym]["satup"] = int((z["satup"] > 0).sum())
        st, rb = z["status"], z["red"]
        for k, ts in enumerate(z["t"]):
            tu = datetime.datetime.fromisoformat(str(ts)).replace(tzinfo=datetime.timezone.utc)
            tl = tu.astimezone(LOCAL)
            d, h = doy_index(tl.date()), tl.hour
            years.add(tl.year)
            first = tu if first is None or tu < first else first
            last = tu if last is None or tu > last else last
            for p in range(len(pads)):
                n_pad[p, d, h] += 1
                if st[k, p] == 2:
                    red_pad[p, d, h] += 1
                    for b in range(len(rule_keys)):
                        if rb[k, p] & (1 << b):
                            rules_pad[p, b, d, h] += 1
                if st[k, p] >= 1:
                    watch_pad[p, d, h] += 1
        if z["map_red"].ndim == 3 and z["map_red"].shape[1] > 1:
            mo = int(ym[4:6]) - 1
            if map_red is None:
                map_red = np.zeros((12, 24) + z["map_red"].shape[1:], np.int64)
            map_n[mo] += z["map_n"]
            map_red[mo] += z["map_red"]

    # ---- per-pad planes, smoothed over +/-7 days -----------------------------------------------
    planes = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for p in range(len(pads)):
            n = circular_window_sum(n_pad[p])
            red = circular_window_sum(red_pad[p])
            watch = circular_window_sum(watch_pad[p])
            pv = np.where(n > 0, np.round(100 * red / n), 255).astype(np.uint8)
            pw = np.where(n > 0, np.round(100 * watch / n), 255).astype(np.uint8)
            pn = np.clip(n, 0, 254).astype(np.uint8)
            planes += [pv, pw, pn]
            for b in range(len(rule_keys)):
                rr = circular_window_sum(rules_pad[p, b])
                planes.append(np.where(red > 0, np.round(100 * rr / red), 0).astype(np.uint8))
    with open(os.path.join(a.site, "climo_pads.bin"), "wb") as fp:
        fp.write(b"".join(x.tobytes() for x in planes))

    # ---- monthly maps ---------------------------------------------------------------------------
    ny = nx = 0
    if map_red is not None:
        ny, nx = map_red.shape[2:]
        with np.errstate(invalid="ignore", divide="ignore"):
            pm = np.where(map_n[:, :, None, None] > 0,
                          np.round(100 * map_red / np.maximum(map_n[:, :, None, None], 1)), 255
                          ).astype(np.uint8)
        with open(os.path.join(a.site, "climo_maps.bin"), "wb") as fp:
            fp.write(gzip.compress(pm.tobytes(), compresslevel=6))

    # ---- basemap on the live page's projection ---------------------------------------------------
    try:
        import mrms_classify as mc
        la = mc.DOMAIN["lat_max"] - (np.arange(ny) + 0.5) * (mc.DOMAIN["lat_max"] - mc.DOMAIN["lat_min"]) / max(ny, 1)
        lo = mc.DOMAIN["lon_min"] + (np.arange(nx) + 0.5) * (mc.DOMAIN["lon_max"] - mc.DOMAIN["lon_min"]) / max(nx, 1)
        fig, ax, pc, _, _ = mc._figure(la, lo)
        mc._chrome(ax, pc)
        mc._save(fig, os.path.join(a.site, "basemap.png"))
        domain = mc.DOMAIN
        rules = {k: mc.RULE_NAMES.get(k, k) for k in rule_keys}
        sites = {k: list(v) for k, v in mc.SITES.items()}
    except Exception as e:
        print(f"basemap skipped: {e}")
        domain, rules, sites = None, {k: k for k in rule_keys}, {}

    meta = {
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "period": [first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")],
        "years": sorted(years), "classifier": sorted(classifiers),
        "pads": pads, "rule_keys": rule_keys, "rules": rules, "sites": sites,
        "window_days": WINDOW_D, "timezone": "America/New_York",
        "planes_per_pad": 3 + len(rule_keys), "days": 365, "hours": 24,
        "map": {"ny": int(ny), "nx": int(nx), "months": 12, "hours": 24,
                "samples": map_n.tolist()},
        "domain": domain, "coverage": coverage,
        "notes": ["Lightning is NLDN cloud-to-ground only - intracloud-only lightning is missed, "
                  "so the lightning rule is undercounted.",
                  "Isotherm heights come from the MRMS model freezing level and an assumed lapse "
                  "rate; historical BUFKIT soundings are not available.",
                  "Anvil exception part (a) uses the 0 C slice, not measured echo bases.",
                  "Detached-anvil origin uses the previous hour, not the previous 5-minute frames.",
                  "Satellite cloud tops (GOES band 13) can raise a cumulus class over radar echo - "
                  "never add cloud, never lower a class. Their HEIGHT is derived from the band-13 "
                  "temperature on the MRMS freezing level and a 6.5 C/km lapse rate for all years, "
                  "because the 2 km height product only begins in March 2023; one method keeps the "
                  "record consistent.",
                  "Detached-anvil drift is steered by GOES band-14 anvil-level winds where available "
                  "(band 8, then 32 kt in every direction, as fallbacks).",
                  "GOES-16 is used before 7 Apr 2025 and GOES-19 from then on.",
                  "One fixed classifier version across all years; MRMS itself was updated "
                  "several times over the period."],
    }
    with open(os.path.join(a.site, "climo.json"), "w") as fp:
        json.dump(meta, fp)
    if os.path.exists(os.path.join(a.web, "index.html")):
        shutil.copy(os.path.join(a.web, "index.html"), os.path.join(a.site, "index.html"))
    open(os.path.join(a.site, ".nojekyll"), "w").close()
    tot = sum(c["scored"] for c in coverage.values())
    print(f"merged {len(files)} months, {tot} samples, {first:%Y-%m-%d} .. {last:%Y-%m-%d}; "
          f"classifier {sorted(classifiers)}")


if __name__ == "__main__":
    main()

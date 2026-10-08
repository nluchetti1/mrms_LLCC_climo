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


# The standard settings. The page's "what if" panel starts from these and re-scores from features.
STANDARD = {"ltg_nm": 10.0, "ltg_watch_nm": 20.0, "cu10_nm": 5.0, "cu20_nm": 10.0,
            "att_nm": 3.0, "det_nm": 3.0, "att_ltg_nm": 10.0, "excep_nm": 5.0, "mrr_dbz": 7.5,
            "thick_mrr_dbz": 7.5, "margin_nm": 2.0, "fm_vm": 1000.0, "fm_nm": 5.0,
            "anvil_exc": True, "thick_exc": True}
FEAT_DIST = ["cuthru", "cu10", "cu20", "att", "det", "warm", "thick", "ltg"]


def score_features(f, millv, mill_d, P_, rule_keys):
    """(status, red bits) for one pad-sample from its features, under settings P_ - the same logic
    as evaluate(), applied to distances. f: uint8[9]; millv: uint8 per mill (50 V/m units, 255 none)
    or None; mill_d: nmi from this pad to each mill."""
    E = 1e-6
    d = {k: (np.inf if f[i] == 255 else f[i] / 10.0) for i, k in enumerate(FEAT_DIST)}
    mrr = f[8] / 2.0
    lit = lambda r: d["ltg"] <= r + E
    exc = P_["anvil_exc"] and not (d["warm"] <= P_["excep_nm"] + E) and mrr < P_["mrr_dbz"]
    texc = P_["thick_exc"] and mrr < P_["thick_mrr_dbz"]
    def rules(m):
        r = {"lightning": lit(P_["ltg_nm"] + m),
             "cumulus_through": d["cuthru"] <= m + E,
             "cumulus_5nm": d["cu10"] <= P_["cu10_nm"] + m + E,
             "cumulus_10nm": d["cu20"] <= P_["cu20_nm"] + m + E,
             "attached_anvil": ((d["att"] <= P_["att_nm"] + m + E) and not exc)
                               or ((d["att"] <= P_["att_ltg_nm"] + m + E) and lit(P_["ltg_nm"] + m)),
             "detached_anvil": (d["det"] <= P_["det_nm"] + m + E) and not exc,
             "disturbed": False,
             "thick_layer": (d["thick"] <= m + E) and not texc}
        if "field_mill" in rule_keys:
            r["field_mill"] = False
            if millv is not None:
                thr = int(P_["fm_vm"] // 50)
                ok = (millv != 255) & (mill_d <= P_["fm_nm"] + m + E)
                r["field_mill"] = bool(((millv >= thr) & ok).any())
        return r
    red = rules(0.0)
    bits = sum(1 << b for b, k in enumerate(rule_keys) if red.get(k))
    if bits:
        return 2, bits
    yel = rules(P_["margin_nm"])
    if any(yel.get(k) for k in rule_keys) or lit(P_["ltg_watch_nm"]):
        return 1, 0
    return 0, 0


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
    ap.add_argument("--fieldmill", default="fieldmill/fieldmill_hourly.npz")
    a, _ = ap.parse_known_args()
    # ---- field mills (LLCC 4.1.2), applied here on top of the radar result ------------------------
    # Any one-minute |E| >= 1000 V/m from a mill within 5 nmi of the pad in the 15 minutes before
    # the sample -> violating; within 7 nmi -> watch (the usual 2 nmi margin). The prep script
    # reduced the minute CSVs to per-hour 15-minute maxima. Field-mill EXCEPTIONS in other rules are
    # not modelled; they only ever relieve a violation, so leaving them out is the conservative side.
    # When mills are present, only samples WITH mill data count, for every rule - otherwise hours
    # outside the mill record would look artificially clear and put a step in the statistics.
    FM = None
    if a.fieldmill and os.path.exists(a.fieldmill):
        z = np.load(a.fieldmill, allow_pickle=False)
        FM = {"hours": z["hours_utc"].astype(np.int64), "max15": z["max15"], "n15": z["n15"],
              "pex": z["pad_exceed60"], "pva": z["pad_valid60"], "pads": [str(p) for p in z["pads"]],
              "basis": str(z["time_basis"]),
              "lat": z["mill_lat"], "lon": z["mill_lon"],
              "plat": z["pad_lat"], "plon": z["pad_lon"]}
        def _nm(la1, lo1, la2, lo2):
            k = np.cos(np.radians((la1 + la2) / 2))
            return np.hypot((la2 - la1) * 111.32, (lo2 - lo1) * 111.32 * k) / 1.852
        FM["near5"] = {}; FM["near7"] = {}
        for i, p in enumerate(FM["pads"]):
            d = _nm(FM["plat"][i], FM["plon"][i], FM["lat"], FM["lon"])
            FM["near5"][p] = np.where(d <= 5.0)[0]; FM["near7"][p] = np.where(d <= 7.0)[0]
        print(f"field mills: {len(FM['lat'])} mills, {len(FM['hours'])} hours; time basis: {FM['basis']}")
    fm_ex = {}; fm_va = {}; fm_samples = 0
    FX = {"doy": [], "hour": [], "month": [], "day": [], "use": [], "feat": [], "mill": [],
          "st": [], "rb": []}
    have_feat = True
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
            radar_R = len(rule_keys)
            if FM is not None:
                rule_keys.append("field_mill")
            P, R = len(pads), len(rule_keys)
            n_pad = np.zeros((P, 365, 24), np.int64)
            red_pad = np.zeros_like(n_pad); watch_pad = np.zeros_like(n_pad)
            rules_pad = np.zeros((P, R, 365, 24), np.int64)
            # for the headline: by calendar month and local hour, rule and sole-rule counts,
            # days with any violation, and how often ANY pad is violating
            mh_n = np.zeros((P, 12, 24), np.int64); mh_red = np.zeros_like(mh_n)
            mh_watch = np.zeros_like(mh_n)
            rule_cnt = np.zeros((P, R), np.int64); sole_cnt = np.zeros((P, R), np.int64)
            red_days = [set() for _ in range(P)]; all_days = set()
            any_red = any_n = 0
        classifiers.add(str(z["classifier"]))
        ym = os.path.basename(f)[:6]
        coverage[ym] = {"scored": int(len(z["t"])), "planned": int(z["hours_planned"]),
                        "skipped": int(len(z["skipped"]))}
        if "sat" in z.files:
            coverage[ym]["sat"] = int(z["sat"].sum()); coverage[ym]["wind"] = int(z["wind"].sum())
            coverage[ym]["satup"] = int((z["satup"] > 0).sum())
        st, rb = z["status"], z["red"]
        zf = z["feat"] if "feat" in z.files else None
        if zf is None or len(zf) != len(z["t"]):
            have_feat = False
        for k, ts in enumerate(z["t"]):
            tu = datetime.datetime.fromisoformat(str(ts)).replace(tzinfo=datetime.timezone.utc)
            tl = tu.astimezone(LOCAL)
            d, h = doy_index(tl.date()), tl.hour
            use = [True] * len(pads)
            mill_red = [False] * len(pads); mill_watch = [False] * len(pads)
            if FM is not None:
                hi = int(np.searchsorted(FM["hours"], int(tu.timestamp())))
                have_hour = hi < len(FM["hours"]) and FM["hours"][hi] == int(tu.timestamp())
                for p, name in enumerate(pads):
                    n5 = FM["near5"].get(name, [])
                    if not have_hour or len(n5) == 0:
                        use[p] = False; continue
                    ok5 = FM["n15"][hi, n5] > 0
                    if not ok5.any():
                        use[p] = False; continue
                    mill_red[p] = bool(((FM["max15"][hi, n5] >= 1000) & ok5).any())
                    n7 = FM["near7"][name]
                    mill_watch[p] = bool(((FM["max15"][hi, n7] >= 1000) & (FM["n15"][hi, n7] > 0)).any())
                    pi = FM["pads"].index(name)
                    fm_ex[name] = fm_ex.get(name, 0) + int(FM["pex"][hi, pi])
                    fm_va[name] = fm_va.get(name, 0) + int(FM["pva"][hi, pi])
                if not any(use):
                    continue
                fm_samples += 1
            years.add(tl.year)
            first = tu if first is None or tu < first else first
            last = tu if last is None or tu > last else last
            # the field-mill verdict folded into this sample's status and rule bits
            stk = [int(st[k, p]) for p in range(len(pads))]
            rbk = [int(rb[k, p]) for p in range(len(pads))]
            for p in range(len(pads)):
                if mill_red[p]:
                    stk[p] = 2; rbk[p] |= 1 << radar_R
                elif mill_watch[p] and stk[p] == 0:
                    stk[p] = 1
            if have_feat:
                FX["doy"].append(d); FX["hour"].append(h); FX["month"].append(tl.month)
                FX["day"].append(tl.date().toordinal()); FX["use"].append([1 if u_ else 0 for u_ in use])
                FX["feat"].append(zf[k]); FX["st"].append(list(stk)); FX["rb"].append(list(rbk))
                if FM is not None:
                    mv = FM["max15"][hi].astype(np.int64); nv = FM["n15"][hi]
                    FX["mill"].append(np.where((nv > 0) & (mv < 65535), np.minimum(mv // 50, 254), 255)
                                      .astype(np.uint8))
            all_days.add(tl.date()); any_n += 1
            if any(stk[p] == 2 for p in range(len(pads)) if use[p]):
                any_red += 1
            mo_ = tl.month - 1
            for p in range(len(pads)):
                if not use[p]:
                    continue
                mh_n[p, mo_, h] += 1
                if stk[p] == 2:
                    mh_red[p, mo_, h] += 1
                    red_days[p].add(tl.date())
                    bits = [b for b in range(len(rule_keys)) if rbk[p] & (1 << b)]
                    for b in bits:
                        rule_cnt[p, b] += 1
                    if len(bits) == 1:
                        sole_cnt[p, bits[0]] += 1
                if stk[p] >= 1:
                    mh_watch[p, mo_, h] += 1
                n_pad[p, d, h] += 1
                if stk[p] == 2:
                    red_pad[p, d, h] += 1
                    for b in range(len(rule_keys)):
                        if rbk[p] & (1 << b):
                            rules_pad[p, b, d, h] += 1
                if stk[p] >= 1:
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

    # ---- the headline: the numbers worth putting in front of leadership --------------------------
    MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    def pct(a, b):
        return round(100.0 * a / b, 1) if b else None
    headline = {"pads": {}, "any_pad_violating_pct": pct(any_red, any_n),
                "all_clear_pct": None, "days": len(all_days)}
    for p, name in enumerate(pads):
        n, red, watch = mh_n[p].sum(), mh_red[p].sum(), mh_watch[p].sum()
        if not n:
            continue
        # worst month and hour, ignoring thin bins
        with np.errstate(invalid="ignore", divide="ignore"):
            P_mh = np.where(mh_n[p] >= 20, mh_red[p] / np.maximum(mh_n[p], 1), np.nan)
        worst = np.unravel_index(np.nanargmax(P_mh), P_mh.shape) if np.isfinite(P_mh).any() else None
        # best daytime hour in summer (Jun-Sep, 07-19 local), and summer against winter
        su, wi = [5, 6, 7, 8], [10, 11, 0, 1, 2]
        su_n, su_red = mh_n[p][su].sum(0), mh_red[p][su].sum(0)
        day = list(range(7, 20))
        best = None
        if su_n[day].sum():
            with np.errstate(invalid="ignore", divide="ignore"):
                ph = np.where(su_n >= 20, su_red / np.maximum(su_n, 1), np.nan)
            cand = [(ph[hh], hh) for hh in day if np.isfinite(ph[hh])]
            if cand:
                v, hh = min(cand)
                best = {"hour": int(hh), "pct": round(100 * float(v), 1)}
        rules_sorted = sorted(((rule_keys[b], pct(rule_cnt[p, b], red), pct(sole_cnt[p, b], red))
                               for b in range(len(rule_keys)) if rule_cnt[p, b]),
                              key=lambda x: -x[1])
        headline["pads"][name] = {
            "violating_pct": pct(red, n), "watch_pct": pct(watch, n),
            "hours_per_year": round(float(red) / n * 8766),
            "samples": int(n),
            "rules": [{"key": k_, "share": a_, "sole": b_} for k_, a_, b_ in rules_sorted],
            "worst": ({"month": MON[worst[0]], "hour": int(worst[1]),
                       "pct": round(100 * float(P_mh[worst]), 1)} if worst is not None else None),
            "best_summer_daytime": best,
            "summer_pct": pct(mh_red[p][su].sum(), mh_n[p][su].sum()),
            "winter_pct": pct(mh_red[p][wi].sum(), mh_n[p][wi].sum()),
            "days_with_violation_pct": pct(len(red_days[p]), len(all_days)),
            "mill_minutes_pct": (round(100.0 * fm_ex[name] / fm_va[name], 2)
                                 if FM is not None and fm_va.get(name) else None),
        }

    if FM is not None:
        headline["field_mills"] = {"samples": fm_samples, "time_basis": FM["basis"]}

    # ---- "what if" features: the page re-scores from these as the settings change ------------------
    features = None
    if have_feat and FX["feat"]:
        S = len(FX["feat"])
        feat = np.stack(FX["feat"]).astype(np.uint8)                   # [S, P, 9]
        use_ = np.array(FX["use"], np.uint8)                           # [S, P]
        mills = np.stack(FX["mill"]) if FX["mill"] else np.zeros((S, 0), np.uint8)
        day0 = min(FX["day"])
        parts_ = [np.array(FX["doy"], np.uint16), np.array(FX["hour"], np.uint8),
                  np.array(FX["month"], np.uint8), (np.array(FX["day"]) - day0).astype(np.uint16),
                  use_, feat, mills]
        blob = b"".join(p_.tobytes() for p_ in parts_)
        with open(os.path.join(a.site, "climo_features.bin"), "wb") as fp:
            fp.write(gzip.compress(blob, compresslevel=6))
        mill_d = {}
        if FM is not None:
            for name in pads:
                i_ = FM["pads"].index(name)
                k_ = np.cos(np.radians((FM["plat"][i_] + FM["lat"]) / 2))
                mill_d[name] = np.hypot((FM["lat"] - FM["plat"][i_]) * 111.32,
                                        (FM["lon"] - FM["plon"][i_]) * 111.32 * k_) / 1.852
        # the check: at standard settings, re-scoring from features must reproduce every verdict
        agree = total = 0
        st_ = np.array(FX["st"]); rb_ = np.array(FX["rb"])
        for i_ in range(S):
            for p_, name in enumerate(pads):
                if not use_[i_, p_]:
                    continue
                s2, b2 = score_features(feat[i_, p_], mills[i_] if mills.shape[1] else None,
                                        mill_d.get(name), STANDARD, rule_keys)
                total += 1
                agree += int(s2 == st_[i_, p_] and b2 == rb_[i_, p_])
        rate = 100.0 * agree / max(total, 1)
        print(f"what-if features: {S} samples; re-scored at standard settings, {agree}/{total} "
              f"pad-samples match the classifier ({rate:.3f}%)")
        features = {"samples": S, "pads": len(pads), "k": int(feat.shape[2]), "mills": int(mills.shape[1]),
                    "names": FEAT_DIST + ["mrr1"], "day0": int(day0), "standard": STANDARD,
                    "agreement_pct": round(rate, 3),
                    "mill_lat": (FM["lat"].tolist() if FM is not None else []),
                    "mill_lon": (FM["lon"].tolist() if FM is not None else [])}
    meta = {
        "features": features,
        "headline": headline,
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
                 ] + ([
                  "Field mills (LLCC 4.1.2): violating if any one-minute |E| >= 1000 V/m from a mill "
                  "within 5 nmi of the pad in the 15 minutes before; watch within 7 nmi. Field-mill "
                  "exceptions in other rules are not modelled - they only relieve a violation, so "
                  "this is the conservative side.",
                  "With field mills included, only hours with mill data count, for every rule, so the "
                  "period is the field-mill record's."] if FM is not None else []) + [
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

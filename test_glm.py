"""GLM in the climatology: the real climo_run.main() + climo_merge.py over 12 synthetic hours, with
real GLM-format netCDF files served from a fake S3 listing. Runs in either climatology repo.
    python test_glm.py"""
import datetime as dt, io, json, os, shutil, sys, tempfile
from contextlib import redirect_stdout
import numpy as np, netCDF4
sys.path.insert(0, os.getcwd())
import climo_run as CR, mrms_classify as mc, climo_merge as CM
try:
    import kmlb_classify as K
except ImportError:
    K = None
FAILS = []
def check(c, msg):
    print(("  ok    " if c else "  FAIL  ") + msg)
    if not c: FAILS.append(msg)
LA = np.round(29.595 - 0.01 * np.arange(200), 3); LO = np.round(-81.595 + 0.01 * np.arange(200), 3)
LC39A = (28.6084, -80.6043)
FLASH = (LC39A[0] + 3 / 60.0, LC39A[1])            # 3 nmi north of LC-39A
DAY = dt.datetime(2025, 7, 31)

def nc_bytes(start, n):
    ds = netCDF4.Dataset("glm.nc", "w", memory=4096, format="NETCDF4")
    ds.time_coverage_start = start.strftime("%Y-%m-%dT%H:%M:%S.0Z")
    ds.createDimension("number_of_flashes", None)
    v = {k: ds.createVariable(k, t, ("number_of_flashes",)) for k, t in
         (("flash_lat", "f4"), ("flash_lon", "f4"), ("flash_time_offset_of_first_event", "f4"), ("flash_quality_flag", "i2"))}
    if n:
        v["flash_lat"][:] = np.full(n, FLASH[0]); v["flash_lon"][:] = np.full(n, FLASH[1])
        v["flash_time_offset_of_first_event"][:] = np.linspace(1, 15, n); v["flash_quality_flag"][:] = np.zeros(n)
    return bytes(ds.close())

def keys_for(hr):
    out = []
    for k in range(180):
        st = hr + dt.timedelta(seconds=20 * k)
        if hr.hour == 17 and st.minute >= 30 and k % 2:          # 17:30-18:00: half the files missing
            continue
        out.append(f"GLM-L2-LCFA/{st:%Y}/{st.timetuple().tm_yday:03d}/{st:%H}/OR_GLM-L2-LCFA_G19_s{st:%Y}{st.timetuple().tm_yday:03d}{st:%H%M%S}0_e_c.nc")
    return out

# netCDF files built up front in the main thread: netCDF4/HDF5 must not run on several threads
FILES = {}
def prebuild():
    for h in range(9, 22):
        hr = DAY.replace(hour=h)
        for k in keys_for(hr):
            st = dt.datetime.strptime(k.split("_s")[1][:13], "%Y%j%H%M%S")
            FILES[k] = nc_bytes(st, 3 if dt.datetime(2025, 7, 31, 14, 30) <= st < dt.datetime(2025, 7, 31, 15) else 0)

class Sess:
    def get(self, url, timeout=None):
        class R:
            def __init__(s, content=b"", text=""): s.content, s.text = content, text
            def raise_for_status(s): pass
        if "list-type=2" in url and "GLM-L2-LCFA" in url:
            prefix = url.split("prefix=")[1].split("&")[0]
            y, j, h = prefix.split("/")[1:4]
            hr = dt.datetime(int(y), 1, 1, int(h)) + dt.timedelta(days=int(j) - 1)
            return R(text="<ListBucketResult>" + "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys_for(hr)) + "</ListBucketResult>")
        if "GLM-L2-LCFA" in url:
            return R(content=FILES[url.split("/", 3)[3]])
        return R(content=b"x")

LOg, LAg = np.meshgrid(LO, LA)
ST = np.hypot((LAg - 29.3) * 111.2, (LOg + 81.3) * 97.7) < 10       # weak echo far from the pads
def nearest(sess, product, t, tol): return f"fake://{product}/{t:%Y%m%d-%H%M%S}"
def fetch(sess, url, fill):
    prod = url.split("/")[2]
    if prod.startswith("Model_0degC"): f = np.full((200, 200), 4800.0, np.float32)
    elif prod.startswith("MergedReflectivityQC_03"): f = np.where(ST, 20.0, -99.0).astype(np.float32)
    elif prod.startswith("NLDN"): f = np.full((200, 200), -1.0, np.float32)
    else:
        key = next(k for k, p in CR.ARCHIVE.items() if p == prod)
        f = np.where(ST, 20.0 if key not in ("hmax", "vii") else (3000.0 if key == "hmax" else 0.0),
                     mc.PRODUCTS[key][1]).astype(np.float32)
    return f, LA.copy(), LO.copy(), None
CR.nearest, CR.fetch = nearest, fetch
CR.anvil_wind = lambda sess, t: None
CR.sat_tops_bt = lambda sess, t, la, lo, z0: None
CR._session = lambda: Sess()
if K is not None:
    K.list_day = lambda sess, d: []
CR.month_hours = lambda ym, days=None: [DAY.replace(hour=h) for h in range(10, 22)]
prebuild()
W = tempfile.mkdtemp(prefix="glmh_")
sys.argv = ["climo_run.py", "2025-07", "--days", "31", "--procs", "1", "--out", os.path.join(W, "out")]
CR.main()
z = np.load(os.path.join(W, "out", "202507.npz"))
hrs = [int(t[11:13]) for t in z["t"]]
p = list(mc.SITES).index("LC-39A"); bl = mc.RULE_KEYS.index("lightning")
lit = ((z["red"][:, p] >> bl) & 1).astype(bool)
print(f"   glm ok {z['glm'].astype(int).tolist()}\n   glm_n {z['glm_n'].tolist()}\n   lightning at LC-39A {lit.astype(int).tolist()}")
check(z["glm"].tolist() == [h != 18 for h in hrs], "GLM usable every hour except 18Z (half its files missing)")
check(lit.tolist() == [h == 15 for h in hrs], "lightning rule fires at LC-39A only at 15Z - from GLM, with NLDN empty")
check(z["glm_n"][hrs.index(15)] == 3 * 90, f"15Z: all flashes in the window counted ({z['glm_n'][hrs.index(15)]})")
check(str(z["classifier"]).endswith("+glm"), f"classifier recorded as {z['classifier']}")
buf = io.StringIO()
sys.argv = ["climo_merge.py", "--inp", os.path.join(W, "out"), "--site", os.path.join(W, "site"), "--web", "web",
            "--fieldmill", "fieldmill/fieldmill_hourly.npz"]
with redirect_stdout(buf):
    CM.main()
line = next((l for l in buf.getvalue().splitlines() if "match the classifier" in l), "")
print("   " + line.strip())
check("(100.000%)" in line, "merge re-scores lightning from features: 100% agreement with the classifier")
meta = json.load(open(os.path.join(W, "site", "climo.json")))
check(meta["coverage"]["202507"].get("glm") == 11, f"coverage: GLM in {meta['coverage']['202507'].get('glm')} of 12 hours")
check(any("GLM total lightning" in n for n in meta["notes"]) and not any("NLDN cloud-to-ground only" in n for n in meta["notes"]),
      "lightning note says NLDN + GLM")
shutil.rmtree("/tmp/glm_site_" + os.path.basename(os.getcwd()), ignore_errors=True)
shutil.copytree(os.path.join(W, "site"), "/tmp/glm_site_" + os.path.basename(os.getcwd()))
print(f"\n{len(FAILS)} failure(s)"); sys.exit(1 if FAILS else 0)

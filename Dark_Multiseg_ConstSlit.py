import ctypes
import struct
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyvisa import ResourceManager

# DLL / Config
DLL32 = r"C:\Program Files (x86)\QD\dlls\LotHW_stdcall.dll"
DLL64 = r"C:\Program Files (x86)\QD\Monochromator Control\LotHW64.dll"
CONFIG_XML = (
    r"C:\Users\PMT_lab\PycharmProjects\PythonProject\ccgData_LOT_MSH-300_SN38594.xml"
)

dllpath = DLL64 if struct.calcsize("P") * 8 == 64 else DLL32
dll = ctypes.WinDLL(dllpath)

dll.LOT_get.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
]
dll.LOT_get.restype = ctypes.c_int
dll.LOT_set.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
]
dll.LOT_set.restype = ctypes.c_int
dll.LOT_version.argtypes = [ctypes.c_char_p]
dll.LOT_version.restype = ctypes.c_int
dll.LOT_get_last_error.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_int),
]
dll.LOT_get_last_error.restype = ctypes.c_int

dll.LOT_build_system_model.argtypes = [ctypes.c_char_p]
dll.LOT_build_system_model.restype = ctypes.c_int
dll.LOT_get_comms_list.argtypes = [ctypes.c_char_p]
dll.LOT_get_comms_list.restype = ctypes.c_int
dll.LOT_get_hardware_list.argtypes = [ctypes.c_char_p]
dll.LOT_get_hardware_list.restype = ctypes.c_int
dll.LOT_initialise.argtypes = []
dll.LOT_initialise.restype = ctypes.c_int
dll.LOT_close.argtypes = []
dll.LOT_close.restype = ctypes.c_int
dll.LOT_select_wavelength.argtypes = [ctypes.c_double]
dll.LOT_select_wavelength.restype = ctypes.c_int

# Tokens (from DLLToken.txt)
TOKEN_MONO_CURRENT_WL = 11
TOKEN_MVSSCurrentWidth = 403
TOKEN_MVSSConstantBandwidth = 404
TOKEN_FWHEEL_POSITION = 102

FILTERWHEEL_ID = b"fwheel"
SHUTTER_POS = 6
OPEN_POS = 1


def cdouble(x):
    return ctypes.byref(ctypes.c_double(float(x)))


def ok(rc, what):
    if rc == 0:
        return
    err = ctypes.c_int()
    sid = ctypes.create_string_buffer(64)
    addr = ctypes.c_int()
    dll.LOT_get_last_error(ctypes.byref(err), sid, ctypes.byref(addr))
    raise RuntimeError(
        f"{what} failed rc={rc}, last_error={err.value}, id='{sid.value.decode(errors='ignore')}', addr={addr.value}"
    )


# Initialization
def initialise():
    buf = ctypes.create_string_buffer(256)
    cfg = ctypes.c_char_p(CONFIG_XML.encode("ascii"))

    ok(dll.LOT_version(buf), "LOT_version")
    ok(dll.LOT_build_system_model(cfg), "LOT_build_system_model")
    ok(dll.LOT_get_comms_list(buf), "LOT_get_comms_list")
    ok(dll.LOT_get_hardware_list(buf), "LOT_get_hardware_list")
    ok(dll.LOT_initialise(), "LOT_initialise")

    print(f"INFO: Monochromator/DLL OK → {buf.value.decode(errors='ignore')}")


# Helpers
def lot_get_double(idb: bytes, token: int) -> float:
    v = ctypes.c_double()
    ok(dll.LOT_get(idb, token, 0, ctypes.byref(v)), f"LOT_get({idb!r}, {token})")
    return v.value


def get_actual_wl() -> float:
    return lot_get_double(b"mono", TOKEN_MONO_CURRENT_WL)


def reselect_current_wl():
    """Re-assert current wavelength to make the firmware apply slit calculations."""
    wl = get_actual_wl()
    ok(dll.LOT_select_wavelength(ctypes.c_double(wl)), f"LOT_select_wavelength({wl})")


def move_and_wait(target_nm: float, tol_nm=0.05, timeout_s=3.0) -> float:
    ok(
        dll.LOT_select_wavelength(ctypes.c_double(float(target_nm))),
        f"LOT_select_wavelength({target_nm})",
    )
    t0 = time.time()
    while True:
        act = get_actual_wl()
        if abs(act - target_nm) <= tol_nm:
            return act
        if time.time() - t0 > timeout_s:
            return act
        time.sleep(0.2)


def set_bw(idb: bytes, bw_nm: float):
    ok(
        dll.LOT_set(idb, TOKEN_MVSSConstantBandwidth, 0, cdouble(bw_nm)),
        f"{idb!r}: set MVSSConstantBandwidth={bw_nm}",
    )
    reselect_current_wl()


def get_filter_position() -> int:
    """Return current filter-wheel position (1–6)."""
    return int(round(lot_get_double(FILTERWHEEL_ID, TOKEN_FWHEEL_POSITION)))


def set_filter_position(pos: int):
    v = ctypes.c_double(float(pos))
    ok(
        dll.LOT_set(FILTERWHEEL_ID, TOKEN_FWHEEL_POSITION, 0, ctypes.byref(v)),
        f"set_filter_position({pos})",
    )
    time.sleep(0.8)

    actual = get_filter_position()
    if actual != pos:
        print(f"[FW WARNING] Requested {pos}, but actual = {actual}. Retrying...")
        ok(
            dll.LOT_set(FILTERWHEEL_ID, TOKEN_FWHEEL_POSITION, 0, ctypes.byref(v)),
            f"set_filter_position({pos}) retry",
        )
        time.sleep(1.0)
        actual = get_filter_position()
        if actual != pos:
            print(f"[FW ERROR] Filter wheel STILL wrong. Using {actual} instead.")
    else:
        print(f"[FW] Filter wheel OK at position {actual}")


# Constant-width slit control
class WidthController:
    """Maintain a target slit width by adapting the constant bandwidth."""

    def __init__(self, idb: bytes, target_mm: float, start_bw_nm=0.07):
        self.idb = idb
        self.target = float(target_mm)
        self.k = None
        self.bw = float(start_bw_nm)
        self.min_bw = 0.005
        self.max_bw = 5.0

    def measure_width(self) -> float:
        return lot_get_double(self.idb, TOKEN_MVSSCurrentWidth)

    def clamp_bw(self, bw: float) -> float:
        return max(self.min_bw, min(self.max_bw, float(bw)))

    def tune(self, wl_nm: float, tol_mm=0.002, max_iter=6) -> tuple[float, float]:
        bw0 = self.clamp_bw(self.bw)
        set_bw(self.idb, bw0)
        time.sleep(0.05)

        w1 = self.measure_width()
        err1 = self.target - w1
        if abs(err1) <= tol_mm:
            self.bw = bw0
            return (bw0, w1)

        if self.k is None:
            probe_bw = self.clamp_bw(bw0 + 0.02)
            set_bw(self.idb, probe_bw)
            time.sleep(0.05)
            w2 = self.measure_width()

            dw = w2 - w1
            db = probe_bw - bw0
            k_est = (dw / db) if abs(db) > 1e-9 else None
            if k_est is None or abs(k_est) < 1e-6:
                self.k = 0.5
            else:
                self.k = k_est

            set_bw(self.idb, bw0)
            time.sleep(0.05)
            w1 = self.measure_width()

        bw_prev = bw0
        w_prev = w1

        for _ in range(max_iter):
            bw_new = self.clamp_bw(bw_prev + (self.target - w_prev) / self.k)
            set_bw(self.idb, bw_new)
            time.sleep(0.05)
            w_new = self.measure_width()

            if abs(self.target - w_new) <= tol_mm:
                if abs(bw_new - bw_prev) > 1e-9:
                    k_new = (w_new - w_prev) / (bw_new - bw_prev)
                    if abs(k_new) > 1e-6:
                        self.k = 0.7 * self.k + 0.3 * k_new
                self.bw = bw_new
                return (bw_new, w_new)

            if abs(bw_new - bw_prev) > 1e-9:
                k_new = (w_new - w_prev) / (bw_new - bw_prev)
                if abs(k_new) > 1e-6:
                    self.k = 0.7 * self.k + 0.3 * k_new

            bw_prev, w_prev = bw_new, w_new
            self.bw = bw_new

        return (bw_prev, w_prev)


# Keithley 6487
class Keithley6487:
    def __init__(self, serial_port="3", verbosity=1):
        rm = ResourceManager()
        self.dev = rm.open_resource(f"ASRL{serial_port}::INSTR")
        self.dev.read_termination = "\r"
        self.dev.write_termination = "\r"
        self.dev.timeout = 1000000

        self.dev.write("*RST")
        self.dev.write("FUNC CURR")
        self.dev.write("RANG 2E-6")
        self.dev.write("RANG:AUTO OFF")
        self.dev.write("SYST:ZCH OFF")
        self.dev.write("SYST:ZCOR OFF")
        self.dev.write("SYST:AZER:STAT OFF")
        self.dev.write("SENS:CURR:DAMP:STAT OFF")
        self.dev.write("*CLS")

        if verbosity:
            print("INFO:", "Connected to", self.dev.query("*IDN?").strip())

    def take_measurement(self, duration_s=1.0, nplc=3, sample_rate=50):
        n_points = max(1, int(duration_s * sample_rate / nplc))

        self.dev.write("*CLS")
        self.dev.write("FORM:ELEM READ,TIME")
        self.dev.write(f"NPLC {float(nplc)}")
        self.dev.write(f"TRIG:COUN {int(n_points)}")
        self.dev.write(f"TRAC:POIN {int(n_points)}")
        self.dev.write("TRAC:CLE")
        self.dev.write("TRAC:FEED:CONT NEXT")
        self.dev.write("TRIG:SOUR IMM")
        self.dev.write("INIT")

        time.sleep(n_points * float(nplc) / sample_rate + 0.5)

        data = self.dev.query_ascii_values("TRAC:DATA?")
        currents = np.array(data[0::2], dtype=float)
        times_ = np.array(data[1::2], dtype=float)
        return currents, times_

    def close(self):
        try:
            self.dev.write("OUTP OFF")
        except Exception:
            pass
        self.dev.close()


# Parameters
CURRENT_UNIT = "nA"
CURRENT_SCALE = 1e9

# Photocurrent acquisition
MEAS_DURATION = 1.0
SCAN_NPLC = 3
SCAN_SAMPLE_RATE = 50

# Dark-current acquisition
PRE_DARK_DURATION_S = 400.0
POST_DARK_N_SECONDS = 10
DARK_WINDOW_S = 1.0
DARK_LOG_INTERVAL_S = 1.0
DARK_NPLC = 3
DARK_SAMPLE_RATE = 50
DARK_SETTLE_PRE_S = 15.0
DARK_SETTLE_POST_S = 2.0
SAVE_DARK_RAW_POINTS = True

# Default slit targets before segment overrides
ENT_TARGET_MM = 0.25
EXT_TARGET_MM = 0.25

scan_segments = [
    {"start": 280, "stop": 396, "step": 5, "width": 0.310},
    {"start": 400, "stop": 701, "step": 5, "width": 0.120},
]
WL_GLOBAL_START = min(s["start"] for s in scan_segments)
WL_GLOBAL_STOP = max(s["stop"] for s in scan_segments)


def update_controller_target(
    controller: WidthController, new_target_mm: float, reset_bw_to=None
):
    if abs(controller.target - float(new_target_mm)) > 1e-12:
        print(f"[Slits] Changing {controller.idb!r} target width → {new_target_mm} mm")
        controller.target = float(new_target_mm)
        controller.k = None
        if reset_bw_to is not None:
            controller.bw = float(reset_bw_to)


# Dark-current logging
def dark_log_loop(
    pico: Keithley6487,
    mode: str,
    total_s: float,
    settle_s: float,
    window_s: float,
    interval_s: float,
    nplc: int,
    sample_rate: int,
    results: list,
):
    """Log dark-current windows and return a ten-window baseline summary."""
    mode = mode.lower().strip()
    if mode not in ("pre", "post"):
        raise ValueError("mode must be 'pre' or 'post'")

    set_filter_position(SHUTTER_POS)
    print(
        f"[Dark-{mode}] Shutter CLOSED (pos {SHUTTER_POS}). Settling {settle_s:.1f} s..."
    )
    time.sleep(settle_s)

    print(
        f"[Dark-{mode}] Live logging for {total_s:.1f} s "
        f"(window={window_s:.1f}s, interval={interval_s:.1f}s, NPLC={nplc}, sr={sample_rate})"
    )

    t_start = time.time()
    next_t = t_start
    chunk_means = []

    chunk_idx = 0
    while True:
        now = time.time()
        t_rel = now - t_start
        if t_rel >= total_s:
            break

        if now < next_t:
            time.sleep(min(0.05, next_t - now))
            continue

        currents_A, times_s = pico.take_measurement(
            duration_s=window_s, nplc=nplc, sample_rate=sample_rate
        )
        currents_nA = (-np.array(currents_A, dtype=float)) * CURRENT_SCALE
        times_s = np.array(times_s, dtype=float)

        N = len(currents_nA)
        mean_nA = float(np.mean(currents_nA)) if N else float("nan")
        std_nA = float(np.std(currents_nA, ddof=1)) if N >= 2 else float("nan")
        ts = datetime.now().isoformat(timespec="seconds")

        chunk_means.append(mean_nA)

        print(
            f"[DARK-{mode.upper()}] {ts} | t={t_rel:7.1f}s | I={mean_nA: .4f} ± {std_nA:.4f} nA | N={N}"
        )

        results.append(
            {
                "record_type": f"{mode}_dark_summary",
                "timestamp": ts,
                "phase_time_s": float(t_rel),
                "chunk_idx": int(chunk_idx),
                "Requested WL (nm)": np.nan,
                "Actual WL (nm)": np.nan,
                "mean_current_nA": mean_nA,
                "std_current_nA": std_nA,
                "n_points_in_window": int(N),
                "window_s": float(window_s),
                "interval_s": float(interval_s),
                "nplc": float(nplc),
                "sample_rate": int(sample_rate),
            }
        )

        if SAVE_DARK_RAW_POINTS:
            if (
                N
                and np.all(np.isfinite(times_s))
                and (np.max(times_s) - np.min(times_s) > 0)
            ):
                t_in_chunk = times_s - times_s[0]
            else:
                t_in_chunk = (
                    np.linspace(0, window_s, num=N, endpoint=False)
                    if N
                    else np.array([])
                )

            for j in range(N):
                results.append(
                    {
                        "record_type": f"{mode}_dark_raw",
                        "timestamp": ts,
                        "phase_time_s": float(t_rel),
                        "chunk_idx": int(chunk_idx),
                        "point_idx": int(j),
                        "t_in_chunk_s": float(t_in_chunk[j]) if N else np.nan,
                        "Requested WL (nm)": np.nan,
                        "Actual WL (nm)": np.nan,
                        "current_nA": float(currents_nA[j]) if N else np.nan,
                        "nplc": float(nplc),
                        "sample_rate": int(sample_rate),
                    }
                )

        chunk_idx += 1
        next_t = time.time() + interval_s

    use_n = min(10, len(chunk_means))
    if use_n == 0:
        return float("nan"), float("nan")

    if mode == "pre":
        sel = np.array(chunk_means[-use_n:], dtype=float)
    else:
        sel = np.array(chunk_means[:use_n], dtype=float)

    mean_sel = float(np.mean(sel))
    std_sel = float(np.std(sel, ddof=1)) if use_n >= 2 else float("nan")

    print(
        f"[Dark-{mode}] Summary over {'last' if mode=='pre' else 'first'} {use_n} seconds: "
        f"{mean_sel:.4f} ± {std_sel:.4f} nA"
    )

    return mean_sel, std_sel


# MAIN
if __name__ == "__main__":
    initialise()
    pico = Keithley6487(serial_port="3", verbosity=1)

    ent_ctl = WidthController(b"Ent", ENT_TARGET_MM, start_bw_nm=0.07)
    ext_ctl = WidthController(b"EXT", EXT_TARGET_MM, start_bw_nm=0.07)

    results = []

    try:
        # Pre-dark
        pre_dark_current, pre_dark_std = dark_log_loop(
            pico=pico,
            mode="pre",
            total_s=PRE_DARK_DURATION_S,
            settle_s=DARK_SETTLE_PRE_S,
            window_s=DARK_WINDOW_S,
            interval_s=DARK_LOG_INTERVAL_S,
            nplc=DARK_NPLC,
            sample_rate=DARK_SAMPLE_RATE,
            results=results,
        )

        # Open optical path
        set_filter_position(OPEN_POS)
        print("[FW] Filter wheel → open filter")

        # Wavelength scan
        for i, seg in enumerate(scan_segments):
            seg_start = seg["start"]
            seg_stop = seg["stop"]
            seg_step = seg["step"]
            seg_w = seg["width"]

            update_controller_target(ent_ctl, seg_w)
            update_controller_target(ext_ctl, seg_w)

            if i < len(scan_segments) - 1:
                wavelengths = np.arange(seg_start, seg_stop, seg_step)
            else:
                wavelengths = np.arange(seg_start, seg_stop + seg_step, seg_step)

            print(
                f"\n=== Segment {seg_start}-{seg_stop} nm (step {seg_step} nm), target width {seg_w} mm ==="
            )

            for wl_req in wavelengths:
                actual = move_and_wait(wl_req)

                fw_pos = get_filter_position()
                print(f"[FW] actual filter wheel position = {fw_pos}")

                be, we = ent_ctl.tune(actual)
                bx, wx = ext_ctl.tune(actual)

                time.sleep(0.3)

                currents, _ = pico.take_measurement(
                    duration_s=MEAS_DURATION,
                    nplc=SCAN_NPLC,
                    sample_rate=SCAN_SAMPLE_RATE,
                )

                currents_nA = (-np.array(currents, dtype=float)) * CURRENT_SCALE
                mean_current = (
                    float(np.mean(currents_nA)) if len(currents_nA) else float("nan")
                )
                std_current = (
                    float(np.std(currents_nA, ddof=1))
                    if len(currents_nA) >= 2
                    else float("nan")
                )

                print(
                    f"\n Requested: {wl_req:6.1f} nm"
                    f" | Actual: {actual:7.3f} nm"
                    f" | Mean Current: {mean_current:7.4f} ± {std_current:7.4f} {CURRENT_UNIT}"
                    f" | Pre-Dark(mean last10s): {pre_dark_current:7.4f} ± {pre_dark_std:7.4f} {CURRENT_UNIT}"
                    f" | Ent W={we:.3f} mm (BW={be:.3f} nm)"
                    f" | Ext W={wx:.3f} mm (BW={bx:.3f} nm)"
                )

                results.append(
                    {
                        "record_type": "scan",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "phase_time_s": np.nan,
                        "chunk_idx": np.nan,
                        "Requested WL (nm)": float(wl_req),
                        "Actual WL (nm)": float(actual),
                        "mean_current_nA": float(mean_current),
                        "std_current_nA": float(std_current),
                        "n_points_in_window": int(len(currents_nA)),
                        "window_s": float(MEAS_DURATION),
                        "interval_s": np.nan,
                        "nplc": float(SCAN_NPLC),
                        "sample_rate": int(SCAN_SAMPLE_RATE),
                        "Pre-Dark Current mean (nA)": float(pre_dark_current),
                        "Pre-Dark Current std (nA)": float(pre_dark_std),
                        "Entry Width (mm)": float(we),
                        "Exit Width (mm)": float(wx),
                        "Entry BW set (nm)": float(be),
                        "Exit BW set (nm)": float(bx),
                    }
                )

        # Post-dark
        print("\n[Post-Dark] Measuring dark current after scan…")
        post_dark_current, post_dark_std = dark_log_loop(
            pico=pico,
            mode="post",
            total_s=float(POST_DARK_N_SECONDS),
            settle_s=DARK_SETTLE_POST_S,
            window_s=DARK_WINDOW_S,
            interval_s=DARK_LOG_INTERVAL_S,
            nplc=DARK_NPLC,
            sample_rate=DARK_SAMPLE_RATE,
            results=results,
        )

        # Save & plot
        seg_tag = "__".join(
            [
                f'{s["start"]:.0f}-{s["stop"]:.0f}_step{s["step"]:.0f}_w{s["width"]:.3f}mm'
                for s in scan_segments
            ]
        )

        outname = (
            f"scan_constWidth_viaBW_PHD_ST5567_phd_(10nA-8mm)_DC_QE_multiSeg_"
            f"{seg_tag}_"
            f"{WL_GLOBAL_START:.1f}-{WL_GLOBAL_STOP:.1f}_"
            f"{datetime.now():%Y%m%d_%H%M%S}.csv"
        )

        df = pd.DataFrame(results)
        df["Post-Dark Current mean (nA)"] = post_dark_current
        df["Post-Dark Current std (nA)"] = post_dark_std

        df.to_csv(outname, index=False)
        print(f"\n✅ Data saved to {outname}")

        # Scan plot
        df_scan = df[df["record_type"] == "scan"].copy()

        plt.figure(figsize=(8, 5))
        plt.plot(
            df_scan["Actual WL (nm)"],
            df_scan["mean_current_nA"],
            marker=".",
            linewidth=1.0,
            label="Mean photocurrent",
        )
        plt.errorbar(
            df_scan["Actual WL (nm)"],
            df_scan["mean_current_nA"],
            yerr=df_scan["std_current_nA"],
            fmt="none",
            linewidth=1.0,
            capsize=3,
        )
        plt.xlabel("Wavelength (nm)")
        plt.ylabel(f"Photocurrent ({CURRENT_UNIT})")
        plt.title("Photocurrent vs Wavelength – Multi-segment constant width (via BW)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Pre-dark diagnostic plot
        df_pre = df[df["record_type"] == "pre_dark_summary"].copy()
        if len(df_pre):
            plt.figure(figsize=(9, 4))
            plt.plot(
                df_pre["phase_time_s"],
                df_pre["mean_current_nA"],
                marker=".",
                linewidth=1.0,
            )
            plt.xlabel("Time (s)")
            plt.ylabel(f"Dark current ({CURRENT_UNIT})")
            plt.title("Pre-dark current (per-second mean)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    finally:
        try:
            pico.close()
        except Exception:
            pass
        try:
            dll.LOT_close()
        except Exception:
            pass

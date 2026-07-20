import ctypes
import os
import struct
import time
import traceback
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytrinamic.connections import ConnectionManager
from pytrinamic.tmcl import TMCLRequest
from pyvisa import ResourceManager

# DLL / MONOCHROMATOR CONFIG
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


# LOT TOKENS
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
        f"{what} failed rc={rc}, "
        f"last_error={err.value}, "
        f"id='{sid.value.decode(errors='ignore')}', "
        f"addr={addr.value}"
    )


def initialise_monochromator():
    buf = ctypes.create_string_buffer(256)
    cfg = ctypes.c_char_p(CONFIG_XML.encode("ascii"))

    ok(dll.LOT_version(buf), "LOT_version")
    ok(dll.LOT_build_system_model(cfg), "LOT_build_system_model")
    ok(dll.LOT_get_comms_list(buf), "LOT_get_comms_list")
    ok(dll.LOT_get_hardware_list(buf), "LOT_get_hardware_list")
    ok(dll.LOT_initialise(), "LOT_initialise")

    print(f"INFO: Monochromator/DLL OK → {buf.value.decode(errors='ignore')}")


# SAFE SAVE HELPERS
def atomic_save_dataframe(df: pd.DataFrame, filename: str):
    temp_name = filename + ".tmp"
    df.to_csv(temp_name, index=False)
    os.replace(temp_name, filename)


def autosave_results(results: list, filename: str, label="AUTOSAVE"):
    try:
        if not results:
            return

        df = pd.DataFrame(results)
        atomic_save_dataframe(df, filename)

        print(f"[{label}] Saved {len(df)} rows → {filename}")

    except Exception as e:
        print(f"[{label} WARNING] Could not save file: {e}")


def emergency_save_results(results: list, prefix: str):
    try:
        if not results:
            print(f"[{prefix}] No results available to save.")
            return None

        emergency_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(results)
        atomic_save_dataframe(df, emergency_name)

        print(f"[{prefix}] Emergency data saved to: {emergency_name}")
        return emergency_name

    except Exception as e:
        print(f"[{prefix} WARNING] Emergency save failed: {e}")
        return None


def build_output_names(
    scan_segments, angle_sequence, wavelength_plan, pmt_tag="PMT_ST5567"
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    wl_min = min(p["wl"] for p in wavelength_plan)
    wl_max = max(p["wl"] for p in wavelength_plan)

    angle_min = min(angle_sequence)
    angle_max = max(angle_sequence)

    seg_tag = "__".join(
        [
            f'{s["start"]:.0f}-{s["stop"]:.0f}_step{s["step"]:.0f}_w{s["width"]:.3f}mm'
            for s in scan_segments
        ]
    )

    base = (
        f"angular_spectral_scan_RAW_{pmt_tag}_"
        f"angles{angle_min:.0f}_to_{angle_max:.0f}_"
        f"WL{wl_min:.1f}-{wl_max:.1f}_"
        f"{seg_tag}_"
        f"{timestamp}"
    )

    final_name = base + ".csv"
    autosave_name = "AUTOSAVE_" + base + ".csv"

    return final_name, autosave_name


# MONOCHROMATOR HELPERS
def lot_get_double(idb: bytes, token: int) -> float:
    v = ctypes.c_double()
    ok(dll.LOT_get(idb, token, 0, ctypes.byref(v)), f"LOT_get({idb!r}, {token})")
    return v.value


def get_actual_wl() -> float:
    return lot_get_double(b"mono", TOKEN_MONO_CURRENT_WL)


def reselect_current_wl():
    wl = get_actual_wl()
    ok(dll.LOT_select_wavelength(ctypes.c_double(wl)), f"LOT_select_wavelength({wl})")


def move_and_wait_wavelength(target_nm: float, tol_nm=0.05, timeout_s=8.0) -> float:
    ok(
        dll.LOT_select_wavelength(ctypes.c_double(float(target_nm))),
        f"LOT_select_wavelength({target_nm})",
    )

    t0 = time.time()

    while True:
        actual = get_actual_wl()

        if abs(actual - target_nm) <= tol_nm:
            return actual

        if time.time() - t0 > timeout_s:
            print(
                f"[WL WARNING] Requested {target_nm:.3f} nm, actual {actual:.3f} nm after timeout."
            )
            return actual

        time.sleep(0.2)


def get_filter_position() -> int:
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
        print(f"[FW WARNING] Requested {pos}, actual {actual}. Retrying...")
        ok(
            dll.LOT_set(FILTERWHEEL_ID, TOKEN_FWHEEL_POSITION, 0, ctypes.byref(v)),
            f"set_filter_position({pos}) retry",
        )
        time.sleep(1.0)
        actual = get_filter_position()

        if actual != pos:
            print(
                f"[FW ERROR] Filter wheel still wrong. Requested {pos}, actual {actual}."
            )
    else:
        print(f"[FW] Filter wheel OK at position {actual}")


def close_shutter():
    set_filter_position(SHUTTER_POS)


def open_shutter():
    set_filter_position(OPEN_POS)


def set_bw(idb: bytes, bw_nm: float):
    ok(
        dll.LOT_set(idb, TOKEN_MVSSConstantBandwidth, 0, cdouble(bw_nm)),
        f"{idb!r}: set MVSSConstantBandwidth={bw_nm}",
    )
    reselect_current_wl()


# CONSTANT-WIDTH SLIT CONTROLLER
class WidthController:
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

    def tune(self, tol_mm=0.002, max_iter=6) -> tuple[float, float]:
        bw0 = self.clamp_bw(self.bw)
        set_bw(self.idb, bw0)
        time.sleep(0.05)

        w1 = self.measure_width()

        if abs(self.target - w1) <= tol_mm:
            self.bw = bw0
            return bw0, w1

        if self.k is None:
            probe_bw = self.clamp_bw(bw0 + 0.02)
            set_bw(self.idb, probe_bw)
            time.sleep(0.05)

            w2 = self.measure_width()

            dw = w2 - w1
            db = probe_bw - bw0

            if abs(db) > 1e-9 and abs(dw / db) > 1e-6:
                self.k = dw / db
            else:
                self.k = 0.5

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
                return bw_new, w_new

            if abs(bw_new - bw_prev) > 1e-9:
                k_new = (w_new - w_prev) / (bw_new - bw_prev)
                if abs(k_new) > 1e-6:
                    self.k = 0.7 * self.k + 0.3 * k_new

            bw_prev, w_prev = bw_new, w_new
            self.bw = bw_new

        return bw_prev, w_prev


def update_controller_target(
    controller: WidthController, new_target_mm: float, reset_bw_to=None
):
    if abs(controller.target - float(new_target_mm)) > 1e-12:
        print(
            f"[Slits] Changing {controller.idb!r} target width → {new_target_mm:.3f} mm"
        )
        controller.target = float(new_target_mm)
        controller.k = None

        if reset_bw_to is not None:
            controller.bw = float(reset_bw_to)


# Keithley 6487
class Keithley6487:
    def __init__(self, serial_port="3", verbosity=1):
        self.serial_port = serial_port
        self.resource_name = f"ASRL{serial_port}::INSTR"
        self.rm = ResourceManager()
        self.dev = None

        self.connect_and_configure(verbosity=verbosity)

    def connect_and_configure(self, verbosity=1):
        self.dev = self.rm.open_resource(self.resource_name)

        self.dev.read_termination = "\r"
        self.dev.write_termination = "\r"

        # Bound READ? calls so a stalled instrument can be reconnected.
        self.dev.timeout = 30000

        try:
            self.dev.clear()
        except Exception:
            pass

        self.dev.write("*RST")
        time.sleep(0.5)

        self.dev.write("FUNC CURR")
        self.dev.write("RANG 2E-6")
        self.dev.write("RANG:AUTO OFF")
        self.dev.write("SYST:ZCH OFF")
        self.dev.write("SYST:ZCOR OFF")
        self.dev.write("SYST:AZER:STAT OFF")
        self.dev.write("SENS:CURR:DAMP:STAT OFF")
        self.dev.write("NPLC 1")
        self.dev.write("FORM:ELEM READ")
        self.dev.write("*CLS")

        if verbosity:
            print("INFO:", "Connected to", self.dev.query("*IDN?").strip())

    def reconnect(self):
        print("[Keithley] Reconnecting VISA session...")

        try:
            if self.dev is not None:
                self.dev.close()
        except Exception:
            pass

        time.sleep(1.0)
        self.connect_and_configure(verbosity=0)

    def take_measurement(self, duration_s=1.0, nplc=1, sample_rate=5, max_retries=3):
        """Acquire repeated direct readings with retry and reconnection support."""
        n_reads = max(3, int(duration_s * sample_rate))
        dt = max(0.05, duration_s / n_reads)

        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                print(f"[Keithley] Direct-read attempt {attempt}/{max_retries}")

                try:
                    self.dev.clear()
                except Exception:
                    pass

                self.dev.write("*CLS")
                self.dev.write("ABOR")
                self.dev.write("FUNC CURR")
                self.dev.write("RANG 2E-6")
                self.dev.write("RANG:AUTO OFF")
                self.dev.write("SYST:ZCH OFF")
                self.dev.write("SYST:ZCOR OFF")
                self.dev.write("SYST:AZER:STAT OFF")
                self.dev.write("SENS:CURR:DAMP:STAT OFF")
                self.dev.write(f"NPLC {float(nplc)}")
                self.dev.write("FORM:ELEM READ")

                currents = []
                times_ = []

                t0 = time.time()

                for _ in range(n_reads):
                    raw = self.dev.query("READ?").strip()

                    val = float(raw.split(",")[0])

                    currents.append(val)
                    times_.append(time.time() - t0)

                    time.sleep(dt)

                return np.array(currents, dtype=float), np.array(times_, dtype=float)

            except Exception as e:
                last_error = e
                print(
                    f"[Keithley WARNING] Direct read failed on attempt {attempt}/{max_retries}: {e}"
                )

                try:
                    self.dev.write("ABOR")
                    self.dev.write("*CLS")
                except Exception:
                    pass

                time.sleep(2.0)

                if attempt == 2:
                    try:
                        self.reconnect()
                    except Exception as reconnect_error:
                        print(f"[Keithley WARNING] Reconnect failed: {reconnect_error}")

        raise RuntimeError(
            f"Keithley direct-read measurement failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def close(self):
        try:
            if self.dev is not None:
                self.dev.write("OUTP OFF")
        except Exception:
            pass

        try:
            if self.dev is not None:
                self.dev.close()
        except Exception:
            pass


# ROTATIONAL STAGE
class RotStage:
    def __init__(
        self,
        port="COM5",
        baud=9600,
        addr=1,
        motor=0,
        units_per_deg=9400,
        vmax_int=2047,
        amax_int=2047,
    ):
        self.addr = addr
        self.motor = motor
        self.units_per_deg = float(units_per_deg)

        self.cm = ConnectionManager(
            f"--interface serial_tmcl --port {port} --data_rate={baud}"
        )
        self.iface = self.cm.connect()

        self.set_velocity(vmax_int)
        self.set_acceleration(amax_int)

    def _tmcl(self, cmd, typ, val):
        req = TMCLRequest(self.addr, cmd, typ, self.motor, int(val))
        return self.iface.send_request(req)

    def _sap(self, ap, val):
        return self._tmcl(5, ap, val)

    def _gap(self, ap):
        return self._tmcl(6, ap, 0).value

    def signed_int32(self, value: int) -> int:
        value = int(value)

        if value >= 2**31:
            value -= 2**32

        return value

    def set_velocity(self, vmax_int: int):
        self._sap(4, vmax_int)

    def set_acceleration(self, amax_int: int):
        self._sap(5, amax_int)

    def zero(self):
        self._sap(1, 0)

    def deg_to_units(self, deg: float) -> int:
        return int(round(float(deg) * self.units_per_deg))

    def units_to_deg(self, units: int) -> float:
        return float(units) / self.units_per_deg

    def goto(self, deg: float):
        units = self.deg_to_units(deg)
        self._tmcl(4, 0, units)

    def move_by(self, deg: float):
        units = self.deg_to_units(deg)
        self._tmcl(4, 1, units)

    def stop(self):
        self._tmcl(3, 0, 0)

    def get_internal_units_raw(self) -> int:
        return int(self._gap(1))

    def get_internal_units(self) -> int:
        raw = self.get_internal_units_raw()
        return self.signed_int32(raw)

    def get_estimated_angle(self) -> float:
        return self.units_to_deg(self.get_internal_units())

    def print_position_debug(self):
        raw = self.get_internal_units_raw()
        signed = self.get_internal_units()
        angle = self.get_estimated_angle()

        print(f"[Stage DEBUG] raw={raw}, signed={signed}, angle={angle:.3f}°")

    def wait_until_reached(self, target_deg: float, tol_deg=0.15, timeout_s=45.0):
        t0 = time.time()

        while True:
            actual_deg = self.get_estimated_angle()
            err = abs(actual_deg - target_deg)

            if err <= tol_deg:
                print(f"[Stage] Reached {actual_deg:.3f}°")
                return actual_deg

            if time.time() - t0 > timeout_s:
                raw = self.get_internal_units_raw()
                signed = self.get_internal_units()

                print(
                    f"[Stage WARNING] Timeout. "
                    f"Target={target_deg:.3f}°, actual={actual_deg:.3f}°, "
                    f"raw_units={raw}, signed_units={signed}"
                )
                return actual_deg

            time.sleep(0.25)

    def close(self):
        try:
            self.stop()
        except Exception:
            pass

        self.cm.disconnect()


# USER SETTINGS

CURRENT_UNIT = "nA"
CURRENT_SCALE = 1e9

# Keithley photocurrent scan settings
MEAS_DURATION = 1.0
SCAN_NPLC = 1
SCAN_SAMPLE_RATE = 5

# Dark settings
GLOBAL_PRE_DARK_DURATION_S = 30.0
GLOBAL_PRE_DARK_SETTLE_S = 15.0
GLOBAL_PRE_DARK_LAST_N_FOR_MEAN = 3

FINAL_POST_DARK_POINTS = 10
FINAL_POST_DARK_LAST_N_FOR_MEAN = 5
FINAL_POST_DARK_SETTLE_S = 2.0

DARK_WINDOW_S = 1.0
DARK_INTERVAL_S = 1.0
DARK_NPLC = 1
DARK_SAMPLE_RATE = 5

SAVE_DARK_RAW_POINTS = True

# Wavelength scan settings
scan_segments = [
    {"start": 280, "stop": 396, "step": 5, "width": 0.310},
    {"start": 400, "stop": 701, "step": 5, "width": 0.120},
]

ENT_START_BW_NM = 0.07
EXT_START_BW_NM = 0.07

# Hardware ports
KEITHLEY_SERIAL_PORT = "3"
ROT_STAGE_PORT = "COM5"

# Rotation stage settings
ZERO_STAGE_AT_START = False
UNITS_PER_DEG = 9400

STAGE_VELOCITY = 2047
STAGE_ACCELERATION = 2047
STAGE_TOL_DEG = 0.15
STAGE_TIMEOUT_S = 45.0

# Angular sequence
ANGLE_SEQUENCE = [
    0,
    60,
    55,
    50,
    45,
    40,
    35,
    30,
    25,
    20,
    15,
    10,
    5,
    0,
    -5,
    -10,
    -15,
    -20,
    -25,
    -30,
    -35,
    -40,
    -45,
    -50,
    -55,
    -60,
    0,
]


# MEASUREMENT HELPERS
def currents_to_nA(currents_A):
    return -np.array(currents_A, dtype=float) * CURRENT_SCALE


def measure_current_window(
    pico: Keithley6487,
    duration_s: float,
    nplc: float,
    sample_rate: int,
):
    currents_A, times_s = pico.take_measurement(
        duration_s=duration_s,
        nplc=nplc,
        sample_rate=sample_rate,
        max_retries=3,
    )

    currents_nA = currents_to_nA(currents_A)

    n = len(currents_nA)
    mean_nA = float(np.mean(currents_nA)) if n else float("nan")
    std_nA = float(np.std(currents_nA, ddof=1)) if n >= 2 else float("nan")

    return currents_nA, np.array(times_s, dtype=float), mean_nA, std_nA, n


def build_wavelength_plan(scan_segments):
    plan = []

    for i, seg in enumerate(scan_segments):
        start = float(seg["start"])
        stop = float(seg["stop"])
        step = float(seg["step"])
        width = float(seg["width"])

        if step <= 0:
            raise ValueError("Wavelength step must be positive.")

        wavelengths = np.arange(start, stop + 0.5 * step, step)

        for wl in wavelengths:
            if wl <= stop + 1e-9:
                plan.append(
                    {
                        "wl": float(wl),
                        "width": width,
                        "segment_index": int(i),
                        "segment_start": start,
                        "segment_stop": stop,
                        "segment_step": step,
                    }
                )

    return plan


def summarize_last_n(values, last_n=3):
    values = np.array(values, dtype=float)

    if len(values) == 0:
        return float("nan"), float("nan"), 0

    use_n = min(int(last_n), len(values))
    selected = values[-use_n:]

    mean_val = float(np.mean(selected))
    std_val = float(np.std(selected, ddof=1)) if use_n >= 2 else float("nan")

    return mean_val, std_val, use_n


def log_dark_by_duration(
    pico: Keithley6487,
    results: list,
    mode: str,
    angle_deg: float,
    total_s: float,
    settle_s: float,
    summary_last_n: int = 3,
    autosave_name: str | None = None,
):
    close_shutter()

    print(f"\n[Dark-{mode}] Shutter CLOSED. Settling {settle_s:.1f} s...")
    time.sleep(settle_s)

    print(f"[Dark-{mode}] Logging for {total_s:.1f} s at angle {angle_deg:.2f}°")

    t_start = time.time()
    chunk_idx = 0
    means = []

    while True:
        phase_time = time.time() - t_start

        if phase_time >= total_s:
            break

        currents_nA, times_s, mean_nA, std_nA, n = measure_current_window(
            pico,
            duration_s=DARK_WINDOW_S,
            nplc=DARK_NPLC,
            sample_rate=DARK_SAMPLE_RATE,
        )

        ts = datetime.now().isoformat(timespec="seconds")
        means.append(mean_nA)

        print(
            f"[DARK-{mode.upper()}] {ts} | "
            f"angle={angle_deg:7.2f}° | "
            f"t={phase_time:7.1f}s | "
            f"I={mean_nA: .5f} ± {std_nA:.5f} nA | N={n}"
        )

        results.append(
            {
                "record_type": f"{mode}_dark_summary",
                "timestamp": ts,
                "angle_deg_requested": float(angle_deg),
                "angle_deg_actual": np.nan,
                "phase_time_s": float(phase_time),
                "chunk_idx": int(chunk_idx),
                "Requested WL (nm)": np.nan,
                "Actual WL (nm)": np.nan,
                "mean_current_nA": mean_nA,
                "std_current_nA": std_nA,
                "n_points_in_window": int(n),
                "window_s": float(DARK_WINDOW_S),
                "nplc": float(DARK_NPLC),
                "sample_rate": int(DARK_SAMPLE_RATE),
            }
        )

        if SAVE_DARK_RAW_POINTS:
            if (
                n
                and np.all(np.isfinite(times_s))
                and (np.max(times_s) - np.min(times_s) > 0)
            ):
                t_in_chunk = times_s - times_s[0]
            else:
                t_in_chunk = np.linspace(0, DARK_WINDOW_S, num=n, endpoint=False)

            for j in range(n):
                results.append(
                    {
                        "record_type": f"{mode}_dark_raw",
                        "timestamp": ts,
                        "angle_deg_requested": float(angle_deg),
                        "angle_deg_actual": np.nan,
                        "phase_time_s": float(phase_time),
                        "chunk_idx": int(chunk_idx),
                        "point_idx": int(j),
                        "t_in_chunk_s": float(t_in_chunk[j]),
                        "current_nA": float(currents_nA[j]),
                        "nplc": float(DARK_NPLC),
                        "sample_rate": int(DARK_SAMPLE_RATE),
                    }
                )

        if autosave_name is not None:
            autosave_results(results, autosave_name)

        chunk_idx += 1
        time.sleep(DARK_INTERVAL_S)

    dark_mean, dark_std, used_n = summarize_last_n(means, last_n=summary_last_n)

    print(
        f"[Dark-{mode}] Summary over LAST {used_n} dark readings: "
        f"{dark_mean:.5f} ± {dark_std:.5f} nA"
    )

    if autosave_name is not None:
        autosave_results(results, autosave_name)

    return dark_mean, dark_std, used_n


def log_dark_by_points(
    pico: Keithley6487,
    results: list,
    mode: str,
    angle_deg_requested: float,
    angle_deg_actual: float,
    n_points_dark: int,
    settle_s: float,
    summary_last_n: int = 3,
    autosave_name: str | None = None,
):
    close_shutter()

    print(
        f"\n[Dark-{mode}] Shutter CLOSED. "
        f"Angle={angle_deg_requested:.2f}°, settling {settle_s:.1f} s..."
    )

    time.sleep(settle_s)

    means = []

    for chunk_idx in range(n_points_dark):
        currents_nA, times_s, mean_nA, std_nA, n = measure_current_window(
            pico,
            duration_s=DARK_WINDOW_S,
            nplc=DARK_NPLC,
            sample_rate=DARK_SAMPLE_RATE,
        )

        ts = datetime.now().isoformat(timespec="seconds")
        means.append(mean_nA)

        print(
            f"[DARK-{mode.upper()}] {ts} | "
            f"angle={angle_deg_actual:7.3f}° | "
            f"point={chunk_idx + 1}/{n_points_dark} | "
            f"I={mean_nA: .5f} ± {std_nA:.5f} nA | N={n}"
        )

        results.append(
            {
                "record_type": f"{mode}_dark_summary",
                "timestamp": ts,
                "angle_deg_requested": float(angle_deg_requested),
                "angle_deg_actual": float(angle_deg_actual),
                "phase_time_s": np.nan,
                "chunk_idx": int(chunk_idx),
                "Requested WL (nm)": np.nan,
                "Actual WL (nm)": np.nan,
                "mean_current_nA": mean_nA,
                "std_current_nA": std_nA,
                "n_points_in_window": int(n),
                "window_s": float(DARK_WINDOW_S),
                "nplc": float(DARK_NPLC),
                "sample_rate": int(DARK_SAMPLE_RATE),
            }
        )

        if SAVE_DARK_RAW_POINTS:
            if (
                n
                and np.all(np.isfinite(times_s))
                and (np.max(times_s) - np.min(times_s) > 0)
            ):
                t_in_chunk = times_s - times_s[0]
            else:
                t_in_chunk = np.linspace(0, DARK_WINDOW_S, num=n, endpoint=False)

            for j in range(n):
                results.append(
                    {
                        "record_type": f"{mode}_dark_raw",
                        "timestamp": ts,
                        "angle_deg_requested": float(angle_deg_requested),
                        "angle_deg_actual": float(angle_deg_actual),
                        "chunk_idx": int(chunk_idx),
                        "point_idx": int(j),
                        "t_in_chunk_s": float(t_in_chunk[j]),
                        "current_nA": float(currents_nA[j]),
                        "nplc": float(DARK_NPLC),
                        "sample_rate": int(DARK_SAMPLE_RATE),
                    }
                )

        if autosave_name is not None:
            autosave_results(results, autosave_name)

        time.sleep(DARK_INTERVAL_S)

    dark_mean, dark_std, used_n = summarize_last_n(means, last_n=summary_last_n)

    print(
        f"[Dark-{mode}] Summary over LAST {used_n} dark readings: "
        f"{dark_mean:.5f} ± {dark_std:.5f} nA"
    )

    if autosave_name is not None:
        autosave_results(results, autosave_name)

    return dark_mean, dark_std, used_n


def move_stage_safely(stage: RotStage, target_angle: float):
    print(f"\n[Move] Preparing to move to {target_angle:.2f}°")
    close_shutter()

    print("[Move] Shutter closed. Moving rotation stage...")
    stage.goto(target_angle)

    actual_angle = stage.wait_until_reached(
        target_deg=target_angle,
        tol_deg=STAGE_TOL_DEG,
        timeout_s=STAGE_TIMEOUT_S,
    )

    return actual_angle


def measure_spectral_scan_at_angle(
    pico: Keithley6487,
    ent_ctl: WidthController,
    ext_ctl: WidthController,
    results: list,
    angle_req: float,
    angle_actual: float,
    angle_block_idx: int,
    wavelength_plan: list,
    global_pre_dark_mean_nA: float,
    global_pre_dark_std_nA: float,
    global_pre_dark_used_n: int,
    autosave_name: str | None = None,
):
    print(f"\n[SPECTRAL SCAN] Starting wavelength scan at angle {angle_actual:.3f}°")

    open_shutter()
    print("[SPECTRAL SCAN] Shutter OPEN.")

    last_width = None

    for point_idx, point in enumerate(wavelength_plan):
        wl_req = float(point["wl"])
        width_mm = float(point["width"])
        segment_index = int(point["segment_index"])

        print(
            f"\n[SCAN POINT] angle={angle_actual:.3f}° | "
            f"{point_idx + 1}/{len(wavelength_plan)} | "
            f"WL={wl_req:.1f} nm | width={width_mm:.3f} mm"
        )

        actual_wl = move_and_wait_wavelength(wl_req)

        if last_width is None or abs(width_mm - last_width) > 1e-12:
            update_controller_target(ent_ctl, width_mm)
            update_controller_target(ext_ctl, width_mm)
            last_width = width_mm

        be, we = ent_ctl.tune()
        bx, wx = ext_ctl.tune()

        time.sleep(0.3)

        currents_nA, times_s, mean_nA, std_nA, n = measure_current_window(
            pico,
            duration_s=MEAS_DURATION,
            nplc=SCAN_NPLC,
            sample_rate=SCAN_SAMPLE_RATE,
        )

        fw_pos = get_filter_position()
        ts = datetime.now().isoformat(timespec="seconds")

        print(
            f"[PHOTO] {ts} | "
            f"angle={angle_actual:7.3f}° | "
            f"WL={actual_wl:8.3f} nm | "
            f"I={mean_nA: .5f} ± {std_nA:.5f} nA | "
            f"N={n} | "
            f"Global pre-dark(last {global_pre_dark_used_n})={global_pre_dark_mean_nA:.5f} nA | "
            f"Ent W={we:.3f} mm | Ext W={wx:.3f} mm"
        )

        results.append(
            {
                "record_type": "scan",
                "timestamp": ts,
                "angle_block_idx": int(angle_block_idx),
                "angle_deg_requested": float(angle_req),
                "angle_deg_actual": float(angle_actual),
                "scan_point_idx": int(point_idx),
                "scan_point_number": int(point_idx + 1),
                "n_scan_points_in_angle": int(len(wavelength_plan)),
                "segment_index": int(segment_index),
                "Requested WL (nm)": float(wl_req),
                "Actual WL (nm)": float(actual_wl),
                "mean_current_nA": float(mean_nA),
                "std_current_nA": float(std_nA),
                "n_points_in_window": int(n),
                "global_pre_dark_mean_nA": float(global_pre_dark_mean_nA),
                "global_pre_dark_std_nA": float(global_pre_dark_std_nA),
                "global_pre_dark_used_n": int(global_pre_dark_used_n),
                "Entry Width (mm)": float(we),
                "Exit Width (mm)": float(wx),
                "Entry BW set (nm)": float(be),
                "Exit BW set (nm)": float(bx),
                "target_slit_width_mm": float(width_mm),
                "filter_wheel_position": int(fw_pos),
                "window_s": float(MEAS_DURATION),
                "nplc": float(SCAN_NPLC),
                "sample_rate": int(SCAN_SAMPLE_RATE),
            }
        )

        if autosave_name is not None:
            autosave_results(results, autosave_name)

    print(f"\n[SPECTRAL SCAN] Completed angle {angle_actual:.3f}")

    if autosave_name is not None:
        autosave_results(results, autosave_name)


# MAIN SEQUENCE
if __name__ == "__main__":

    results = []

    pico = None
    stage = None

    final_post_dark_mean = np.nan
    final_post_dark_std = np.nan
    final_post_dark_used_n = 0

    global_pre_dark_mean = np.nan
    global_pre_dark_std = np.nan
    global_pre_dark_used_n = 0

    final_outname = None
    autosave_name = None

    try:
        wavelength_plan = build_wavelength_plan(scan_segments)

        final_outname, autosave_name = build_output_names(
            scan_segments=scan_segments,
            angle_sequence=ANGLE_SEQUENCE,
            wavelength_plan=wavelength_plan,
            pmt_tag="PMT_ST5567",
        )

        print("\n========== SCAN PLAN ==========")
        print(f"Number of angle positions: {len(ANGLE_SEQUENCE)}")
        print(f"Number of wavelength points per angle: {len(wavelength_plan)}")
        print(
            f"Total photocurrent points: {len(ANGLE_SEQUENCE) * len(wavelength_plan)}"
        )
        print("Angles:", ANGLE_SEQUENCE)
        print(
            f"Wavelength range: {min(p['wl'] for p in wavelength_plan):.1f} "
            f"to {max(p['wl'] for p in wavelength_plan):.1f} nm"
        )
        print(f"[SAVE] Final output file: {final_outname}")
        print(f"[SAVE] Autosave file:    {autosave_name}")
        print("================================\n")

        results.append(
            {
                "record_type": "run_start",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "final_output_file": final_outname,
                "autosave_file": autosave_name,
                "n_angles": int(len(ANGLE_SEQUENCE)),
                "n_wavelength_points_per_angle": int(len(wavelength_plan)),
                "total_photocurrent_points": int(
                    len(ANGLE_SEQUENCE) * len(wavelength_plan)
                ),
            }
        )
        autosave_results(results, autosave_name, label="INITIAL SAVE")

        initialise_monochromator()

        pico = Keithley6487(serial_port=KEITHLEY_SERIAL_PORT, verbosity=1)

        stage = RotStage(
            port=ROT_STAGE_PORT,
            units_per_deg=UNITS_PER_DEG,
            vmax_int=STAGE_VELOCITY,
            amax_int=STAGE_ACCELERATION,
        )

        if ZERO_STAGE_AT_START:
            print(
                "[Stage] ZERO_STAGE_AT_START=True. Setting current physical position as 0°."
            )
            stage.zero()
            time.sleep(0.5)

        print(f"[Stage] Current estimated angle: {stage.get_estimated_angle():.3f}°")
        stage.print_position_debug()

        first_width = float(wavelength_plan[0]["width"])

        ent_ctl = WidthController(b"Ent", first_width, start_bw_nm=ENT_START_BW_NM)
        ext_ctl = WidthController(b"EXT", first_width, start_bw_nm=EXT_START_BW_NM)

        # Global pre-dark
        print("\n========== GLOBAL PRE-DARK ==========")

        global_pre_dark_mean, global_pre_dark_std, global_pre_dark_used_n = (
            log_dark_by_duration(
                pico=pico,
                results=results,
                mode="global_pre",
                angle_deg=stage.get_estimated_angle(),
                total_s=GLOBAL_PRE_DARK_DURATION_S,
                settle_s=GLOBAL_PRE_DARK_SETTLE_S,
                summary_last_n=GLOBAL_PRE_DARK_LAST_N_FOR_MEAN,
                autosave_name=autosave_name,
            )
        )

        # Angular wavelength scan
        print("\n========== ANGULAR + WAVELENGTH SCAN ==========")

        for angle_idx, angle_req in enumerate(ANGLE_SEQUENCE):

            print("\n" + "=" * 90)
            print(
                f"[ANGLE BLOCK] {angle_idx + 1}/{len(ANGLE_SEQUENCE)} | "
                f"Target angle = {angle_req:.2f}°"
            )
            print("=" * 90)

            results.append(
                {
                    "record_type": "angle_block_start",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "angle_block_idx": int(angle_idx),
                    "angle_deg_requested": float(angle_req),
                }
            )
            autosave_results(results, autosave_name)

            angle_actual = move_stage_safely(stage, angle_req)

            measure_spectral_scan_at_angle(
                pico=pico,
                ent_ctl=ent_ctl,
                ext_ctl=ext_ctl,
                results=results,
                angle_req=angle_req,
                angle_actual=angle_actual,
                angle_block_idx=angle_idx,
                wavelength_plan=wavelength_plan,
                global_pre_dark_mean_nA=global_pre_dark_mean,
                global_pre_dark_std_nA=global_pre_dark_std,
                global_pre_dark_used_n=global_pre_dark_used_n,
                autosave_name=autosave_name,
            )

            results.append(
                {
                    "record_type": "angle_block_summary",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "angle_block_idx": int(angle_idx),
                    "angle_deg_requested": float(angle_req),
                    "angle_deg_actual": float(angle_actual),
                    "global_pre_dark_mean_nA": float(global_pre_dark_mean),
                    "global_pre_dark_std_nA": float(global_pre_dark_std),
                    "global_pre_dark_used_n": int(global_pre_dark_used_n),
                    "n_wavelength_points": int(len(wavelength_plan)),
                }
            )

            autosave_results(results, autosave_name)

        # Final post-dark
        print("\n========== FINAL POST-DARK ==========")

        final_angle_now = stage.get_estimated_angle()

        final_post_dark_mean, final_post_dark_std, final_post_dark_used_n = (
            log_dark_by_points(
                pico=pico,
                results=results,
                mode="final_post",
                angle_deg_requested=final_angle_now,
                angle_deg_actual=final_angle_now,
                n_points_dark=FINAL_POST_DARK_POINTS,
                settle_s=FINAL_POST_DARK_SETTLE_S,
                summary_last_n=FINAL_POST_DARK_LAST_N_FOR_MEAN,
                autosave_name=autosave_name,
            )
        )

        results.append(
            {
                "record_type": "final_post_dark_summary",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "angle_deg_requested": float(final_angle_now),
                "angle_deg_actual": float(final_angle_now),
                "final_post_dark_mean_nA": float(final_post_dark_mean),
                "final_post_dark_std_nA": float(final_post_dark_std),
                "final_post_dark_used_n": int(final_post_dark_used_n),
            }
        )

        autosave_results(results, autosave_name)

        # Save data
        df = pd.DataFrame(results)

        df["global_pre_dark_mean_nA"] = global_pre_dark_mean
        df["global_pre_dark_std_nA"] = global_pre_dark_std
        df["global_pre_dark_used_n"] = global_pre_dark_used_n

        df["final_post_dark_mean_nA"] = final_post_dark_mean
        df["final_post_dark_std_nA"] = final_post_dark_std
        df["final_post_dark_used_n"] = final_post_dark_used_n

        atomic_save_dataframe(df, final_outname)

        print(f"\n✅ Final data saved to: {final_outname}")
        print(f"✅ Autosave copy also available at: {autosave_name}")

        # Raw diagnostic plot
        df_scan = df[df["record_type"] == "scan"].copy()

        if len(df_scan):
            plt.figure(figsize=(9, 5))

            for angle in sorted(df_scan["angle_deg_actual"].dropna().unique()):
                sub = df_scan[np.isclose(df_scan["angle_deg_actual"], angle)]
                plt.plot(
                    sub["Actual WL (nm)"],
                    sub["mean_current_nA"],
                    marker=".",
                    linewidth=1.0,
                    label=f"{angle:.1f}°",
                )

            plt.xlabel("Wavelength (nm)")
            plt.ylabel(f"Raw photocurrent ({CURRENT_UNIT})")
            plt.title("Raw photocurrent spectrum at each angle")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.show()

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Measurement stopped by user using Ctrl+C.")

        if autosave_name is not None:
            autosave_results(results, autosave_name, label="INTERRUPT AUTOSAVE")

        emergency_save_results(results, prefix="INTERRUPTED_SCAN")

    except Exception as e:
        print("\n[ERROR] Measurement crashed.")
        print(f"[ERROR] {type(e).__name__}: {e}")
        print("\n[ERROR TRACEBACK]")
        traceback.print_exc()

        if autosave_name is not None:
            autosave_results(results, autosave_name, label="CRASH AUTOSAVE")

        emergency_save_results(results, prefix="CRASHED_SCAN")

        raise

    finally:
        print("\n[Cleanup] Emergency/normal cleanup started...")

        try:
            print("[Cleanup] Closing monochromator shutter...")
            close_shutter()
        except Exception as e:
            print(f"[Cleanup WARNING] Could not close shutter: {e}")

        if stage is not None:
            try:
                print("[Cleanup] Returning rotational stage to 0.00°...")
                stage.goto(0.0)

                final_angle = stage.wait_until_reached(
                    target_deg=0.0,
                    tol_deg=STAGE_TOL_DEG,
                    timeout_s=STAGE_TIMEOUT_S,
                )

                print(f"[Cleanup] Stage returned to {final_angle:.3f}°")
                stage.print_position_debug()

            except Exception as e:
                print(f"[Cleanup WARNING] Could not return stage to 0°: {e}")

        if pico is not None:
            try:
                print("[Cleanup] Closing Keithley...")
                pico.close()
            except Exception as e:
                print(f"[Cleanup WARNING] Could not close Keithley: {e}")

        if stage is not None:
            try:
                print("[Cleanup] Closing rotation stage connection...")
                stage.close()
            except Exception as e:
                print(f"[Cleanup WARNING] Could not close stage: {e}")

        try:
            print("[Cleanup] Closing monochromator DLL...")
            dll.LOT_close()
        except Exception as e:
            print(f"[Cleanup WARNING] Could not close LOT DLL: {e}")

        print("[Cleanup] Done.")

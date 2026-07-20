"""Hold the LOT MSH-300 at a wavelength and log Keithley 6487 current.

The interactive interface supports wavelength, slit, filter-wheel, shutter, and
status commands. An optional watchdog reasserts the wavelength and slit widths.
"""

import argparse
import csv
import ctypes
import struct
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
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
DEFAULT_OPEN_POS = 1

# Serializes DLL access across the REPL, watchdog, and logger threads.
MONO_LOCK = threading.RLock()


def cdouble(x):
    return ctypes.byref(ctypes.c_double(float(x)))


def ok(rc, what):
    if rc == 0:
        return
    err = ctypes.c_int()
    sid = ctypes.create_string_buffer(64)
    addr = ctypes.c_int()
    with MONO_LOCK:
        dll.LOT_get_last_error(ctypes.byref(err), sid, ctypes.byref(addr))
    raise RuntimeError(
        f"{what} failed rc={rc}, last_error={err.value}, "
        f"id='{sid.value.decode(errors='ignore')}', addr={addr.value}"
    )


# Low-level helpers
def lot_get_double(idb: bytes, token: int) -> float:
    v = ctypes.c_double()
    with MONO_LOCK:
        ok(dll.LOT_get(idb, token, 0, ctypes.byref(v)), f"LOT_get({idb!r}, {token})")
    return v.value


def lot_set_double(idb: bytes, token: int, value: float):
    v = ctypes.c_double(float(value))
    with MONO_LOCK:
        ok(
            dll.LOT_set(idb, token, 0, ctypes.byref(v)),
            f"LOT_set({idb!r}, {token}, {value})",
        )


def get_actual_wl() -> float:
    return lot_get_double(b"mono", TOKEN_MONO_CURRENT_WL)


def select_wavelength_nm(wl_nm: float):
    with MONO_LOCK:
        ok(
            dll.LOT_select_wavelength(ctypes.c_double(float(wl_nm))),
            f"LOT_select_wavelength({wl_nm})",
        )


def reselect_current_wl():
    wl = get_actual_wl()
    with MONO_LOCK:
        ok(
            dll.LOT_select_wavelength(ctypes.c_double(wl)),
            f"LOT_select_wavelength({wl})",
        )


def set_bw(idb: bytes, bw_nm: float):
    with MONO_LOCK:
        ok(
            dll.LOT_set(idb, TOKEN_MVSSConstantBandwidth, 0, cdouble(bw_nm)),
            f"{idb!r}: set MVSSConstantBandwidth={bw_nm}",
        )
    reselect_current_wl()


def get_bw(idb: bytes) -> float:
    return lot_get_double(idb, TOKEN_MVSSConstantBandwidth)


def get_filter_position() -> int:
    """Current filter-wheel position (1–6)."""
    pos = lot_get_double(FILTERWHEEL_ID, TOKEN_FWHEEL_POSITION)
    return int(round(pos))


def set_filter_position(pos: int):
    """Move filter wheel to a given slot (1–6)."""
    lot_set_double(FILTERWHEEL_ID, TOKEN_FWHEEL_POSITION, pos)


# Constant-width controller
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

    def clamp_bw(self, bw):
        return max(self.min_bw, min(self.max_bw, bw))

    def tune(self, tol_mm=0.002, max_iter=6) -> tuple[float, float]:
        set_bw(self.idb, self.clamp_bw(self.bw))
        time.sleep(0.05)
        w1 = self.measure_width()
        if abs(self.target - w1) <= tol_mm:
            return (self.bw, w1)

        if self.k is None:
            probe = self.clamp_bw(self.bw + 0.02)
            set_bw(self.idb, probe)
            time.sleep(0.05)
            w2 = self.measure_width()
            dw = w2 - w1
            db = probe - self.bw
            self.k = (dw / db) if abs(db) > 1e-9 else 0.5
            if abs(self.k) < 1e-6:
                self.k = 0.5

        bw = self.bw
        for _ in range(max_iter):
            bw = self.clamp_bw(bw + (self.target - w1) / self.k)
            set_bw(self.idb, bw)
            time.sleep(0.05)
            w2 = self.measure_width()
            if abs(self.target - w2) <= tol_mm:
                self.bw = bw
                return (bw, w2)
            db = bw - self.bw
            if abs(db) > 1e-9:
                new_k = (w2 - w1) / db
                if abs(new_k) > 1e-6:
                    self.k = 0.7 * self.k + 0.3 * new_k
            w1 = w2
            self.bw = bw
        return (bw, w1)


# Keeper
class MonoKeeper:
    def __init__(self, hold_reassert_s: float | None = 0.0):
        self.hold_reassert_s = hold_reassert_s or 0.0
        self._stop = threading.Event()
        self._thread = None
        self.ent_ctl: WidthController | None = None
        self.ext_ctl: WidthController | None = None

        self.shutter_is_closed = False
        self._last_open_filter: int | None = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        last = time.time()
        while not self._stop.is_set():
            if self.shutter_is_closed:
                time.sleep(0.1)
                continue

            if (
                self.hold_reassert_s > 0
                and (time.time() - last) >= self.hold_reassert_s
            ):
                try:
                    reselect_current_wl()
                    if self.ent_ctl:
                        self.ent_ctl.tune()
                    if self.ext_ctl:
                        self.ext_ctl.tune()
                except Exception as e:
                    print(f"[{datetime.now():%H:%M:%S}] Watchdog warn: {e}")
                last = time.time()

            time.sleep(0.1)

    def goto(self, wl_nm: float, tol_nm=0.05, timeout_s=5.0) -> float:
        select_wavelength_nm(wl_nm)

        # Keep the shutter slot selected while changing wavelength.
        if self.shutter_is_closed:
            try:
                self.set_filter(SHUTTER_POS)
            except Exception as e:
                print("Shutter restore failed:", e)

        t0 = time.time()
        while True:
            act = get_actual_wl()
            if abs(act - wl_nm) <= tol_nm or time.time() - t0 > timeout_s:
                return act
            time.sleep(0.05)

    def set_constant_bandwidth(self, ent_bw_nm: float | None, ext_bw_nm: float | None):
        if ent_bw_nm is not None:
            set_bw(b"Ent", ent_bw_nm)
        if ext_bw_nm is not None:
            set_bw(b"EXT", ext_bw_nm)

    def hold_width(self, ent_mm: float | None, ext_mm: float | None, start_bw_nm=0.07):
        if ent_mm is not None:
            self.ent_ctl = self.ent_ctl or WidthController(
                b"Ent", ent_mm, start_bw_nm=start_bw_nm
            )
            self.ent_ctl.target = ent_mm
            be, we = self.ent_ctl.tune()
            print(f"   Entry slit tuned: width={we:.3f} mm via BW={be:.3f} nm")
        if ext_mm is not None:
            self.ext_ctl = self.ext_ctl or WidthController(
                b"EXT", ext_mm, start_bw_nm=start_bw_nm
            )
            self.ext_ctl.target = ext_mm
            bx, wx = self.ext_ctl.tune()
            print(f"   Exit  slit tuned: width={wx:.3f} mm via BW={bx:.3f} nm")

    def status(self):
        try:
            wl = get_actual_wl()
            we = lot_get_double(b"Ent", TOKEN_MVSSCurrentWidth)
            be = get_bw(b"Ent")
            wx = lot_get_double(b"EXT", TOKEN_MVSSCurrentWidth)
            bx = get_bw(b"EXT")
            try:
                fw = get_filter_position()
            except Exception:
                fw = -1
            print(
                f"[{datetime.now():%H:%M:%S}] "
                f"WL={wl:.3f} nm | "
                f"Entry={we:.3f} mm (BW={be:.3f} nm) | "
                f"Exit={wx:.3f} mm (BW={bx:.3f} nm) | "
                f"Filter={fw}"
            )
        except Exception as e:
            print("Status error:", e)

    def set_filter(self, pos: int):
        """Set filter-wheel to a specific position (1–6)."""
        set_filter_position(pos)
        if pos != SHUTTER_POS:
            self._last_open_filter = pos
        print(
            f"[{datetime.now():%H:%M:%S}] Filter wheel → position {pos} "
            f"({'shutter' if pos == SHUTTER_POS else 'open'})"
        )

    def shutter_close(self):
        """Move filter wheel to the shutter (blank disk) position."""
        try:
            cur = int(round(get_filter_position()))
            if cur != SHUTTER_POS:
                self._last_open_filter = cur
        except Exception:
            pass

        self.set_filter(SHUTTER_POS)
        self.shutter_is_closed = True
        print("Shutter CLOSED.")

    def shutter_open(self, pos: int | None = None):
        if self._last_open_filter is None:
            self._last_open_filter = DEFAULT_OPEN_POS
        target = self._last_open_filter if pos is None else pos
        self.set_filter(target)
        self.shutter_is_closed = False
        print(f"Shutter OPEN → filter {target}")


# Keithley 6487
class Keithley6487:
    def __init__(self, serial_port="3", verbosity=1):
        rm = ResourceManager()
        self.dev = rm.open_resource(f"ASRL{serial_port}::INSTR")
        self.dev.read_termination = "\r"
        self.dev.write_termination = "\r"
        self.dev.timeout = 30000
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

    def take_measurement(self, duration_s=2.0, nplc=10, sample_rate=50):
        n_points = max(1, int(duration_s * sample_rate / nplc))
        self.dev.write("*CLS")
        self.dev.write("FORM:ELEM READ,TIME")
        self.dev.write(f"NPLC {int(nplc)}")
        self.dev.write(f"TRIG:COUN {int(n_points)}")
        self.dev.write(f"TRAC:POIN {int(n_points)}")
        self.dev.write("TRAC:CLE")
        self.dev.write("TRAC:FEED:CONT NEXT")
        self.dev.write("TRIG:SOUR IMM")
        self.dev.write("INIT")
        time.sleep(n_points * nplc / sample_rate + 0.5)
        data = self.dev.query_ascii_values("TRAC:DATA?")
        currents = np.array(data[0::2], dtype=float)
        times = np.array(data[1::2], dtype=float)
        return currents, times

    def close(self):
        try:
            self.dev.write("OUTP OFF")
        except Exception:
            pass
        self.dev.close()


# Logger thread
class MeasLogger:
    HEADER = [
        "timestamp",
        "wl_nm",
        "entry_width_mm",
        "exit_width_mm",
        "entry_bw_nm",
        "exit_bw_nm",
        "current_nA",
    ]

    def __init__(
        self,
        keeper: MonoKeeper,
        pico: Keithley6487,
        csv_path: Path,
        interval_s=2.0,
        window_s=2.0,
        nplc=10,
        sample_rate=50,
        echo=True,
    ):
        self.keeper = keeper
        self.pico = pico
        self.csv_path = Path(csv_path)
        self.interval_s = float(interval_s)
        self.window_s = float(window_s)
        self.nplc = int(nplc)
        self.sample_rate = int(sample_rate)
        self.echo = echo
        self._stop = threading.Event()
        self._thread = None
        self._csv_lock = threading.Lock()
        self._csv_file = None
        self._csv_writer = None

    def start(self):
        if self._thread is not None:
            return
        new_file = not self.csv_path.exists()
        self._csv_file = self.csv_path.open("a", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        if new_file:
            self._csv_writer.writerow(self.HEADER)
            self._csv_file.flush()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def _run(self):
        next_t = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now < next_t:
                time.sleep(min(0.05, next_t - now))
                continue
            try:
                currents, _ = self.pico.take_measurement(
                    duration_s=self.window_s,
                    nplc=self.nplc,
                    sample_rate=self.sample_rate,
                )
                mean_current_nA = -float(np.mean(currents)) * 1e9

                wl = get_actual_wl()
                we = lot_get_double(b"Ent", TOKEN_MVSSCurrentWidth)
                wx = lot_get_double(b"EXT", TOKEN_MVSSCurrentWidth)
                be = get_bw(b"Ent")
                bx = get_bw(b"EXT")

                row = [
                    datetime.now().isoformat(timespec="seconds"),
                    f"{wl:.3f}",
                    f"{we:.3f}",
                    f"{wx:.3f}",
                    f"{be:.3f}",
                    f"{bx:.3f}",
                    f"{mean_current_nA:.6f}",
                ]
                with self._csv_lock:
                    self._csv_writer.writerow(row)
                    self._csv_file.flush()

                if self.echo:
                    print(
                        f"[LOG] {row[0]} | WL={wl:.3f} nm | "
                        f"Ent={we:.3f} mm (BW={be:.3f} nm) | "
                        f"Ext={wx:.3f} mm (BW={bx:.3f} nm) | "
                        f"I={mean_current_nA:.3f} nA"
                    )
            except Exception as e:
                print(f"[LOG] warn: {e}")

            next_t += self.interval_s


# Boot + REPL
def initialise():
    buf = ctypes.create_string_buffer(256)
    cfg = ctypes.c_char_p(CONFIG_XML.encode("ascii"))
    ok(dll.LOT_version(buf), "LOT_version")
    ok(dll.LOT_build_system_model(cfg), "LOT_build_system_model")
    ok(dll.LOT_get_comms_list(buf), "LOT_get_comms_list")
    ok(dll.LOT_get_hardware_list(buf), "LOT_get_hardware_list")
    ok(dll.LOT_initialise(), "LOT_initialise")
    print(f"INFO: Monochromator/DLL OK → {buf.value.decode(errors='ignore')}")


def shutdown():
    try:
        dll.LOT_close()
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(
        description="Keep MSH-300 on a wavelength + log picoammeter periodically."
    )
    p.add_argument(
        "--wl", type=float, required=True, help="Target wavelength (nm) to hold."
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--const-bw",
        type=float,
        nargs=2,
        metavar=("ENTRY_BW_NM", "EXIT_BW_NM"),
        help="Use constant bandwidth (nm) for entry/exit (use -1 to skip one).",
    )
    group.add_argument(
        "--const-width",
        type=float,
        nargs=2,
        metavar=("ENTRY_MM", "EXIT_MM"),
        help="Use target physical slit widths (mm) for entry/exit, maintained via constant BW.",
    )
    p.add_argument(
        "--watchdog",
        type=float,
        default=0.0,
        help="Seconds between watchdog re-assertions (0 disables).",
    )

    p.add_argument(
        "--visa-port", default="3", help="ASRL port for Keithley 6487, e.g. '3'."
    )
    p.add_argument(
        "--log-csv",
        type=Path,
        default=Path(f"mono_log_{datetime.now():%Y%m%d_%H%M%S}.csv"),
        help="CSV output path.",
    )
    p.add_argument(
        "--meas-interval",
        type=float,
        default=2.0,
        help="Seconds between logged samples.",
    )
    p.add_argument(
        "--meas-window",
        type=float,
        default=2.0,
        help="Integration window per sample (s).",
    )
    p.add_argument("--nplc", type=int, default=10, help="NPLC for 6487.")
    p.add_argument(
        "--sample-rate",
        type=int,
        default=50,
        help="Pseudo sampling rate to size trace buffer.",
    )
    p.add_argument(
        "--no-echo",
        action="store_true",
        help="Do not print each logged sample to console.",
    )
    return p.parse_args()


def run_command(keeper: MonoKeeper, cmd: str):
    parts = cmd.strip().split()
    if not parts:
        return
    k = parts[0].lower()

    if k in ("quit", "exit"):
        raise SystemExit

    elif k == "wl" and len(parts) == 2:
        wl = float(parts[1])
        act = keeper.goto(wl)
        print(f"→ WL set {wl:.3f} nm (actual {act:.3f} nm)")
        keeper.status()

    elif k == "bw" and len(parts) == 3:
        which, val = parts[1].lower(), float(parts[2])
        if which == "ent":
            keeper.set_constant_bandwidth(val, None)
        elif which == "ext":
            keeper.set_constant_bandwidth(None, val)
        else:
            print("Use: bw ent <nm> | bw ext <nm>")
        keeper.status()

    elif k == "wid" and len(parts) == 3:
        which, val = parts[1].lower(), float(parts[2])
        if which == "ent":
            keeper.hold_width(val, None)
        elif which == "ext":
            keeper.hold_width(None, val)
        else:
            print("Use: wid ent <mm> | wid ext <mm>")
        keeper.status()

    elif k == "status":
        keeper.status()

    elif k == "reselect":
        reselect_current_wl()
        keeper.status()

    elif k == "filter" and len(parts) == 2:
        pos = int(parts[1])
        if not (1 <= pos <= 6):
            print("Filter position must be between 1 and 6.")
        else:
            keeper.set_filter(pos)

    elif k == "shutter" and len(parts) == 2:
        sub = parts[1].lower()
        try:
            if sub == "close":
                keeper.shutter_close()
            elif sub == "open":
                keeper.shutter_open()
            else:
                print("Use: shutter open | shutter close")
        except Exception as e:
            print(f"Shutter command failed: {e}")

    else:
        print("Unknown command.")


def main():
    args = parse_args()

    initialise()
    keeper = MonoKeeper(hold_reassert_s=args.watchdog)
    keeper.start()

    actual = keeper.goto(args.wl)
    print(f"Moved to WL {args.wl:.3f} nm (actual {actual:.3f} nm)")

    if args.const_bw:
        ent_bw, ext_bw = args.const_bw
        keeper.set_constant_bandwidth(
            None if ent_bw < 0 else ent_bw, None if ext_bw < 0 else ext_bw
        )
        keeper.status()
    elif args.const_width:
        ent_mm, ext_mm = args.const_width
        keeper.hold_width(
            None if ent_mm < 0 else ent_mm, None if ext_mm < 0 else ext_mm
        )
        keeper.status()

    pico = Keithley6487(serial_port=args.visa_port, verbosity=1)
    logger = MeasLogger(
        keeper,
        pico,
        args.log_csv,
        interval_s=args.meas_interval,
        window_s=args.meas_window,
        nplc=args.nplc,
        sample_rate=args.sample_rate,
        echo=not args.no_echo,
    )
    logger.start()
    print(f"\nLogging to: {args.log_csv.resolve()}")
    print("\nInteractive commands (type then Enter):")
    print("  wl <nm>                      -> move to wavelength")
    print("  bw ent <nm> | bw ext <nm>    -> set constant bandwidth (nm)")
    print("  wid ent <mm> | wid ext <mm>  -> hold target physical slit width (mm)")
    print("  filter <1-6>                 -> move order-sorting filter wheel")
    print("  shutter open|close           -> open/close shutter (blank filter)")
    print("  status                       -> print WL, widths, BW, and filter")
    print(
        "  reselect                     -> re-assert current wavelength (firmware recalcs)"
    )
    print("  quit                         -> exit")
    print(
        "Tip: chain multiple commands with ';' e.g.  bw ent 2.0 ; bw ext 2.0 ; status\n"
    )

    try:
        while True:
            line = input("mono> ").strip()
            if not line:
                continue
            for chunk in [c for c in line.split(";") if c.strip()]:
                try:
                    run_command(keeper, chunk)
                except SystemExit:
                    raise
    except KeyboardInterrupt:
        print("\n^C received; shutting down…")
    except SystemExit:
        pass
    finally:
        logger.stop()
        pico.close()
        keeper.stop()
        shutdown()
        print("Closed. Bye.")


if __name__ == "__main__":
    main()

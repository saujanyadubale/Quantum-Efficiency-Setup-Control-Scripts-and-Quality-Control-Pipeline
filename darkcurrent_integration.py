import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyvisa import ResourceManager

# Settings
VISA_PORT = "3"
MINUTES = 15.0

SETTLE_S = 1.0
NPLC = 3
SAMPLE_RATE = 50
RANGE_A = 2e-6

# Logging
LOG_INTERVAL_S = 1.0
WINDOW_S = 1.0

# Current sign convention
FLIP_SIGN = True

CURRENT_UNIT = "nA"
CURRENT_SCALE = 1e9


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
        self.dev.write(f"RANG {RANGE_A:.6g}")
        self.dev.write("RANG:AUTO OFF")
        self.dev.write("SYST:ZCH OFF")
        self.dev.write("SYST:ZCOR OFF")
        self.dev.write("SYST:AZER:STAT OFF")
        self.dev.write("SENS:CURR:DAMP:STAT OFF")
        self.dev.write("*CLS")

        if verbosity:
            print("INFO: Connected to", self.dev.query("*IDN?").strip())

    def set_range(self, range_a: float):
        self.dev.write(f"RANG {float(range_a):.6g}")
        self.dev.write("RANG:AUTO OFF")

    def take_measurement(self, duration_s=2.0, nplc=5, sample_rate=3):
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
        times_ = np.array(data[1::2], dtype=float)
        return currents, times_

    def close(self):
        try:
            self.dev.write("OUTP OFF")
        except Exception:
            pass
        self.dev.close()


# Main
if __name__ == "__main__":
    pico = Keithley6487(serial_port=VISA_PORT, verbosity=1)

    try:
        pico.set_range(RANGE_A)

        if SETTLE_S > 0:
            print(f"[INFO] Settling for {SETTLE_S:.1f} s ...")
            time.sleep(SETTLE_S)

        total_s = MINUTES * 60.0
        print(
            f"[INFO] Live logging for {MINUTES:.2f} minutes "
            f"(NPLC={NPLC}, sample_rate={SAMPLE_RATE}, window={WINDOW_S}s, interval={LOG_INTERVAL_S}s) ..."
        )

        rows = []
        t_start = time.time()
        next_t = t_start

        sign = -1.0 if FLIP_SIGN else 1.0

        while True:
            now = time.time()
            t_rel = now - t_start
            if t_rel >= total_s:
                break

            if now < next_t:
                time.sleep(min(0.05, next_t - now))
                continue

            currents_A, _ = pico.take_measurement(
                duration_s=WINDOW_S, nplc=NPLC, sample_rate=SAMPLE_RATE
            )
            currents_nA = (sign * currents_A) * CURRENT_SCALE

            mean_nA = float(np.mean(currents_nA)) if len(currents_nA) else float("nan")
            std_nA = (
                float(np.std(currents_nA, ddof=1))
                if len(currents_nA) >= 2
                else float("nan")
            )
            ts = datetime.now().isoformat(timespec="seconds")

            rows.append(
                {
                    "timestamp": ts,
                    "t_rel_s": float(t_rel),
                    "mean_current_nA": mean_nA,
                    "std_current_nA": std_nA,
                    "n_points_in_window": int(len(currents_nA)),
                    "window_s": WINDOW_S,
                    "interval_s": LOG_INTERVAL_S,
                    "nplc": int(NPLC),
                    "sample_rate": int(SAMPLE_RATE),
                    "range_A": float(RANGE_A),
                }
            )

            print(
                f"[LOG] {ts} | t={t_rel:7.1f} s | I={mean_nA: .4f} ± {std_nA:.4f} nA | N={len(currents_nA)}"
            )

            next_t += LOG_INTERVAL_S

        df = pd.DataFrame(rows)

        overall_mean = float(df["mean_current_nA"].mean()) if len(df) else float("nan")
        overall_std = (
            float(df["mean_current_nA"].std(ddof=1)) if len(df) >= 2 else float("nan")
        )
        print(
            f"\n[INFO] Finished. Overall mean of window-means = {overall_mean:.4f} ± {overall_std:.4f} nA"
        )

        outname = Path(
            f"dark_current_live_ST6061_{datetime.now():%Y%m%d_%H%M%S}_{MINUTES:.0f}min.csv"
        )
        df["overall_mean_window_means_nA"] = overall_mean
        df["overall_std_window_means_nA"] = overall_std
        df.to_csv(outname, index=False)
        print(f"[OK] Saved: {outname.resolve()}")

        plt.figure(figsize=(9, 5))
        plt.plot(df["t_rel_s"], df["mean_current_nA"], marker=".", linewidth=1.0)
        plt.xlabel("Time (s)")
        plt.ylabel(f"Current ({CURRENT_UNIT})")
        plt.title(
            f"Dark current vs time ({MINUTES:.0f} min) | mean={overall_mean:.3f} nA"
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    finally:
        try:
            pico.close()
        except Exception:
            pass

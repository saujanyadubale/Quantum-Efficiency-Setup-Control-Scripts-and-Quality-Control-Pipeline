# ANTARES PMT Quantum-Efficiency Measurement Scripts

This repository contains the control, data-acquisition, diagnostic, and quality-control scripts developed for the quantum-efficiency measurement setup used to characterize extracted ANTARES photomultiplier tubes.

The scripts were used to control the LOT MSH-300 monochromator, acquire photocurrent measurements with a Keithley 6487 picoammeter, operate the Thorlabs HDR50/M motorized rotation stage during angular scans, investigate PMT dark-current stabilization, and perform quality-control analysis of the measured quantum-efficiency spectra.

## Repository Contents

### `DLLToken.txt`

Contains the hardware-control token definitions required to communicate with the LOT monochromator control library.

The tokens identify monochromator parameters such as:

* wavelength;
* grating selection;
* slit width;
* constant bandwidth;
* filter-wheel position;
* monochromator and hardware status.


### `ccgData_LOT_MSH-300_SN38594.xml`

Configuration file describing the LOT MSH-300 monochromator system used in the measurement setup.

The XML file is loaded by the control scripts when initializing the monochromator hardware and its associated components.


### `mono_keeper_with_pico.py`

Diagnostic and interactive control script for the LOT MSH-300 monochromator and Keithley 6487 picoammeter.

The script can:

* select and maintain a monochromator wavelength;
* monitor and adjust the entrance and exit slit widths;
* control the filter wheel and shutter;
* display the monochromator settings in real time;
* acquire and display real-time picoammeter current measurements;

This script was mainly used for debugging the monochromator settings and observing real-time changes in the measured photocurrent.


### `darkcurrent_integration.py`

Diagnostic script used to study the stabilization of the PMT dark current over time.

The script continuously records the Keithley 6487 current, displays the measurements in real time, saves the data to a CSV file, and produces a dark-current-versus-time plot. It was used to determine an appropriate stabilization and integration period before spectral measurements.


### `Dark_Multiseg_ConstSlit.py`

Control and data-acquisition script used for general wavelength-dependent photocurrent measurements.

The script coordinates:

* the LOT MSH-300 monochromator;
* the Keithley 6487 picoammeter;
* wavelength-dependent slit-width control;
* shutter and filter-wheel operation;
* pre-scan and post-scan dark-current measurements;

It was used to perform multi-segment spectral scans with different slit widths in selected wavelength regions.


### `angularphotocurrent.py`

Control and data-acquisition script used for angular wavelength-dependent photocurrent measurements.

The script coordinates:

* the LOT MSH-300 monochromator;
* the Keithley 6487 picoammeter;
* the motorized PMT rotation stage;
* wavelength-dependent slit-width control;
* shutter and filter-wheel operation;
* pre-scan and post-scan dark-current measurements;
* automatic data saving and recovery in case of interruption.

It was used to measure the PMT photocurrent as a function of wavelength and photocathode rotation angle.


### `anamolystart.py`

Hybrid quality-control pipeline for the measured PMT quantum-efficiency spectra.

The pipeline:

* loads corrected or uncorrected QE spectra;
* interpolates the spectra onto a common wavelength grid;
* constructs a population median and median-absolute-deviation reference;
* extracts spectral-shape features, including boundary deviations, discontinuities, curvature, roughness, peak wavelength, FWHM, slopes, and tail behaviour;
* applies PCA reconstruction analysis and Isolation Forest anomaly detection;
* evaluates low-QE or potentially degraded spectra separately from measurement-shape anomalies;
* calculates shape, degradation, and QC-priority scores;
* assigns the spectra to `GOLD`, `SILVER`, `BRONZE`, or `REVIEW` quality categories;
* generates full anomaly tables, ranked leaderboards, and diagnostic plots.

  
## Hardware and Software

The control scripts were developed for the following laboratory equipment:

* LOT MSH-300 monochromator;
* Keithley 6487 picoammeter;
* motorized rotation stage using a Trinamic controller;
* Windows monochromator control DLL supplied with the LOT system.

The Python scripts use packages including:

```text
numpy
pandas
matplotlib
scipy
scikit-learn
pyvisa
pytrinamic
```

## Configuration

Several scripts contain setup-specific paths and communication settings, including:

* monochromator DLL locations;
* monochromator XML configuration-file path;
* Keithley serial port;
* rotation-stage COM port;
* input and output directories.

These settings must be updated before running the scripts on another computer or measurement setup.

For example:

```python
DLL64 = r"C:\Program Files (x86)\QD\Monochromator Control\LotHW64.dll"
CONFIG_XML = r"path\to\ccgData_LOT_MSH-300_SN38594.xml"
KEITHLEY_SERIAL_PORT = "3"
ROT_STAGE_PORT = "COM5"
```

The input and output directories in `gooddetect.py` must also be changed to match the location of the QE data files.

## Measurement Data

The repository contains the control and analysis software used in the thesis. Large raw measurement datasets and hardware-specific proprietary DLL files are not included.

## Thesis Context

These scripts were developed as part of a study investigating the wavelength-dependent and angular quantum-efficiency response of Hamamatsu R7081-20 photomultiplier tubes recovered from the ANTARES neutrino telescope and considered for possible reuse in the Southern Wide-field Gamma-ray Observatory.

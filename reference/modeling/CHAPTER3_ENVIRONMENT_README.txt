CHAPTER 3 ENVIRONMENT - INITIAL WORKING IMPLEMENTATION

SUPERSEDED TIMELINE NOTICE - 2026-10-31
This file describes historical v1 behavior, which has a confirmed cross-slot
FIFO defect. The corrected implementation is chapter3_iov_env_v2.py.
Read CHAPTER3_TIMELINE_V2_README.txt and run validate_chapter3_timeline_v2.py.
The v1 code remains for historical diagnostics, not new thesis evidence.

Files
- chapter3_iov_env.py: Standalone environment scaffold for the Chapter 3 specification.
- test_chapter3_iov_env.py: Deterministic checks for task units, destination choices, queues, radio delay, energy, movement, and seeded repeatability.

Run the checks from this directory
    python -m unittest -v test_chapter3_iov_env.py
    python validate_chapter3_model.py

The environment does not import or replace ddqfinaltraining.py. It has no cloud destination, no charging action, and no trust/attack terms. It models one task per vehicle per one-second slot, indivisible task assignment, local/UAV/RSU execution, separate local and server FIFO availability, horizontal UAV movement at fixed altitude, per-task deadlines, and per-task result records. The default G2A mode gives each active link the reference's 20 MHz bandwidth and models co-channel interference from other tasks assigned to the same UAV. A global equal-share/no-interference mode remains available only for comparison.

Reference-derived defaults
- Hao et al., IEEE Transactions on Mobile Computing, 2024, DOI 10.1109/TMC.2024.3350078: 1-3 MB input, 300-500 Megacycles, 1.5 Gcycles/s UE CPU, 10-20 Gcycles/s UAV capacity, 250-300 ms deadline range, 20 MHz G2A bandwidth, -100 dBm noise, and 49 m horizontal movement limit per slot.
- Zeng, Y., Xu, J., and Zhang, R., "Energy Minimization for Wireless Communication with Rotary-Wing UAV," IEEE Transactions on Wireless Communications, vol. 18, no. 4, pp. 2329-2345, April 2019, DOI 10.1109/TWC.2019.2902559; preprint arXiv:1804.02238: rotary-wing propulsion power equation and generic rotor constants. The parameters are not calibrated to a named vehicle.

Explicit provisional assumptions
- Two RSUs at quarter/three-quarter positions in the 1000 m square.
- 20 Gcycles/s capacity per RSU and a 1 Gbit/s fixed RSU link.
- 2 GHz carrier frequency for the vehicle-UAV channel.
- 100 N UAV weight and the generic rotor geometry from the Zeng model.
- Vehicle CPU effective switched capacitance of 1e-28 J*s^3, used only as a provisional energy assumption.
- The default link mode is a simplified interference model: fixed UE transmit power, per-link 20 MHz, and interference only among same-UAV offloads. It omits cross-UAV interference, power control, and explicit UAV coverage association.
- The comparison mode `global_equal_share` divides total 20 MHz among active UAV links and omits interference. It is not the reference equation and should not be used as the primary radio model.
- Four movement choices plus hover are cardinal horizontal moves, with at most 49 m displacement per one-second step.

These values are exposed in Chapter3Config and surfaced in environment info where relevant. They must remain labeled assumptions until replaced by system-specific data or supported by references. Use sensitivity runs for the RSU CPU and link values before drawing conclusions that depend on RSU performance.

The environment is not yet connected to a DDQN training entrypoint, and no policy training has been run. The fixed-policy and sensitivity checks in `validate_chapter3_model.py` are diagnostic only and are not evidence of a learned policy. It includes a UAV-selective heuristic (one nearest distinct vehicle per UAV, remaining tasks to nearest RSUs) for comparison. Review `CHAPTER3_MODEL_VALIDATION_2026-10-31.txt` before using any outputs. The radio model, RSU assumptions, reward weights, and aircraft calibration need further review before training.

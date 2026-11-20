"""Fast end-to-end smoke test: small stream, calibration, and the main policies (about 30 s)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dcsa.stream import (StreamCfg, make_stream, run, calib_arrays, dcsa_offline, Threshold,
                         FastestMean, DCSAOnline)


def test_smoke():
    cfg = StreamCfg(n_jobs=300, rho=0.15, warmup=50)
    cal = make_stream(cfg, 0, role=1)
    rc = run(cal, Threshold(0.2, 1.0, "ref"), record_calib=True)
    S, NOM, E, R, D = calib_arrays(rc["calib"], warmup=50)
    lam, rows = dcsa_offline(S, E, R, D, alpha=0.2)
    assert rows.shape == (40, 4) and lam >= 0
    test = make_stream(cfg, 0, role=0)
    for pol in (FastestMean(), Threshold(0.2, 1.0, "CC"), DCSAOnline(0.2, lam if lam > 0 else 1.0, 0.02)):
        r = run(test, pol)
        assert 0 <= r["sel_risk"] <= 1 and 0 <= r["admit_rate"] <= 1
    print("ok")


if __name__ == "__main__":
    test_smoke()

"""Unit test of the Alibaba batch_task parser on a tiny hand-written fixture in the trace format
(the fixture is NOT trace data)."""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dcsa.alibaba import parse_batch_task, to_jobs

FIXTURE = """M1,10,j_1,1,Terminated,100,110,100,0.5
M2_1,5,j_1,1,Terminated,110,130,50,0.5
R3_1,2,j_1,1,Terminated,110,115,100,0.5
J4_2_3,1,j_1,1,Terminated,130,140,100,0.5
task_abc,1,j_2,1,Terminated,105,106,100,0.5
M1,3,j_3,1,Terminated,200,220,200,0.2
R2_1,3,j_3,1,Terminated,220,230,100,0.2
M1,3,j_4,1,Failed,200,220,200,0.2
"""

def test_parse():
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(FIXTURE)
    recs = parse_batch_task(f.name)
    assert [r["name"] for r in recs] == ["j_1", "j_3"]
    j1 = recs[0]
    assert j1["n"] == 4 and sorted(j1["edges"]) == [(0, 1), (0, 2), (1, 3), (2, 3)]
    assert np.isclose(j1["cpu_s"][0], 10 * 10 * 1.0) and np.isclose(j1["cpu_s"][1], 5 * 20 * 0.5)
    jobs = to_jobs(recs, np.random.default_rng(0))
    assert len(jobs) == 2 and jobs[0].n == 4 and list(jobs[0].preds[3]) == [1, 2]
    os.unlink(f.name)
    print("ok")

if __name__ == "__main__":
    test_parse()

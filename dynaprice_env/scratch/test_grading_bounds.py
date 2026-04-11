import math
import sys
import os

# Add parent directory to path to import server
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.grading import _squash_to_open_interval

def test_bounds():
    test_cases = [
        (0.0, 1.0),
        (1000.0, 1.0),
        (-1000.0, 1.0),
        (1e308, 1.0),
        (-1e308, 1.0),
        (float('nan'), 1.0),
        (float('inf'), 1.0),
        (float('-inf'), 1.0),
        (1.0, 1e-10),
        (0.0000000001, 1.0), # nearly zero
    ]
    
    print(f"{'Signal':>10} | {'Scale':>10} | {'Result':>10}")
    print("-" * 40)
    for sig, scale in test_cases:
        try:
            res = _squash_to_open_interval(sig, scale)
            print(f"{str(sig):>10} | {str(scale):>10} | {res:>10.6f}")
            assert 0 < res < 1, f"Failed for {sig}, {scale}: {res} not in (0, 1)"
            assert res != 0.0 and res != 1.0, f"Failed for {sig}, {scale}: {res} is boundary"
        except Exception as e:
            print(f"Error testing {sig}, {scale}: {e}")

if __name__ == "__main__":
    test_bounds()

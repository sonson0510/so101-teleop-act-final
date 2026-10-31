import unittest
import time
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teleop_core import CommandLimiter, CommandRunner, MockBackend
from evaluation import score


class ContractTest(unittest.TestCase):
    def test_limit_and_invalid_action(self):
        limiter = CommandLimiter([-1] * 6, [1] * 6, [0.1] * 6)
        action, limited = limiter.apply([2] * 6, [0] * 6)
        self.assertTrue(limited)
        np.testing.assert_allclose(action, [0.1] * 6)
        with self.assertRaises(ValueError):
            limiter.apply([np.nan] * 6, [0] * 6)
        with self.assertRaises(ValueError):
            limiter.apply([0] * 5, [0] * 6)

    def test_mock_reject_disconnect_and_no_reset(self):
        backend = MockBackend()
        backend.connect()
        self.assertEqual(backend.observe().image.shape, (240, 320, 3))
        self.assertTrue(backend.send_action(np.ones(6)).accepted)
        backend.reject_next = True
        rejected = backend.send_action(np.zeros(6))
        self.assertFalse(rejected.accepted)
        self.assertEqual(len(backend.sent), 1)
        self.assertFalse(hasattr(backend, "reset"))
        backend.disconnect()
        self.assertFalse(backend.send_action(np.zeros(6)).accepted)
        with self.assertRaises(ConnectionError):
            backend.observe()

    def test_runner_authority_stop_reject_and_disconnect(self):
        backend = MockBackend()
        runner = CommandRunner(backend, CommandLimiter([-1] * 6, [1] * 6, [.1] * 6))
        runner.connect()
        self.assertFalse(runner.submit(np.ones(6), source="policy").accepted)
        limited = runner.submit(np.ones(6))
        self.assertTrue(limited.accepted and limited.limited)
        backend.reject_next = True
        self.assertFalse(runner.submit(np.zeros(6)).accepted)
        runner.stop()
        self.assertFalse(runner.submit(np.zeros(6)).accepted)
        runner.resume()
        self.assertTrue(runner.submit(np.zeros(6)).accepted)
        backend.disconnect()
        self.assertFalse(runner.submit(np.zeros(6)).accepted)
        self.assertFalse(runner.connected)

    def test_mock_delay_is_observable(self):
        backend = MockBackend(delay_s=0.01)
        runner = CommandRunner(backend, CommandLimiter([-1] * 6, [1] * 6, [.1] * 6))
        runner.connect()
        start = time.perf_counter()
        result = runner.submit(np.zeros(6))
        self.assertTrue(result.accepted)
        self.assertGreaterEqual(result.sent_at - start, 0.009)

    def test_success_requires_lift_place_stability_and_retreat(self):
        obj = np.tile([0.20, 0.12, 0.015], (30, 1)).astype(float)
        obj[0] = [0.25, 0.0, 0.015]
        obj[1:5, 2] = 0.06
        grip = obj.copy()
        grip[-1] = [0.20, 0.12, 0.12]
        self.assertTrue(score(obj, grip, gripper_open=True)["success"])
        self.assertFalse(score(obj, grip, gripper_open=False)["success"])
        grip[-1] = obj[-1]
        self.assertFalse(score(obj, grip, gripper_open=True)["success"])
        obj[-1, 0] = 0.3
        self.assertFalse(score(obj, grip, gripper_open=True)["success"])


if __name__ == "__main__":
    unittest.main()


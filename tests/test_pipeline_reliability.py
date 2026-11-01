"""Regression checks for provider failures that previously stalled the monitor."""
import unittest
from unittest.mock import patch

from scripts.hf_upload import _retryable
from scripts.push_tactical_monitor import quota_snapshot


class HubError(Exception):
    response = type("Response", (), {"status_code": 400})()


class PipelineReliabilityTests(unittest.TestCase):
    def test_only_missing_lfs_object_bad_request_is_retried(self):
        self.assertTrue(_retryable(HubError(
            "LFS pointer pointed to a file that does not exist")))
        self.assertFalse(_retryable(HubError("invalid checkpoint identity")))

    def test_quota_failure_is_visible_without_staling_snapshot(self):
        def quota(account):
            if account == "shoumikmitra":
                raise RuntimeError("Kaggle unavailable")
            return 8.5

        with patch("scripts.push_tactical_monitor.gpu_quota", side_effect=quota):
            quotas, warnings = quota_snapshot()

        self.assertEqual(quotas["vedanggggg"], 8.5)
        self.assertIsNone(quotas["shoumikmitra"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("shoumikmitra: GPU quota unavailable", warnings[0])


if __name__ == "__main__":
    unittest.main()

"""Scheduling decisions of the sequential cont1 supervisor (no network)."""
import unittest

from scripts.cont_config import SESSION_SECONDS
from scripts.watch_cont import decide, latest_step, strip_status

HEAD, OTHER = "a" * 40, "b" * 40
NOW = 1_000_000.0


def acct(status="COMPLETE", quota=30.0):
    return {"status": f'x has status "KernelWorkerStatus.{status}"', "quota": quota}


def pool(**over):
    base = {"vedanggggg": acct(quota=27.0), "vedangpandeyyy": acct(quota=19.0), "softmaxsimp": acct(quota=29.9),
            "samaltmannnn": acct(quota=12.0), "shoumikmitra": acct(quota=28.0)}
    base.update(over)
    return base


def go(**kw):
    args = {"run_status": "IN PROGRESS checkpoint-90000; session time reached", "accounts": pool(),
            "head": HEAD, "marker": None, "step": 90_000, "now": NOW}
    args.update(kw)
    return decide(**args)


class DecideTests(unittest.TestCase):
    def test_launches_on_the_account_with_most_quota_for_a_full_session(self):
        d = go()
        self.assertEqual((d["action"], d["account"]), ("launch", "softmaxsimp"))
        self.assertEqual(d["session_seconds"], SESSION_SECONDS)       # 11.5 h

    def test_session_is_shortened_to_leave_quota_headroom(self):
        d = go(accounts={a: acct(quota=5.0) for a in pool()})
        self.assertEqual(d["action"], "launch")
        self.assertEqual(d["session_seconds"], int((5.0 - 0.4) * 3600))

    def test_waits_for_the_weekly_reset_when_no_account_can_run_an_hour(self):
        d = go(accounts={a: acct(quota=1.0) for a in pool()})
        self.assertEqual(d["action"], "wait")

    def test_never_starts_a_second_learner(self):
        d = go(accounts=pool(vedanggggg=acct("RUNNING")))
        self.assertEqual(d["action"], "noop")
        d = go(accounts=pool(samaltmannnn=acct("QUEUED")))
        self.assertEqual(d["action"], "noop")

    def test_unknown_kaggle_status_blocks_launching(self):
        d = go(accounts=pool(vedanggggg={"status": None, "quota": 30.0}))
        self.assertEqual(d["action"], "noop")

    def test_stalled_learner_is_flagged_not_killed(self):
        marker = {"commit": HEAD, "step": 90_000, "attempt": 1, "at_epoch": NOW - 4 * 3600}
        d = go(accounts=pool(vedanggggg=acct("RUNNING")), marker=marker)
        self.assertEqual(d["action"], "noop")
        self.assertTrue(d["stalled"])

    def test_cooldown_after_a_launch(self):
        marker = {"commit": HEAD, "step": 90_000, "attempt": 1, "at_epoch": NOW - 60}
        self.assertEqual(go(marker=marker)["action"], "noop")

    def test_done_and_halted_runs_are_never_relaunched(self):
        self.assertEqual(go(run_status="DONE step=30000 max_steps reached")["action"], "complete")
        self.assertEqual(go(run_status="# 2026-11-02T00:00:00Z\nHALTED step=40000: regression")["action"],
                         "blocked")

    def test_crash_without_progress_blocks_on_the_same_commit_only(self):
        marker = {"commit": HEAD, "step": 90_000, "attempt": 1, "at_epoch": NOW - 7200}
        err = "# 2026-11-02T00:00:00Z\nValueError: boom\nTraceback"
        self.assertEqual(go(run_status=err, marker=marker)["action"], "blocked")
        marker["commit"] = OTHER           # a new commit (a fix) gets a fresh chance
        self.assertEqual(go(run_status=err, marker=marker)["action"], "launch")

    def test_crash_after_progress_is_retried(self):
        marker = {"commit": HEAD, "step": 80_000, "attempt": 1, "at_epoch": NOW - 7200}
        err = "# 2026-11-02T00:00:00Z\nRuntimeError: transient"
        self.assertEqual(go(run_status=err, marker=marker, step=90_000)["action"], "launch")

    def test_silent_crash_loop_stops_after_two_attempts(self):
        marker = {"commit": HEAD, "step": 90_000, "attempt": 1, "at_epoch": NOW - 7200}
        d = go(marker=marker)                         # attempt 2 is allowed
        self.assertEqual((d["action"], d["attempt"]), ("launch", 2))
        marker["attempt"] = 2
        self.assertEqual(go(marker=marker)["action"], "blocked")

    def test_status_header_and_checkpoint_discovery(self):
        self.assertEqual(strip_status("# 2026-11-02T00:00:00Z\nFAILED x"), "FAILED x")
        self.assertEqual(strip_status("IN PROGRESS checkpoint-5"), "IN PROGRESS checkpoint-5")
        files = {"r/checkpoint-5/state.pt", "r/checkpoint-5/config.json",
                 "r/checkpoint-9/state.pt",                       # upload interrupted: no config
                 "other/checkpoint-99/state.pt", "other/checkpoint-99/config.json"}
        self.assertEqual(latest_step(files, "r"), 5)


if __name__ == "__main__":
    unittest.main()

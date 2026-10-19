"""Only definitive billing rejections can be cleared for an explicit resume."""
import json
from pathlib import Path
import tempfile
import unittest

import resume_distillation_v8 as resume


class ResumeTests(unittest.TestCase):
    def test_billing_journal_is_moved_verbatim_to_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"batch_jobs/batch_047.json"
            path.parent.mkdir()
            resume.batch.io.write_json(path,dict(status="submission_unresolved",request_id="request-test",
                error='API HTTP 400: {"error":{"code":"billing_hard_limit_reached"}}'))
            original = path.read_bytes()
            self.assertTrue(resume.recover_billing_journal(path))
            self.assertFalse(path.exists())
            archive = next((Path(tmp)/"queue_history").glob("*.json"))
            self.assertEqual(archive.read_bytes(),original)

    def test_timeout_or_other_http_error_cannot_be_blindly_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"batch_jobs/batch_047.json"
            path.parent.mkdir()
            for error in ('API TimeoutError; outcome may be unknown',
                          'API HTTP 400: {"error":{"code":"invalid_request"}}'):
                resume.batch.io.write_json(path,dict(status="submission_unresolved",request_id="test",error=error))
                with self.assertRaises(ValueError):
                    resume.recover_billing_journal(path)
                self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()

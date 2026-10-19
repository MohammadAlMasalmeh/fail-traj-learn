"""Capacity rejection is the only safe automatic re-enqueue condition."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import queue_batch_trajectories as queue


class QueueTests(unittest.TestCase):
    def rejection(self):
        return dict(id="batch-rejected",status="failed",input_file_id="file-test",
                    errors=dict(data=[dict(code="token_limit_exceeded")]),
                    request_counts=dict(total=0,completed=0,failed=0))

    def test_only_zero_execution_capacity_rejections_are_retryable(self):
        remote = self.rejection()
        self.assertTrue(queue.capacity_rejection(remote))
        for mutation in (dict(status="in_progress"),dict(errors=dict(data=[dict(code="invalid_request")])),
                         dict(request_counts=dict(completed=1,failed=0)),
                         dict(request_counts=dict(completed=0,failed=1))):
            self.assertFalse(queue.capacity_rejection(dict(remote,**mutation)))

    def test_active_batch_blocks_next_enqueue(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/"batch_jobs").mkdir()
            shard = dict(path="batch_inputs/batch_001.jsonl",custom_ids=["D1_sol"])
            queue.batch.io.write_json(out/"batch_jobs/batch_001.json",dict(batch=dict(status="in_progress")))
            with patch.object(queue.batch,"api") as api:
                self.assertFalse(queue.enqueue_next(out,dict(shards=[shard]),"test"))
                api.assert_not_called()

    def test_requeue_preserves_rejection_and_reuses_uploaded_exact_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/"batch_jobs").mkdir()
            shard = dict(path="batch_inputs/batch_001.jsonl",custom_ids=["D1_sol"])
            state = dict(batch=self.rejection(),custom_ids=shard["custom_ids"])
            queue.batch.io.write_json(out/"batch_jobs/batch_001.json",state)
            queue.batch.io.write_json(out/"batch_jobs/batch_001_upload.json",dict(id="file-test"))
            with patch.object(queue.batch,"api",return_value=dict(id="batch-next",status="validating")) as api, patch.object(
                queue.batch,"upload"
            ) as upload:
                self.assertTrue(queue.enqueue_next(out,dict(shards=[shard],run_id="run",endpoint="/v1/responses"),"test"))
            upload.assert_not_called()
            self.assertEqual(api.call_args.args[3]["input_file_id"],"file-test")
            self.assertEqual(queue.batch.io.load(out/"queue_history/batch_001_batch-rejected.json"),state)
            self.assertEqual(queue.batch.io.load(out/"batch_jobs/batch_001.json")["batch"]["id"],"batch-next")


if __name__ == "__main__":
    unittest.main()

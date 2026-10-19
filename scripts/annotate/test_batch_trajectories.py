"""Offline checks for paid Batch deduplication, provenance and output validation."""
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import annotation_runtime as io
import batch_trajectories as batch
import prepare_distillation_v8 as selection
from test_annotate_trajectories import EP, annotation, segment


class BatchTests(unittest.TestCase):
    def fixture(self, out, count=2):
        episodes = [dict(EP, id=f"D{i:04d}") for i in range(count)]
        source = dict(episodes=episodes)
        shards = [dict(path="batch_inputs/batch_001.jsonl", custom_ids=[e["id"]+"_sol" for e in episodes])]
        frozen = dict(run_id="test-run", endpoint="/v1/responses", planned=count, shards=shards)
        io.write_json(out/"batch_manifest.json", frozen)
        return source, frozen

    def test_shards_preserve_exact_requests_and_remove_explicit_tier(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/"manifest.json").write_text("{}")
            (out/"prompt.txt").write_text("test prompt")
            source = dict(episodes=[dict(EP,id=f"D{i}") for i in range(3)])
            body = dict(model="gpt-6.1-sol", input=[{"evidence":"unchanged"}], service_tier="default")
            with patch.object(batch,"configure"), patch.object(batch.annotation,"verify",return_value=source), patch.object(
                batch,"V8_SHA256",io.digest(out/"prompt.txt")
            ), patch.object(batch.annotation,"payload",return_value=body):
                frozen = batch.freeze_shards(out,max_requests=2)
            self.assertEqual([len(s["custom_ids"]) for s in frozen["shards"]],[2,1])
            rows = [json.loads(line) for s in frozen["shards"] for line in (out/s["path"]).read_text().splitlines()]
            self.assertEqual([r["custom_id"] for r in rows],["D0_sol","D1_sol","D2_sol"])
            for row in rows:
                self.assertNotIn("service_tier",row["body"])
                self.assertEqual(row["body"]["input"],body["input"])

    def test_second_submit_does_not_buy_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source,frozen = self.fixture(out)
            with patch.object(batch,"verify",return_value=(source,frozen)), patch.object(
                io,"credentials",return_value={"openai":"test-key"}
            ), patch.object(batch,"upload",return_value={"id":"file-test"}) as upload, patch.object(
                batch,"api",return_value={"id":"batch-test","status":"validating"}
            ) as api:
                batch.submit(out)
                batch.submit(out)
            self.assertEqual(upload.call_count,1)
            self.assertEqual(api.call_count,1)

    def test_unknown_create_outcome_never_resubmits(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source,frozen = self.fixture(out)
            with patch.object(batch,"verify",return_value=(source,frozen)), patch.object(
                io,"credentials",return_value={"openai":"test-key"}
            ), patch.object(batch,"upload",return_value={"id":"file-test"}), patch.object(
                batch,"api",side_effect=RuntimeError("network interrupted")
            ) as api:
                with self.assertRaisesRegex(RuntimeError,"interrupted"):
                    batch.submit(out)
                with self.assertRaisesRegex(ValueError,"unresolved"):
                    batch.submit(out)
            self.assertEqual(api.call_count,1)

    def test_lost_create_reply_can_be_reconciled_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _,frozen = self.fixture(out)
            jobs = out/"batch_jobs"
            jobs.mkdir()
            io.write_json(jobs/"batch_001.json",dict(input_file_id="file-test",status="submission_unresolved"))
            remote = dict(id="batch-found",input_file_id="file-test",status="in_progress",
                          metadata=dict(run_id="test-run",shard="batch_001"))
            with patch.object(batch,"api",return_value=dict(data=[remote],has_more=False)) as api:
                batch.reconcile(out,frozen,"test-key")
            self.assertEqual(api.call_args.args[0],"GET")
            self.assertEqual(io.load(jobs/"batch_001.json")["batch"]["id"],"batch-found")

    def test_request_tampering_is_rejected_before_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/"manifest.json").write_text("{}")
            (out/"prompt.txt").write_text("test prompt")
            source = dict(episodes=[dict(EP,id="D0")])
            body = dict(model="gpt-6.1-sol",input=[])
            with patch.object(batch,"configure"), patch.object(batch.annotation,"verify",return_value=source), patch.object(
                batch,"V8_SHA256",io.digest(out/"prompt.txt")
            ), patch.object(batch.annotation,"payload",return_value=body):
                frozen = batch.freeze_shards(out)
                batch.verify(out)
                with (out/frozen["shards"][0]["path"]).open("ab") as stream:
                    stream.write(b"\n")
                with self.assertRaisesRegex(ValueError,"bytes changed"):
                    batch.verify(out)

    def test_out_of_order_results_keep_correct_episode_identity_and_batch_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source,frozen = self.fixture(out)
            results = out/"batch_results"
            results.mkdir()
            lines = []
            for ep in reversed(source["episodes"]):
                value = annotation([segment(0,20,"a1","transport")])
                value["episode_id"] = ep["id"]
                body = dict(model="gpt-6.1-sol",status="completed",
                            usage=dict(input_tokens=1000,output_tokens=100),
                            output=[dict(content=[dict(type="output_text",text=json.dumps(value))])])
                lines.append(dict(custom_id=ep["id"]+"_sol",response=dict(status_code=200,body=body),error=None))
            (results/"batch_001_output.jsonl").write_text("\n".join(json.dumps(v) for v in lines))
            with patch.object(batch.annotation,"LABEL_MODE","failure_binary"), patch.object(batch.annotation,"SCHEME","atomic"):
                batch.ingest(out,source,frozen)
            summary = batch.report(out)
            self.assertEqual(summary["validated"],2)
            self.assertAlmostEqual(summary["generation_estimate_usd"],.003)
            for ep in source["episodes"]:
                self.assertEqual(io.load(out/(ep["id"]+"_sol_annotation.json"))["episode_id"],ep["id"])
            # A duplicate row must never silently overwrite or count twice.
            with (results/"batch_001_output.jsonl").open("a") as stream:
                stream.write("\n"+json.dumps(lines[0]))
            with self.assertRaisesRegex(ValueError,"duplicate"):
                batch.ingest(out,source,frozen)

    def test_invalid_or_incomplete_response_is_retained_and_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source,frozen = self.fixture(out,1)
            results = out/"batch_results"
            results.mkdir()
            body = dict(model="gpt-6.1-sol",status="incomplete",usage=dict(input_tokens=1000,output_tokens=100),output=[])
            row = dict(custom_id="D0000_sol",response=dict(status_code=200,body=body),error=None)
            (results/"batch_001_output.jsonl").write_text(json.dumps(row))
            batch.ingest(out,source,frozen)
            record = io.load(results/"D0000_sol_record.json")
            self.assertEqual(record["status"],"stopped")
            self.assertEqual(record["generation_estimate_usd"],.0015)
            self.assertTrue((out/"D0000_sol_raw.json").exists())
            self.assertFalse((out/"D0000_sol_annotation.json").exists())

    def test_selection_is_reproducible_balanced_and_episode_disjoint(self):
        rows = [dict(dataset=f"d{i%3}",episode_index=i,task=f"task{i%4}",recorded_success=i%2==0) for i in range(100)]
        first = selection.balanced_take(rows,20,random.Random(7))
        self.assertEqual(first,selection.balanced_take(rows,20,random.Random(7)))
        self.assertEqual(sum(r["recorded_success"] for r in first),10)
        used = {(r["dataset"],r["episode_index"]) for r in first}
        second = selection.balanced_take([r for r in rows if (r["dataset"],r["episode_index"]) not in used],20,random.Random(8))
        self.assertFalse(used & {(r["dataset"],r["episode_index"]) for r in second})


if __name__ == "__main__":
    unittest.main()

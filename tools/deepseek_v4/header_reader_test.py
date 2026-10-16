"""Protocol preflight: header-only access, malformed bounds, and index completeness."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

from header_reader import gguf, safetensors
from model_inventory import inspect


class HeaderProtocolTest(unittest.TestCase):
    def setUp(self):
        # Caller must run from the task scratch directory, including on Windows.
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def safetensors_file(self, text, payload_size=1024 * 1024):
        path = self.root / "model.safetensors"
        raw = text.encode()
        with path.open("wb") as handle:
            handle.write(struct.pack("<Q", len(raw)))
            handle.write(raw)
            handle.truncate(8 + len(raw) + payload_size)
        return path, 8 + len(raw)

    def test_only_header_is_read(self):
        path, header_size = self.safetensors_file(json.dumps({
            "layers.0.attn_norm.weight": {"dtype": "BF16", "shape": [4096], "data_offsets": [0, 8192]}}))
        result = safetensors(path)
        self.assertEqual(result["header_bytes_read"], header_size)
        self.assertEqual(result["tensors"][0]["stored_bytes"], 8192)

    def test_bounds_and_duplicate_keys(self):
        path, _ = self.safetensors_file('{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,9999999]}}')
        with self.assertRaisesRegex(ValueError, "invalid tensor descriptor"):
            safetensors(path)
        path, _ = self.safetensors_file('{"x":{},"x":{}}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            safetensors(path)

    def test_missing_shard_is_reported(self):
        self.safetensors_file('{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}')
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"x": "model.safetensors", "y": "missing.safetensors"}}))
        output = self.root.parent / (self.root.name + ".inventory.json")
        try:
            inspect(self.root, output)
            report = json.loads(output.read_text())
            self.assertEqual(report["tensor_payload_bytes_read"], 0)
            self.assertEqual(report["index_check"]["missing_tensors"], 1)
            self.assertIn("missing indexed shard: missing.safetensors", report["issues"])
        finally:
            output.unlink(missing_ok=True)
            output.with_name(output.stem + ".tensors.jsonl").unlink(missing_ok=True)

    def test_gguf_header_only(self):
        path = self.root / "fixture.gguf"
        string = lambda value: struct.pack("<Q", len(value)) + value.encode()
        header = b"GGUF" + struct.pack("<IQQ", 3, 1, 1)
        header += string("general.architecture") + struct.pack("<I", 8) + string("deepseek_v4")
        header += string("blk.0.attn_norm.weight") + struct.pack("<IQIQ", 1, 4096, 30, 0)
        path.write_bytes(header + bytes((-len(header)) % 32) + bytes(8192))
        result = gguf(path)
        self.assertEqual(result["header_bytes_read"], len(header))
        self.assertEqual(result["metadata"]["general.architecture"], "deepseek_v4")
        self.assertEqual(result["tensors"][0]["shape_order"], "ggml_ne")


if __name__ == "__main__":
    unittest.main()

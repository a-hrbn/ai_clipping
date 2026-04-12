import unittest
import sys
import types

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services.transcriber import CHUNK_SECONDS


class TranscriberChunkingTests(unittest.TestCase):
    def test_non_overlapping_chunks_use_exact_chunk_offsets(self) -> None:
        offsets = [idx * CHUNK_SECONDS for idx in range(4)]
        self.assertEqual(offsets, [0, 600, 1200, 1800])


if __name__ == "__main__":
    unittest.main()

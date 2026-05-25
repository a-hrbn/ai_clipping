import unittest
from pathlib import Path

from config import BACKEND_ROOT, PROJECT_ROOT, _prepare_local_cache_path, _resolve_local_project_path


class ConfigPathTests(unittest.TestCase):
    def test_relative_cache_paths_resolve_inside_project(self) -> None:
        resolved = _resolve_local_project_path(
            "./tmp/storage",
            field_name="storage_dir",
        )

        assert resolved is not None
        resolved_path = Path(resolved)
        self.assertEqual(resolved_path, (BACKEND_ROOT / "tmp" / "storage").resolve())
        self.assertTrue(resolved_path.is_absolute())
        self.assertEqual(resolved_path.relative_to(PROJECT_ROOT), Path("backend/tmp/storage"))

    def test_external_cache_paths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_local_project_path("/tmp/storage", field_name="storage_dir")

    def test_prepare_local_cache_path_skips_existing_unwritable_directory(self) -> None:
        root = BACKEND_ROOT / "tmp" / "test_config_unwritable"
        root.mkdir(parents=True, exist_ok=True)
        root.chmod(0o500)
        try:
            resolved = _prepare_local_cache_path(root, fallback_name="test_config_fallback")
        finally:
            root.chmod(0o700)
            root.rmdir()

        self.assertNotEqual(resolved, root.resolve())
        self.assertTrue((resolved / ".").exists())


if __name__ == "__main__":
    unittest.main()

import unittest

from services import profile_config


class ProfileConfigTests(unittest.TestCase):
    def test_jynxzi_profile_exists_and_is_balanced_for_mixed_content(self) -> None:
        profile = profile_config.get_profile("jynxzi")

        self.assertEqual(profile["chat_weight"], 1.0)
        self.assertEqual(profile["visual_weight"], 1.0)
        self.assertGreater(profile["dialogue_weight"], profile_config.get_profile("gaming")["dialogue_weight"])
        self.assertGreater(profile["quote_weight"], profile_config.get_profile("gaming")["quote_weight"])
        self.assertGreater(profile["surprise_weight"], profile_config.get_profile("general")["surprise_weight"])
        self.assertGreater(profile["hook_weight"], profile_config.get_profile("general")["hook_weight"])

    def test_unknown_override_keys_are_logged_and_ignored(self) -> None:
        with self.assertLogs(profile_config.logger.name, level="WARNING") as logs:
            weights = profile_config.get_effective_heuristic_weights("jynxzi")

        self.assertNotIn("surprise_via_semantic", weights)
        self.assertTrue(
            any("Ignoring unknown heuristic override keys" in message for message in logs.output)
        )


if __name__ == "__main__":
    unittest.main()

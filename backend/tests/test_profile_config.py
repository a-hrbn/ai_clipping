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


if __name__ == "__main__":
    unittest.main()

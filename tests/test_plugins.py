from __future__ import annotations

import unittest

import updatesupport as us


class PluginRegistryTests(unittest.TestCase):
    def tearDown(self):
        us.unregister_plugin("demo")

    def test_registers_and_looks_up_plugin_surfaces(self):
        metric = lambda column: us.row_metric(  # noqa: E731
            "demo_metric",
            lambda row: row[column],
            columns=(column,),
        )
        q_preset = lambda radius=0.5: us.q_bounded_shift(radius)  # noqa: E731
        report_profile = lambda *args, **kwargs: (args, kwargs)  # noqa: E731
        compiler = lambda data, **kwargs: (data, kwargs)  # noqa: E731
        plugin = us.UpdateSupportPlugin(
            name="demo",
            version="0.0.1",
            description="demo plugin",
            metrics={"demo_metric": metric},
            q_presets={"demo_q": q_preset},
            report_profiles={"demo_report": report_profile},
            compilers={"demo_compiler": compiler},
        )

        us.register_plugin(plugin)

        self.assertEqual(us.get_plugin("demo"), plugin)
        self.assertIn(plugin, us.list_plugins())
        self.assertIs(us.plugin_metric("demo", "demo_metric"), metric)
        self.assertIs(us.plugin_q_preset("demo", "demo_q"), q_preset)
        self.assertIs(us.plugin_report_profile("demo", "demo_report"), report_profile)
        self.assertIs(us.plugin_compiler("demo", "demo_compiler"), compiler)

    def test_missing_plugin_surface_raises_key_error(self):
        us.register_plugin(us.UpdateSupportPlugin(name="demo"))

        with self.assertRaisesRegex(KeyError, "has no metric"):
            us.plugin_metric("demo", "missing")


if __name__ == "__main__":
    unittest.main()

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

    def test_plugin_metadata_and_validation_report_are_serializable(self):
        plugin = us.UpdateSupportPlugin(
            name="demo",
            version="0.0.1",
            description="demo plugin",
            metadata=us.PluginMetadata(
                package="updatesupport-demo",
                homepage="https://example.invalid/updatesupport-demo",
                domain="demo",
                tags=["example", "sdk"],
                min_updatesupport_version="0.1.1",
            ),
        )

        report = us.validate_plugin(plugin)

        self.assertTrue(report.ok)
        self.assertEqual(report.errors, ())
        self.assertEqual(plugin.metadata.tags, ("example", "sdk"))
        self.assertEqual(plugin.as_dict()["metadata"]["package"], "updatesupport-demo")
        self.assertEqual(plugin.as_dict()["metrics"], [])
        self.assertEqual(report.as_dict()["plugin_name"], "demo")

    def test_validation_reports_invalid_plugin_surfaces(self):
        plugin = us.UpdateSupportPlugin(
            name="bad plugin",
            metrics={"not_callable": object()},
            q_presets={1: lambda: None},
            metadata=us.PluginMetadata(tags=["ok", ""]),
        )

        report = us.validate_plugin(plugin)

        self.assertFalse(report.ok)
        codes = {issue.code for issue in report.errors}
        self.assertIn("plugin-name", codes)
        self.assertIn("surface-callable", codes)
        self.assertIn("surface-key", codes)
        self.assertIn("metadata-tags", codes)
        with self.assertRaisesRegex(ValueError, "invalid updatesupport plugin"):
            us.assert_valid_plugin(plugin)

    def test_validation_reports_non_plugin_objects(self):
        report = us.validate_plugin(object())

        self.assertFalse(report.ok)
        self.assertEqual(report.errors[0].code, "plugin-type")

    def test_validation_reports_non_iterable_metadata_tags(self):
        plugin = us.UpdateSupportPlugin(
            name="demo",
            metadata=us.PluginMetadata(tags=1),
        )

        report = us.validate_plugin(plugin)

        self.assertFalse(report.ok)
        self.assertEqual(report.errors[0].code, "metadata-tags")

    def test_duplicate_plugin_registration_requires_replace(self):
        first = us.UpdateSupportPlugin(name="demo", description="first")
        second = us.UpdateSupportPlugin(name="demo", description="second")

        us.register_plugin(first)

        with self.assertRaisesRegex(ValueError, "already registered"):
            us.register_plugin(second)

        us.register_plugin(second, replace=True)
        self.assertIs(us.get_plugin("demo"), second)


if __name__ == "__main__":
    unittest.main()

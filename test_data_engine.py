from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from data_engine import (
    AlignmentEngine,
    ChangePointDetector,
    DataLoader,
    Exporter,
    IntervalAnnotator,
    SensorSchema,
)


def make_schema() -> SensorSchema:
    return SensorSchema.from_dict({
        "master_device": "camera",
        "devices": [
            {
                "name": "camera",
                "type": "image",
                "frequency_hz": 1,
                "timestamp_column": "time",
                "image_column": "filename",
                "interpolation": "nearest",
            },
            {
                "name": "imu",
                "type": "numeric",
                "frequency_hz": 2,
                "timestamp_column": "time",
                "value_columns": ["accel"],
                "interpolation": "linear",
            },
        ],
        "label_classes": [
            {"name": "平地", "color": "#00aa00"},
            {"name": "坡道", "color": "#ffaa00"},
        ],
        "boundary": {
            "enabled": True,
            "margin_seconds": 0.1,
            "ignore_label": "Ignore",
        },
        "change_point_detection": {
            "algorithm": "pelt",
            "model": "l2",
            "min_size": 5,
            "jump": 1,
        },
    })


def make_grouped_schema() -> SensorSchema:
    schema = make_schema()
    schema.label_groups = [
        {"name": "地形", "mode": "single"},
        {"name": "事件", "mode": "multi"},
    ]
    schema.label_classes = [
        {"name": "平地", "color": "#00aa00", "group": "地形"},
        {"name": "坡道", "color": "#ffaa00", "group": "地形"},
        {"name": "打滑", "color": "#cc0000", "group": "事件"},
        {"name": "遮挡", "color": "#555555", "group": "事件"},
    ]
    schema.validate()
    return schema


class SchemaAndLoadingTests(unittest.TestCase):
    def test_legacy_schema_gets_default_single_select_group(self):
        schema = make_schema()
        self.assertEqual(["类别"], schema.label_group_names())
        self.assertEqual("single", schema.label_group_mode("类别"))
        self.assertTrue(all(label["group"] == "类别" for label in schema.label_classes))

    def test_schema_rejects_empty_and_duplicate_label_names(self):
        schema = make_schema()
        schema.label_classes.append({"name": "平地", "color": "#ffffff"})
        with self.assertRaisesRegex(ValueError, "标签名称重复"):
            schema.validate()

        schema.label_classes = [{"name": "", "color": "#ffffff"}]
        with self.assertRaisesRegex(ValueError, "标签名称不能为空"):
            schema.validate()

    def test_schema_yaml_round_trip_preserves_device_type(self):
        schema = make_schema()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema.yaml"
            schema.to_yaml(str(path))
            loaded = SensorSchema.from_yaml(str(path))
        self.assertEqual("image", loaded.devices[0].device_type)
        self.assertEqual("numeric", loaded.devices[1].device_type)
        self.assertIn("type", schema.to_dict()["devices"][0])
        self.assertNotIn("device_type", schema.to_dict()["devices"][0])

    def test_schema_rejects_invalid_group_mode_and_label_reference(self):
        schema = make_grouped_schema()
        schema.label_groups[1]["mode"] = "checkbox"
        with self.assertRaisesRegex(ValueError, "模式无效"):
            schema.validate()

        schema = make_grouped_schema()
        schema.label_classes[0]["group"] = "不存在"
        with self.assertRaisesRegex(ValueError, "不存在的标签组"):
            schema.validate()

    def test_schema_exports_do_not_include_api_key(self):
        schema = make_schema()
        schema.llm_assistant = {
            "enabled": True,
            "model": "test-model",
            "base_url": "https://example.test/v1",
            "api_key": "secret-value",
        }
        self.assertNotIn("api_key", schema.to_dict()["llm_assistant"])
        self.assertEqual(
            "secret-value",
            schema.to_dict(include_secrets=True)["llm_assistant"]["api_key"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema.yaml"
            schema.to_yaml(str(path))
            self.assertNotIn("secret-value", path.read_text(encoding="utf-8"))

    def test_timestamp_parser_supports_seconds_milliseconds_and_iso(self):
        seconds = DataLoader.normalize_timestamps(pd.Series([0.0, 1.0]))
        milliseconds = DataLoader.normalize_timestamps(
            pd.Series([1_700_000_000_000, 1_700_000_001_000])
        )
        iso = DataLoader.normalize_timestamps(
            pd.Series(["2026-07-22T10:00:00Z", "2026-07-22T10:00:01Z"])
        )
        self.assertEqual(1.0, (seconds.iloc[1] - seconds.iloc[0]).total_seconds())
        self.assertEqual(
            1.0, (milliseconds.iloc[1] - milliseconds.iloc[0]).total_seconds()
        )
        self.assertEqual(1.0, (iso.iloc[1] - iso.iloc[0]).total_seconds())


class AlignmentTests(unittest.TestCase):
    def test_alignment_prefixes_master_and_interpolates_subtrack(self):
        schema = make_schema()
        data = {
            "camera": pd.DataFrame({
                "time": pd.to_datetime([0, 1, 2], unit="s"),
                "filename": ["a.jpg", "b.jpg", "c.jpg"],
            }),
            "imu": pd.DataFrame({
                "time": pd.to_datetime([0, 0.5, 1, 1.5, 2], unit="s"),
                "accel": [0.0, 5.0, 10.0, 15.0, 20.0],
            }),
        }
        aligned = AlignmentEngine(schema).align(data, tolerance_seconds=0.6)
        self.assertEqual(
            ["timestamp", "camera.filename", "imu.accel"],
            aligned.columns.tolist(),
        )
        np.testing.assert_allclose(aligned["imu.accel"], [0.0, 10.0, 20.0])

    def test_clapperboard_offsets_are_applied(self):
        schema = SensorSchema.from_dict({
            "master_device": "ref",
            "devices": [
                {
                    "name": "ref", "type": "numeric", "timestamp_column": "time",
                    "value_columns": ["impact"],
                },
                {
                    "name": "late", "type": "numeric", "timestamp_column": "time",
                    "value_columns": ["impact"],
                },
            ],
        })
        data = {
            "ref": pd.DataFrame({"time": [0.0, 1.0, 2.0], "impact": [0, 9, 0]}),
            "late": pd.DataFrame({"time": [5.0, 6.0, 7.0], "impact": [0, 20, 0]}),
        }
        offsets = AlignmentEngine(schema).clapperboard_align(
            data, "ref", "impact", apply=True
        )
        self.assertAlmostEqual(-5.0, offsets["late"])
        expected = pd.to_datetime([0.0, 1.0, 2.0], unit="s")
        pd.testing.assert_series_equal(
            data["late"]["time"], pd.Series(expected, name="time")
        )


class AnnotationAndExportTests(unittest.TestCase):
    def test_annotation_draft_round_trip_is_atomic(self):
        schema = make_grouped_schema()
        schema.boundary["enabled"] = False
        times = pd.date_range("2026-01-01", periods=5, freq="1s")
        df = pd.DataFrame({"timestamp": times})
        source = IntervalAnnotator(schema)
        source.add_interval(times[0], times[2], "平地", df, group="地形")
        source.add_interval(times[1], times[3], ["打滑", "遮挡"], df, group="事件")

        restored = IntervalAnnotator(schema)
        restored.load_list(source.to_list(), df)
        pd.testing.assert_frame_equal(
            source.apply_to_dataframe(df), restored.apply_to_dataframe(df)
        )
        before = restored.to_list()
        with self.assertRaisesRegex(ValueError, "第 1 个标注区间无效"):
            restored.load_list([{
                "start_time": times[0], "end_time": times[1],
                "group": "事件", "labels": ["不存在"],
            }], df)
        self.assertEqual(before, restored.to_list())

    def test_quality_issues_find_gaps_short_intervals_and_single_overlap(self):
        schema = make_grouped_schema()
        schema.boundary["enabled"] = False
        times = pd.date_range("2026-01-01", periods=8, freq="1s")
        df = pd.DataFrame({"timestamp": times})
        annotator = IntervalAnnotator(schema)
        annotator.add_interval(times[0], times[5], "平地", df, group="地形")
        annotator.add_interval(times[3], times[6], "坡道", df, group="地形")
        annotator.add_interval(times[2], times[3], "打滑", df, group="事件")

        issues = annotator.quality_issues(df, min_interval_frames=3)
        codes = [issue["code"] for issue in issues]
        self.assertIn("unlabeled_run", codes)
        self.assertIn("short_interval", codes)
        self.assertIn("single_group_overlap", codes)
        overlap = next(issue for issue in issues if issue["code"] == "single_group_overlap")
        self.assertEqual("地形", overlap["group"])
        self.assertEqual(["平地", "坡道"], overlap["labels"])

    def test_groups_coexist_and_multi_group_unions_overlapping_intervals(self):
        schema = make_grouped_schema()
        schema.boundary["enabled"] = False
        annotator = IntervalAnnotator(schema)
        times = pd.date_range("2026-01-01", periods=6, freq="1s")
        df = pd.DataFrame({"timestamp": times})

        annotator.add_interval(times[0], times[4], "平地", df, group="地形")
        annotator.add_interval(times[1], times[3], "打滑", df, group="事件")
        annotator.add_interval(times[2], times[4], "遮挡", df, group="事件")
        labeled = annotator.apply_to_dataframe(df)

        self.assertEqual("平地", labeled.loc[2, "label__地形"])
        self.assertEqual("平地", labeled.loc[2, "label"])
        self.assertEqual("打滑|遮挡", labeled.loc[2, "label__事件"])
        self.assertEqual("遮挡", labeled.loc[4, "label__事件"])
        self.assertEqual(
            {"打滑": 3, "遮挡": 3},
            annotator.annotation_stats(df, "事件")["label_counts"],
        )

    def test_later_single_group_interval_overwrites_older_interval(self):
        schema = make_grouped_schema()
        schema.boundary["enabled"] = False
        annotator = IntervalAnnotator(schema)
        times = pd.date_range("2026-01-01", periods=5, freq="1s")
        df = pd.DataFrame({"timestamp": times})

        annotator.add_interval(times[1], times[3], "坡道", df, group="地形")
        annotator.add_interval(times[0], times[4], "平地", df, group="地形")
        labeled = annotator.apply_to_dataframe(df)
        self.assertEqual(["平地"] * 5, labeled["label__地形"].tolist())

    def test_annotation_stats_and_next_unlabeled_interval(self):
        schema = make_schema()
        schema.boundary["enabled"] = False
        annotator = IntervalAnnotator(schema)
        times = pd.date_range("2026-01-01", periods=10, freq="1s")
        df = pd.DataFrame({"timestamp": times, "imu.accel": np.arange(10)})
        annotator.add_interval(times[0], times[2], "平地", df)
        annotator.add_interval(times[5], times[6], "坡道", df)

        stats = annotator.annotation_stats(df)
        self.assertEqual(5, stats["labeled_frames"])
        self.assertEqual(5, stats["unlabeled_frames"])
        self.assertEqual({"平地": 3, "坡道": 2}, stats["label_counts"])
        self.assertEqual((times[3], times[4]), annotator.next_unlabeled_interval(df))
        self.assertEqual(
            (times[7], times[9]),
            annotator.next_unlabeled_interval(df, after_time=times[4]),
        )

    def test_single_frame_gap_can_be_selected_and_annotated(self):
        schema = make_schema()
        schema.boundary["enabled"] = False
        annotator = IntervalAnnotator(schema)
        times = pd.date_range("2026-01-01", periods=3, freq="1s")
        df = pd.DataFrame({"timestamp": times})
        annotator.add_interval(times[0], times[1], "平地", df)

        self.assertEqual(
            (times[2], times[2]), annotator.next_unlabeled_interval(df)
        )
        interval = annotator.add_interval(times[2], times[2], "坡道", df)
        self.assertEqual(1, interval["frame_count"])
        self.assertEqual("坡道", annotator.apply_to_dataframe(df).loc[2, "label"])

    def test_change_points_map_back_across_missing_values(self):
        schema = make_schema()
        first = np.zeros(30)
        second = np.ones(30) * 10
        values = np.concatenate([first, [np.nan, np.nan], second])
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=len(values), freq="10ms"),
            "imu.accel": values,
        })
        detector = ChangePointDetector(schema)
        points = detector.detect(df, "imu.accel", pen=1)
        self.assertTrue(points)
        self.assertGreaterEqual(points[0], 30)

    def test_adjacent_intervals_get_boundary_ignore_label(self):
        schema = make_schema()
        annotator = IntervalAnnotator(schema)
        times = pd.date_range("2026-01-01", periods=21, freq="100ms")
        df = pd.DataFrame({"timestamp": times})
        annotator.add_interval(times[0], times[10], "平地", df)
        annotator.add_interval(times[10], times[-1], "坡道", df)
        labeled = annotator.apply_to_dataframe(df)
        self.assertEqual("Ignore", labeled.loc[10, "label"])
        self.assertEqual("平地", labeled.loc[2, "label"])
        self.assertEqual("坡道", labeled.loc[18, "label"])

    def test_hdf5_keeps_image_names_and_stable_label_ids(self):
        schema = make_schema()
        times = pd.date_range("2026-01-01", periods=2, freq="1s")
        df = pd.DataFrame({
            "timestamp": times,
            "camera.filename": ["a.jpg", "b.jpg"],
            "imu.accel": [1.0, 2.0],
            "label": ["平地", "坡道"],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.h5"
            Exporter.to_hdf5(df, [], schema, str(path))
            with h5py.File(path, "r") as file:
                names = [value.decode("utf-8") for value in file["aligned_data/camera.filename"][:]]
                label_ids = file["aligned_data/label_id"][:].tolist()
        self.assertEqual(["a.jpg", "b.jpg"], names)
        self.assertEqual([0, 1], label_ids)

    def test_hdf5_exports_group_ids_and_multi_hot_labels(self):
        schema = make_grouped_schema()
        schema.boundary["enabled"] = False
        annotator = IntervalAnnotator(schema)
        times = pd.date_range("2026-01-01", periods=3, freq="1s")
        source = pd.DataFrame({"timestamp": times, "imu.accel": [1.0, 2.0, 3.0]})
        annotator.add_interval(times[0], times[-1], "平地", source, group="地形")
        annotator.add_interval(times[0], times[1], ["打滑", "遮挡"], source, group="事件")
        labeled = annotator.apply_to_dataframe(source)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grouped.h5"
            Exporter.to_hdf5(labeled, annotator.to_list(), schema, str(path))
            with h5py.File(path, "r") as file:
                group_ids = file["aligned_data/label_id__地形"][:].tolist()
                multi_hot = file["aligned_data/label_multihot__事件"][:].tolist()
                mapping = json.loads(
                    file["aligned_data/label_multihot__事件"].attrs["label_mapping"]
                )
                interval_group = file["intervals/interval_1"].attrs["group"]

        self.assertEqual([0, 0, 0], group_ids)
        self.assertEqual([[1, 1], [1, 1], [0, 0]], multi_hot)
        self.assertEqual({"打滑": 0, "遮挡": 1}, mapping)
        self.assertEqual("事件", interval_group)


if __name__ == "__main__":
    unittest.main()

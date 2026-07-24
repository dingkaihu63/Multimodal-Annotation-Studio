"""Session-oriented workflow, pre-annotation, export, and balance analysis."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_engine import DataLoader, IntervalAnnotator, SensorSchema


MANIFEST_COLUMNS = [
    "session_id", "subject_id", "scene_id", "device", "data_path",
    "timestamp_file", "depth_path", "planned_label",
]


def _clean(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _resolved_path(base_dir: str | Path | None, value: Any) -> str:
    raw = _clean(value)
    if not raw:
        return ""
    path = Path(raw)
    if path.is_absolute() or base_dir is None:
        return str(path)
    return str((Path(base_dir) / path).resolve())


@dataclass
class SessionRecord:
    session_id: str
    subject_id: str = ""
    scene_id: str = ""
    planned_label: str = ""
    devices: dict[str, dict] = field(default_factory=dict)

    def metadata(self) -> dict:
        return {
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "scene_id": self.scene_id,
            "planned_label": self.planned_label,
        }


class SessionCatalog:
    """Validated collection of sessions imported from a long-form CSV manifest."""

    def __init__(self, records: list[SessionRecord] | None = None):
        self.records = records or []

    @classmethod
    def from_dataframe(
        cls, manifest: pd.DataFrame, base_dir: str | Path | None = None,
    ) -> "SessionCatalog":
        required = {"session_id", "subject_id", "scene_id", "device", "data_path"}
        missing = sorted(required - set(manifest.columns))
        if missing:
            raise ValueError(f"Session manifest 缺少字段: {missing}")
        if manifest.empty:
            raise ValueError("Session manifest 不能为空")

        records: list[SessionRecord] = []
        for session_id, rows in manifest.groupby("session_id", sort=False):
            clean_id = _clean(session_id)
            if not clean_id:
                raise ValueError("session_id 不能为空")
            subjects = {_clean(value) for value in rows["subject_id"]}
            scenes = {_clean(value) for value in rows["scene_id"]}
            if len(subjects) != 1 or len(scenes) != 1:
                raise ValueError(f"Session '{clean_id}' 的 subject_id/scene_id 不一致")
            planned_values = {
                _clean(value) for value in rows.get(
                    "planned_label", pd.Series([""] * len(rows), index=rows.index)
                ) if _clean(value)
            }
            if len(planned_values) > 1:
                raise ValueError(f"Session '{clean_id}' 的 planned_label 不一致")
            device_rows: dict[str, dict] = {}
            for _, row in rows.iterrows():
                device_name = _clean(row["device"])
                if not device_name:
                    raise ValueError(f"Session '{clean_id}' 存在空设备名称")
                if device_name in device_rows:
                    raise ValueError(
                        f"Session '{clean_id}' 的设备 '{device_name}' 重复"
                    )
                data_path = _resolved_path(base_dir, row.get("data_path"))
                if not data_path:
                    raise ValueError(
                        f"Session '{clean_id}' 的设备 '{device_name}' 缺少 data_path"
                    )
                device_rows[device_name] = {
                    "data_path": data_path,
                    "timestamp_file": _resolved_path(
                        base_dir, row.get("timestamp_file")
                    ),
                    "depth_path": _resolved_path(base_dir, row.get("depth_path")),
                }
            records.append(SessionRecord(
                session_id=clean_id,
                subject_id=next(iter(subjects)),
                scene_id=next(iter(scenes)),
                planned_label=next(iter(planned_values), ""),
                devices=device_rows,
            ))
        return cls(records)

    @classmethod
    def from_csv(
        cls, source: str | Path | Any, base_dir: str | Path | None = None,
    ) -> "SessionCatalog":
        return cls.from_dataframe(pd.read_csv(source), base_dir=base_dir)

    def session_ids(self) -> list[str]:
        return [record.session_id for record in self.records]

    def get(self, session_id: str) -> SessionRecord:
        for record in self.records:
            if record.session_id == session_id:
                return record
        raise KeyError(session_id)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for record in self.records:
            for device, paths in record.devices.items():
                rows.append({**record.metadata(), "device": device, **paths})
        return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

    @staticmethod
    def template(schema: SensorSchema) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "session_id": "S001", "subject_id": "P001",
                "scene_id": "lab", "device": device.name,
                "data_path": device.data_path,
                "timestamp_file": device.timestamp_file,
                "depth_path": device.depth_path,
                "planned_label": "平地",
            }
            for device in schema.devices
        ], columns=MANIFEST_COLUMNS)

    def schema_for(self, base_schema: SensorSchema, session_id: str) -> SensorSchema:
        record = self.get(session_id)
        session_schema = SensorSchema.from_dict(
            copy.deepcopy(base_schema.to_dict(include_secrets=True))
        )
        unknown_devices = sorted(
            set(record.devices) - {device.name for device in session_schema.devices}
        )
        if unknown_devices:
            raise ValueError(
                f"Session '{session_id}' 引用了 Schema 中不存在的设备: {unknown_devices}"
            )
        session_schema.devices = [
            device for device in session_schema.devices
            if device.name in record.devices
        ]
        if session_schema.master_device not in record.devices:
            raise ValueError(
                f"Session '{session_id}' 缺少主设备 '{session_schema.master_device}'"
            )
        for device_name, paths in record.devices.items():
            device = session_schema.get_device(device_name)
            device.data_path = paths.get("data_path", "")
            device.timestamp_file = paths.get("timestamp_file", "")
            device.depth_path = paths.get("depth_path", "")
        session_schema.llm_assistant = dict(base_schema.llm_assistant)
        session_schema.validate()
        return session_schema

    def load_session(
        self, base_schema: SensorSchema, session_id: str,
    ) -> tuple[SensorSchema, dict[str, pd.DataFrame]]:
        schema = self.schema_for(base_schema, session_id)
        data = {device.name: DataLoader.load(device) for device in schema.devices}
        empty = [name for name, frame in data.items() if frame.empty]
        if empty:
            raise ValueError(f"Session '{session_id}' 的设备数据为空: {empty}")
        return schema, data


class GaitPreAnnotator:
    """FSR-based foot-contact and coarse gait-phase pre-annotation."""

    LEFT_TOKENS = ("heel_left", "mid_left", "fore_left", "left")
    RIGHT_TOKENS = ("heel_right", "mid_right", "fore_right", "right")

    @staticmethod
    def _columns(df: pd.DataFrame, side: str) -> list[str]:
        tokens = GaitPreAnnotator.LEFT_TOKENS if side == "left" else GaitPreAnnotator.RIGHT_TOKENS
        return [
            column for column in df.select_dtypes(include=[np.number]).columns
            if any(token in str(column).lower() for token in tokens)
        ]

    @staticmethod
    def contacts(
        df: pd.DataFrame, threshold_ratio: float = 0.2,
    ) -> pd.DataFrame:
        output = pd.DataFrame(index=df.index)
        for side in ("left", "right"):
            columns = GaitPreAnnotator._columns(df, side)
            if not columns:
                output[f"foot_contact_{side}"] = np.uint8(0)
                continue
            force = df[columns].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
            low, high = float(force.quantile(0.1)), float(force.quantile(0.9))
            threshold = low + max(0.0, min(1.0, threshold_ratio)) * (high - low)
            output[f"foot_contact_{side}"] = (force > threshold).astype(np.uint8)
        return output

    @staticmethod
    def _runs(values: pd.Series, timestamps: pd.Series) -> list[tuple[Any, Any, Any]]:
        if values.empty:
            return []
        runs: list[tuple[Any, Any, Any]] = []
        start = 0
        for position in range(1, len(values)):
            if values.iloc[position] == values.iloc[start]:
                continue
            runs.append((
                timestamps.iloc[start], timestamps.iloc[position - 1], values.iloc[start]
            ))
            start = position
        runs.append((timestamps.iloc[start], timestamps.iloc[-1], values.iloc[start]))
        return runs

    @staticmethod
    def apply_planned_gait(
        df: pd.DataFrame,
        annotator: IntervalAnnotator,
        group: str = "步态阶段",
        threshold_ratio: float = 0.2,
    ) -> int:
        if group not in annotator.schema.label_group_names():
            raise ValueError(f"Schema 中不存在标签组 '{group}'")
        contacts = GaitPreAnnotator.contacts(df, threshold_ratio)
        combined = contacts[["foot_contact_left", "foot_contact_right"]].max(axis=1)
        labels = combined.map({1: "支撑期", 0: "摆动期"})
        valid = set(annotator.schema.label_names(group))
        created = 0
        for start, end, label in GaitPreAnnotator._runs(labels, df["timestamp"]):
            if label not in valid:
                continue
            annotator.add_interval(
                start, end, label, df, group=group, layer="planned",
                source="fsr_auto", confidence=0.75,
            )
            created += 1
        return created


class QualityPreAnnotator:
    """Create planned validity and missing-data quality intervals."""

    @staticmethod
    def apply_missing_data(
        df: pd.DataFrame,
        annotator: IntervalAnnotator,
        validity_group: str = "有效性",
        quality_group: str = "质量异常",
    ) -> int:
        data_columns = [
            column for column in df.columns
            if column != "timestamp" and not str(column).startswith((
                "label", "planned_label", "verified_label", "boundary__"
            ))
        ]
        missing = df[data_columns].isna().any(axis=1) if data_columns else pd.Series(False, index=df.index)
        created = 0
        if validity_group in annotator.schema.label_group_names():
            labels = missing.map({True: "无效", False: "有效"})
            valid = set(annotator.schema.label_names(validity_group))
            for start, end, label in GaitPreAnnotator._runs(labels, df["timestamp"]):
                if label in valid:
                    annotator.add_interval(
                        start, end, label, df, group=validity_group,
                        layer="planned", source="quality_auto", confidence=0.9,
                    )
                    created += 1
        if quality_group in annotator.schema.label_group_names() and missing.any():
            valid = set(annotator.schema.label_names(quality_group))
            if "丢帧" in valid:
                for start, end, is_missing in GaitPreAnnotator._runs(missing, df["timestamp"]):
                    if bool(is_missing):
                        annotator.add_interval(
                            start, end, "丢帧", df, group=quality_group,
                            layer="planned", source="quality_auto", confidence=1.0,
                        )
                        created += 1
        return created


class TrainingExporter:
    """Build the stable frame table expected by downstream training code."""

    DEFAULT_GROUPS = {
        "terrain": "地形",
        "surface": "表面",
        "action": "动作",
        "gait_phase": "步态阶段",
        "operation_event": "操作事件",
        "validity": "有效性",
        "quality": "质量异常",
    }

    @staticmethod
    def build_frame_table(
        df: pd.DataFrame,
        annotator: IntervalAnnotator,
        schema: SensorSchema,
        metadata: dict | None = None,
    ) -> pd.DataFrame:
        metadata = dict(metadata or {})
        labeled = annotator.apply_to_dataframe(df).reset_index(drop=True)
        groups = dict(TrainingExporter.DEFAULT_GROUPS)
        groups.update(schema.training_export.get("group_mapping", {}))
        output = pd.DataFrame({
            "session_id": metadata.get("session_id", annotator.session_id),
            "subject_id": metadata.get("subject_id", ""),
            "scene_id": metadata.get("scene_id", ""),
            "frame_index": np.arange(len(labeled), dtype=np.int64),
            "timestamp": labeled["timestamp"].astype(str),
        })

        image_device = next(
            (device for device in schema.devices if device.device_type in {"image", "mixed"}),
            None,
        )
        output["rgb_path"] = ""
        output["depth_path"] = ""
        if image_device is not None:
            image_column = f"{image_device.name}.{image_device.image_column}"
            if image_column not in labeled.columns:
                image_column = image_device.image_column
            if image_column in labeled.columns:
                filenames = labeled[image_column].fillna("").astype(str)
                output["rgb_path"] = filenames.map(
                    lambda value: os.path.join(image_device.data_path, value) if value else ""
                )
                output["depth_path"] = filenames.map(
                    lambda value: os.path.join(
                        image_device.depth_path, f"{Path(value).stem}.png"
                    ) if value and image_device.depth_path else ""
                )

        for export_name, group_name in groups.items():
            for layer in ("planned", "verified"):
                source = schema.label_column(group_name, layer)
                output[f"{export_name}_{layer}"] = (
                    labeled[source] if source in labeled.columns else ""
                )
            effective_source = schema.label_column(group_name)
            output[export_name] = (
                labeled[effective_source] if effective_source in labeled.columns else ""
            )

        contacts = GaitPreAnnotator.contacts(labeled)
        output["foot_contact_left"] = contacts["foot_contact_left"].to_numpy()
        output["foot_contact_right"] = contacts["foot_contact_right"].to_numpy()
        output["terrain_boundary"] = labeled.get(
            f"boundary__{groups['terrain']}", pd.Series(0, index=labeled.index)
        ).to_numpy()
        output["surface_boundary"] = labeled.get(
            f"boundary__{groups['surface']}", pd.Series(0, index=labeled.index)
        ).to_numpy()
        for event_name, event_label in {
            "action_start": "动作起点", "action_confirm": "确认",
            "action_stop": "停止", "action_cancel": "取消",
        }.items():
            values = output["operation_event"].fillna("").astype(str)
            output[event_name] = values.str.split("|").map(
                lambda labels: np.uint8(event_label in labels)
            )
        output["annotation_version"] = annotator.version
        output["annotator"] = annotator.annotator

        generated = set(output.columns) | {"label", "planned_label", "verified_label"}
        sensor_columns = [
            column for column in labeled.columns
            if column not in generated
            and not str(column).startswith((
                "label__", "planned_label__", "verified_label__", "boundary__"
            ))
            and column != "timestamp"
        ]
        for column in sensor_columns:
            output[f"sensor__{column}"] = labeled[column].to_numpy()
        return output

    @staticmethod
    def balance_report(
        table: pd.DataFrame,
        class_column: str = "terrain",
    ) -> pd.DataFrame:
        dimensions = [class_column, "subject_id", "scene_id"]
        rows: list[dict] = []
        for dimension in dimensions:
            if dimension not in table.columns:
                continue
            counts = table[dimension].fillna("").astype(str)
            counts = counts[counts.str.strip().ne("")].value_counts()
            if counts.empty:
                continue
            maximum = int(counts.max())
            for value, count in counts.items():
                ratio = float(count / maximum) if maximum else 0.0
                rows.append({
                    "dimension": dimension,
                    "value": value,
                    "frames": int(count),
                    "ratio_to_max": ratio,
                    "status": "imbalanced" if ratio < 0.5 else "ok",
                })
        return pd.DataFrame(rows)

    @staticmethod
    def catalog_balance(catalog: SessionCatalog) -> pd.DataFrame:
        session_rows = pd.DataFrame([record.metadata() for record in catalog.records])
        if session_rows.empty:
            return pd.DataFrame()
        return TrainingExporter.balance_report(
            session_rows.rename(columns={"planned_label": "terrain"}), "terrain"
        ).rename(columns={"frames": "sessions"})

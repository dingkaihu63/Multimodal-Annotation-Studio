"""
data_engine.py
============================
动态配置与数据对齐引擎 (Step 1)

提供以下核心能力:
1. 从 YAML 加载或通过 Dict 构建动态传感器 Schema
2. 使用 pd.merge_asof 实现动态多设备高低频时间戳对齐 (Module B)
3. 基于 ruptures 的变点检测与区间自动分割 (Module D)
4. 集成 openai API 辅助分析函数 (Module D)
5. 区间标注管理 (Module C)
6. 统一导出为 HDF5 / CSV (Module E)

设计原则:
- 不硬编码任何传感器名字, 所有设备/列名均来自 Schema
- 支持未来动态扩展新的传感器设备
- 面向 Streamlit 前端, 提供可序列化的返回结构
"""

from __future__ import annotations

import os
import io
import json
import time
import base64
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd
import yaml


# =============================================================================
# 1. 数据类: 传感器配置 & Schema
# =============================================================================

@dataclass
class SensorConfig:
    """单个传感设备的动态配置。"""
    name: str
    device_type: str = "numeric"          # numeric | image | mixed
    frequency_hz: float = 30.0
    data_path: str = ""
    depth_path: str = ""                  # 仅 image 类型使用
    timestamp_file: str = ""              # image 类型的时间戳文件
    timestamp_column: str = "timestamp"
    image_column: str = "filename"        # image 类型使用的列名
    value_columns: list[str] = field(default_factory=list)
    interpolation: str = "linear"        # linear | nearest | forward_fill

    @classmethod
    def from_dict(cls, d: dict) -> "SensorConfig":
        return cls(
            name=d.get("name", "unknown"),
            device_type=d.get("type", "numeric"),
            frequency_hz=float(d.get("frequency_hz", 30.0)),
            data_path=d.get("data_path", ""),
            depth_path=d.get("depth_path", ""),
            timestamp_file=d.get("timestamp_file", ""),
            timestamp_column=d.get("timestamp_column", "timestamp"),
            image_column=d.get("image_column", "filename"),
            value_columns=list(d.get("value_columns", [])),
            interpolation=d.get("interpolation", "linear"),
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        # YAML 使用公开字段名 ``type``，内部用 device_type 避免覆盖内置名。
        data["type"] = data.pop("device_type")
        return data


@dataclass
class SensorSchema:
    """完整的传感器 Schema, 包含设备列表、标签、边界、检测、LLM 设置。"""
    master_device: str = ""
    devices: list[SensorConfig] = field(default_factory=list)
    label_groups: list[dict] = field(default_factory=list)
    label_classes: list[dict] = field(default_factory=list)
    boundary: dict = field(default_factory=dict)
    change_point_detection: dict = field(default_factory=dict)
    llm_assistant: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "SensorSchema":
        label_classes = [dict(item) for item in d.get("label_classes", [])]
        label_groups = [dict(item) for item in d.get("label_groups", [])]
        if not label_groups:
            label_groups = [{"name": "类别", "mode": "single"}]
        default_group = str(label_groups[0].get("name", "类别"))
        for item in label_classes:
            item.setdefault("group", default_group)
        return cls(
            master_device=d.get("master_device", ""),
            devices=[SensorConfig.from_dict(x) for x in d.get("devices", [])],
            label_groups=label_groups,
            label_classes=label_classes,
            boundary=d.get("boundary", {}),
            change_point_detection=d.get("change_point_detection", {}),
            llm_assistant=d.get("llm_assistant", {}),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "SensorSchema":
        """从 YAML 文件加载 Schema。"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        schema = cls.from_dict(data or {})
        schema.validate()
        return schema

    def to_dict(self, include_secrets: bool = False) -> dict:
        llm_config = dict(self.llm_assistant)
        if not include_secrets:
            llm_config.pop("api_key", None)
        return {
            "master_device": self.master_device,
            "devices": [d.to_dict() for d in self.devices],
            "label_groups": self.label_groups,
            "label_classes": self.label_classes,
            "boundary": self.boundary,
            "change_point_detection": self.change_point_detection,
            "llm_assistant": llm_config,
        }

    def to_yaml(self, path: str) -> None:
        """保存当前 Schema 到 YAML 文件。"""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)

    def validate(self) -> None:
        """校验会导致加载或对齐歧义的 Schema 错误。"""
        names = [device.name.strip() for device in self.devices]
        if any(not name for name in names):
            raise ValueError("设备名称不能为空")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"设备名称重复: {duplicates}")
        if self.devices and self.master_device not in names:
            raise ValueError(f"主设备 '{self.master_device}' 不在设备列表中")
        group_names = [str(item.get("name", "")).strip() for item in self.label_groups]
        if any(not name for name in group_names):
            raise ValueError("标签组名称不能为空")
        duplicate_groups = sorted({
            name for name in group_names if group_names.count(name) > 1
        })
        if duplicate_groups:
            raise ValueError(f"标签组名称重复: {duplicate_groups}")
        for group in self.label_groups:
            if group.get("mode", "single") not in {"single", "multi"}:
                raise ValueError(f"标签组 '{group.get('name', '')}' 的模式无效")

        label_names = [str(item.get("name", "")).strip() for item in self.label_classes]
        if any(not name for name in label_names):
            raise ValueError("标签名称不能为空")
        duplicate_labels = sorted({
            name for name in label_names if label_names.count(name) > 1
        })
        if duplicate_labels:
            raise ValueError(f"标签名称重复: {duplicate_labels}")
        unknown_groups = sorted({
            str(item.get("group", "")) for item in self.label_classes
            if str(item.get("group", "")) not in group_names
        })
        if unknown_groups:
            raise ValueError(f"标签引用了不存在的标签组: {unknown_groups}")
        for device in self.devices:
            if device.frequency_hz <= 0:
                raise ValueError(f"设备 '{device.name}' 的采样频率必须大于 0")
            if device.interpolation not in {"linear", "nearest", "forward_fill"}:
                raise ValueError(
                    f"设备 '{device.name}' 的插值策略无效: {device.interpolation}"
                )

    def get_device(self, name: str) -> SensorConfig | None:
        for d in self.devices:
            if d.name == name:
                return d
        return None

    @property
    def master(self) -> SensorConfig | None:
        return self.get_device(self.master_device)

    def label_group_names(self) -> list[str]:
        return [str(group["name"]) for group in self.label_groups]

    def default_label_group(self) -> str:
        return self.label_group_names()[0] if self.label_groups else "类别"

    def label_group_mode(self, group_name: str) -> str:
        for group in self.label_groups:
            if group.get("name") == group_name:
                return str(group.get("mode", "single"))
        return "single"

    def label_names(self, group: str | None = None) -> list[str]:
        return [
            str(item["name"]) for item in self.label_classes
            if group is None or item.get("group", self.default_label_group()) == group
        ]

    def label_group_for(self, label_name: str) -> str | None:
        for item in self.label_classes:
            if item.get("name") == label_name:
                return str(item.get("group", self.default_label_group()))
        return None

    @staticmethod
    def label_column(group_name: str) -> str:
        return f"label__{group_name}"

    def label_color(self, name: str) -> str:
        for c in self.label_classes:
            if c["name"] == name:
                return c.get("color", "#95A5A6")
        return "#95A5A6"


# =============================================================================
# 2. 数据加载器
# =============================================================================

class DataLoader:
    """根据 SensorConfig 加载单个设备数据为 DataFrame。"""

    @staticmethod
    def normalize_timestamps(values: pd.Series) -> pd.Series:
        """将 Unix 秒/毫秒/微秒/纳秒或 ISO 字符串统一为无时区时间。"""
        if pd.api.types.is_datetime64_any_dtype(values):
            parsed = pd.to_datetime(values, errors="coerce", utc=True)
            return parsed.dt.tz_convert(None)

        numeric = pd.to_numeric(values, errors="coerce")
        non_null = values.notna()
        if non_null.any() and numeric[non_null].notna().all():
            magnitude = float(numeric[non_null].abs().median())
            if magnitude >= 1e17:
                unit = "ns"
            elif magnitude >= 1e14:
                unit = "us"
            elif magnitude >= 1e11:
                unit = "ms"
            else:
                unit = "s"
            parsed = pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
        else:
            parsed = pd.to_datetime(values, errors="coerce", utc=True)
        return parsed.dt.tz_convert(None)

    @staticmethod
    def load_numeric(cfg: SensorConfig) -> pd.DataFrame:
        """加载 CSV 数值传感器数据。"""
        if not cfg.data_path or not os.path.exists(cfg.data_path):
            return pd.DataFrame()
        df = pd.read_csv(cfg.data_path)
        # 确保时间戳列存在并转换为 datetime
        if cfg.timestamp_column not in df.columns:
            raise ValueError(
                f"设备 '{cfg.name}' 的时间戳列 '{cfg.timestamp_column}' 不存在, "
                f"可用列: {list(df.columns)}"
            )
        df[cfg.timestamp_column] = DataLoader.normalize_timestamps(
            df[cfg.timestamp_column]
        )
        df = df.dropna(subset=[cfg.timestamp_column]).sort_values(cfg.timestamp_column)
        return df.reset_index(drop=True)

    @staticmethod
    def load_image(cfg: SensorConfig) -> pd.DataFrame:
        """加载图像设备的时间戳表 (CSV: timestamp, filename)。"""
        if cfg.timestamp_file and os.path.exists(cfg.timestamp_file):
            df = pd.read_csv(cfg.timestamp_file)
            if cfg.timestamp_column not in df.columns:
                raise ValueError(
                    f"设备 '{cfg.name}' 的时间戳列 '{cfg.timestamp_column}' 不存在"
                )
            df[cfg.timestamp_column] = DataLoader.normalize_timestamps(
                df[cfg.timestamp_column]
            )
            df = df.dropna(subset=[cfg.timestamp_column]).sort_values(cfg.timestamp_column)
            return df.reset_index(drop=True)
        # 回退: 扫描图像目录, 用文件名排序作为伪时间戳
        if cfg.data_path and os.path.isdir(cfg.data_path):
            files = sorted(
                f for f in os.listdir(cfg.data_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            if not files:
                return pd.DataFrame()
            # 等间隔生成伪时间戳
            n = len(files)
            ts = np.arange(n, dtype=float) / cfg.frequency_hz
            return pd.DataFrame({
                cfg.timestamp_column: pd.to_datetime(ts, unit="s"),
                cfg.image_column: files,
            })
        return pd.DataFrame()

    @staticmethod
    def load(cfg: SensorConfig) -> pd.DataFrame:
        """根据设备类型分发加载。"""
        if cfg.device_type == "image":
            return DataLoader.load_image(cfg)
        if cfg.device_type == "mixed":
            if cfg.data_path and os.path.isfile(cfg.data_path):
                return DataLoader.load_numeric(cfg)
            return DataLoader.load_image(cfg)
        return DataLoader.load_numeric(cfg)


# =============================================================================
# 3. 主轨网格对齐引擎 (Module B)
# =============================================================================

class AlignmentEngine:
    """
    基于 pd.merge_asof 的多设备高低频时间戳对齐引擎。

    核心算法:
    1. 以主设备时间戳 T_master 为主帧
    2. 对每个辅设备, 使用 merge_asof 将其数值映射到最近的主帧时间
    3. 根据插值策略 (nearest / linear / forward_fill) 进行插值填充
    """

    def __init__(self, schema: SensorSchema):
        schema.validate()
        self.schema = schema

    def align(
        self,
        device_data: dict[str, pd.DataFrame],
        tolerance_seconds: float = 0.1,
    ) -> pd.DataFrame:
        """
        将所有设备数据对齐到主设备时间戳。

        Args:
            device_data: {设备名: DataFrame} 字典
            tolerance_seconds: merge_asof 的容差 (秒), 超出则置 NaN

        Returns:
            对齐后的主 DataFrame, 索引为主设备时间戳
        """
        master_name = self.schema.master_device
        if master_name not in device_data:
            raise ValueError(
                f"主设备 '{master_name}' 未在已加载数据中找到, "
                f"可用: {list(device_data.keys())}"
            )

        if tolerance_seconds <= 0:
            raise ValueError("对齐容差必须大于 0 秒")
        master_df = device_data[master_name].copy()
        if master_df.empty:
            raise ValueError(f"主设备 '{master_name}' 数据为空")
        master_cfg = self.schema.master
        ts_col = master_cfg.timestamp_column if master_cfg else "timestamp"
        if ts_col not in master_df.columns:
            raise ValueError(f"主设备 '{master_name}' 缺少时间戳列 '{ts_col}'")

        # 重命名主设备时间戳列为统一时间列
        master_df = master_df.rename(columns={ts_col: "_time"})
        master_df["_time"] = DataLoader.normalize_timestamps(master_df["_time"])
        master_df = (
            master_df.dropna(subset=["_time"])
            .sort_values("_time")
            .drop_duplicates("_time", keep="last")
            .reset_index(drop=True)
        )

        # 所有数据列统一加设备前缀，避免主/辅设备同名列冲突。
        master_df = master_df.rename(columns={
            column: f"{master_name}.{column}"
            for column in master_df.columns if column != "_time"
        })

        aligned = master_df.copy()

        # 对每个辅设备做 merge_asof
        for cfg in self.schema.devices:
            if cfg.name == master_name:
                continue
            if cfg.name not in device_data or device_data[cfg.name].empty:
                continue

            sub = device_data[cfg.name].copy()
            sub_ts = cfg.timestamp_column
            if sub_ts not in sub.columns:
                continue
            sub = sub.rename(columns={sub_ts: "_time"})
            sub["_time"] = DataLoader.normalize_timestamps(sub["_time"])
            sub = (
                sub.dropna(subset=["_time"])
                .sort_values("_time")
                .drop_duplicates("_time", keep="last")
                .reset_index(drop=True)
            )

            # 选取需要的列 (数值列或图像列), 重命名加设备前缀避免冲突
            keep_cols = ["_time"]
            if cfg.device_type in ("image", "mixed"):
                keep_cols.append(cfg.image_column)
            if cfg.device_type in ("numeric", "mixed"):
                keep_cols.extend(cfg.value_columns)

            available = [c for c in keep_cols if c in sub.columns]
            sub = sub[available]

            rename_map = {
                c: f"{cfg.name}.{c}" for c in available if c != "_time"
            }
            sub = sub.rename(columns=rename_map)

            tol = pd.Timedelta(seconds=tolerance_seconds)
            if cfg.interpolation == "linear":
                sub = self._apply_interpolation(
                    sub, cfg, master_df["_time"], tolerance_seconds
                )
                aligned = aligned.merge(sub, on="_time", how="left")
            else:
                direction = "backward" if cfg.interpolation == "forward_fill" else "nearest"
                aligned = pd.merge_asof(
                    aligned.sort_values("_time"),
                    sub.sort_values("_time"),
                    on="_time",
                    direction=direction,
                    tolerance=tol,
                )
            aligned = aligned.reset_index(drop=True)

        # 重命名时间列为 timestamp 并设为索引
        aligned = aligned.rename(columns={"_time": "timestamp"})
        aligned["timestamp"] = pd.to_datetime(aligned["timestamp"])
        return aligned

    def _apply_interpolation(
        self,
        sub: pd.DataFrame,
        cfg: SensorConfig,
        master_time: pd.Series,
        tolerance_seconds: float,
    ) -> pd.DataFrame:
        """根据插值策略在主帧时间网格上重采样辅设备数据。"""
        numeric_cols = [
            c for c in sub.columns
            if c != "_time" and pd.api.types.is_numeric_dtype(sub[c])
        ]
        if not numeric_cols:
            return sub

        if sub.empty:
            return pd.DataFrame({"_time": master_time})

        # 设为索引后用 reindex + interpolate 映射到主帧时间。
        sub_idx = sub.set_index("_time")
        master_index = pd.DatetimeIndex(master_time.drop_duplicates())
        merged_idx = sub_idx.index.union(master_index).sort_values()
        sub_re = sub_idx.reindex(merged_idx)
        sub_re[numeric_cols] = sub_re[numeric_cols].interpolate(
            method="time", limit_area="inside"
        )

        # 容差必须对线性插值同样生效，避免跨越长时间数据缺口造值。
        source_times = pd.DataFrame({"_source_time": sub_idx.index})
        target_times = pd.DataFrame({"_time": master_index})
        nearest = pd.merge_asof(
            target_times,
            source_times,
            left_on="_time",
            right_on="_source_time",
            direction="nearest",
        )
        distances = (nearest["_time"] - nearest["_source_time"]).abs()
        outside = distances > pd.Timedelta(seconds=tolerance_seconds)

        # 只保留主帧时间点；非数值元数据在线性轨中仍按最近邻匹配。
        sub_re = sub_re.reindex(master_index)
        sub_re.loc[outside.to_numpy(), numeric_cols] = np.nan
        non_numeric = [c for c in sub_re.columns if c not in numeric_cols]
        if non_numeric:
            meta = pd.merge_asof(
                target_times,
                sub.reset_index(drop=True)[["_time", *non_numeric]],
                on="_time",
                direction="nearest",
                tolerance=pd.Timedelta(seconds=tolerance_seconds),
            ).set_index("_time")
            sub_re[non_numeric] = meta[non_numeric]
        sub_re = sub_re.reset_index().rename(columns={"index": "_time"})
        return sub_re

    def clapperboard_align(
        self,
        device_data: dict[str, pd.DataFrame],
        impact_device: str,
        impact_column: str,
        threshold: float | None = None,
        apply: bool = False,
    ) -> dict[str, float]:
        """
        打板对齐: 检测物理冲击峰值, 校正各设备系统时间零点漂移。

        Args:
            device_data: 已加载的设备数据
            impact_device: 用于检测冲击的设备名 (如 IMU/FSR)
            impact_column: 冲击信号列名 (如 accel_z)
            threshold: 峰值阈值, None 则自动取 95 分位

        Returns:
            {设备名: 时间偏移量(秒)} 用于校正各设备时间零点
        """
        if impact_device not in device_data:
            return {}
        df = device_data[impact_device]
        if impact_column not in df.columns:
            return {}

        cfg = self.schema.get_device(impact_device)
        if cfg is None or cfg.timestamp_column not in df.columns:
            return {}
        signal = pd.to_numeric(df[impact_column], errors="coerce").to_numpy()
        finite = np.isfinite(signal)
        if finite.sum() < 3:
            return {}
        centered = np.abs(signal - np.nanmedian(signal))
        if threshold is None:
            threshold = float(np.nanpercentile(centered, 95))

        peak_idx = int(np.nanargmax(centered))
        if centered[peak_idx] < threshold:
            return {}
        reference_time = DataLoader.normalize_timestamps(
            pd.Series([df.iloc[peak_idx][cfg.timestamp_column]])
        ).iloc[0]

        offsets: dict[str, float] = {}
        for name, ddf in device_data.items():
            if ddf.empty:
                continue
            device_cfg = self.schema.get_device(name)
            ts_col = device_cfg.timestamp_column if device_cfg else ddf.columns[0]
            if ts_col not in ddf.columns:
                continue
            if name == impact_device:
                offsets[name] = 0.0
                continue
            candidates = [] if device_cfg is None else [
                column for column in device_cfg.value_columns if column in ddf.columns
            ]
            if impact_column in ddf.columns:
                candidates.insert(0, impact_column)
            best: tuple[float, int] | None = None
            for column in dict.fromkeys(candidates):
                values = pd.to_numeric(ddf[column], errors="coerce").to_numpy()
                if np.isfinite(values).sum() < 3:
                    continue
                deviations = np.abs(values - np.nanmedian(values))
                mad = np.nanmedian(deviations) or np.nanstd(values) or 1.0
                idx = int(np.nanargmax(deviations))
                score = float(deviations[idx] / mad)
                if best is None or score > best[0]:
                    best = (score, idx)
            if best is None:
                continue
            device_time = DataLoader.normalize_timestamps(
                pd.Series([ddf.iloc[best[1]][ts_col]])
            ).iloc[0]
            if pd.isna(device_time) or pd.isna(reference_time):
                continue
            offsets[name] = float((reference_time - device_time).total_seconds())

        if apply:
            self.apply_time_offsets(device_data, offsets)
        return offsets

    def apply_time_offsets(
        self,
        device_data: dict[str, pd.DataFrame],
        offsets: dict[str, float],
    ) -> None:
        """原地应用打板时间偏移；offset 为需要加到设备时间戳上的秒数。"""
        for name, offset in offsets.items():
            cfg = self.schema.get_device(name)
            if cfg is None or name not in device_data:
                continue
            df = device_data[name].copy()
            if cfg.timestamp_column not in df.columns:
                continue
            timestamps = DataLoader.normalize_timestamps(df[cfg.timestamp_column])
            df[cfg.timestamp_column] = timestamps + pd.to_timedelta(offset, unit="s")
            device_data[name] = df


# =============================================================================
# 4. 变点检测 (Module D)
# =============================================================================

class ChangePointDetector:
    """基于 ruptures 的变点检测与区间自动分割。"""

    def __init__(self, schema: SensorSchema):
        self.schema = schema
        cfg = schema.change_point_detection
        self.algorithm = cfg.get("algorithm", "pelt")
        self.model = cfg.get("model", "rbf")
        self.min_size = int(cfg.get("min_size", 30))
        self.jump = int(cfg.get("jump", 5))

    def detect(
        self,
        df: pd.DataFrame,
        column: str,
        n_bkps: int | None = None,
        pen: float | None = None,
    ) -> list[int]:
        """
        对指定列做变点检测。

        Args:
            df: 对齐后的 DataFrame
            column: 检测的数值列
            n_bkps: 指定变点数 (binseg/window/bottomup 使用)
            pen: 惩罚项 (pelt 使用)

        Returns:
            变点索引列表 (DataFrame 行索引)
        """
        try:
            import ruptures as rpt
        except ImportError as e:
            raise ImportError(
                "变点检测需要 ruptures 库, 请运行: pip install ruptures"
            ) from e

        if column not in df.columns:
            raise ValueError(f"列 '{column}' 不存在于 DataFrame 中")

        series = pd.to_numeric(df[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        valid_positions = np.flatnonzero(series.notna().to_numpy())
        signal = series.iloc[valid_positions].to_numpy(dtype=float)
        if len(signal) < self.min_size * 2:
            return []

        # 构造算法对象
        if self.algorithm == "pelt":
            algo = rpt.Pelt(
                model=self.model, min_size=self.min_size, jump=self.jump
            ).fit(signal)
            bkps = algo.predict(pen=pen if pen is not None else 10)
        elif self.algorithm == "binseg":
            algo = rpt.Binseg(
                model=self.model, min_size=self.min_size, jump=self.jump
            ).fit(signal)
            bkps = algo.predict(n_bkps=n_bkps or 5)
        elif self.algorithm == "window":
            algo = rpt.Window(
                model=self.model, width=self.min_size, jump=self.jump
            ).fit(signal)
            bkps = algo.predict(n_bkps=n_bkps or 5)
        elif self.algorithm == "bottomup":
            algo = rpt.BottomUp(
                model=self.model, min_size=self.min_size, jump=self.jump
            ).fit(signal)
            bkps = algo.predict(n_bkps=n_bkps or 5)
        else:
            raise ValueError(f"未知变点算法: {self.algorithm}")

        # ruptures 返回的最后一个元素是 len(signal), 去掉
        # ruptures 的索引基于去除 NaN 后的数组，映射回原 DataFrame 行号。
        return [int(valid_positions[b]) for b in bkps if b < len(signal)]

    def segments_to_intervals(
        self, df: pd.DataFrame, bkps: list[int]
    ) -> list[dict]:
        """把变点索引转换为时间区间 [t_start, t_end]。"""
        if df.empty:
            return []
        boundaries = sorted({0, *[b for b in bkps if 0 < b < len(df)], len(df)})
        intervals = []
        ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1] - 1
            if start_idx >= len(df) or end_idx < start_idx:
                continue
            intervals.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": df.iloc[start_idx][ts_col],
                "end_time": df.iloc[end_idx][ts_col],
                "label": None,
            })
        return intervals


# =============================================================================
# 5. 区间标注管理 (Module C)
# =============================================================================

class IntervalAnnotator:
    """
    区间/片段标注管理器。

    功能:
    - 给选定时间区间赋予 Terrain_Class 标签
    - 自动填充该区间内所有帧
    - 支持边界过渡区 (±0.5s) 标记为 Ignore
    """

    def __init__(self, schema: SensorSchema):
        self.schema = schema
        self.intervals: list[dict] = []
        bnd = schema.boundary
        self.boundary_enabled = bnd.get("enabled", True)
        self.boundary_margin = float(bnd.get("margin_seconds", 0.5))
        self.ignore_label = bnd.get("ignore_label", "Ignore")

    def add_interval(
        self,
        start_time, end_time, label: str | list[str],
        df: pd.DataFrame | None = None,
        group: str | None = None,
    ) -> dict:
        """添加一个标注区间。"""
        start = pd.Timestamp(start_time)
        end = pd.Timestamp(end_time)
        if pd.isna(start) or pd.isna(end):
            raise ValueError("区间起止时间不能为空")
        if end < start:
            start, end = end, start
        labels = [label] if isinstance(label, str) else list(label)
        labels = list(dict.fromkeys(str(item).strip() for item in labels if str(item).strip()))
        if not labels:
            raise ValueError("至少需要选择一个标签")
        resolved_group = group or self.schema.label_group_for(labels[0])
        resolved_group = resolved_group or self.schema.default_label_group()
        if resolved_group not in self.schema.label_group_names():
            raise ValueError(f"未知标签组 '{resolved_group}'")
        valid_labels = self.schema.label_names(resolved_group)
        unknown_labels = [item for item in labels if item not in valid_labels]
        if unknown_labels:
            raise ValueError(
                f"标签组 '{resolved_group}' 中不存在标签: {unknown_labels}"
            )
        mode = self.schema.label_group_mode(resolved_group)
        if mode == "single" and len(labels) != 1:
            raise ValueError(f"单选标签组 '{resolved_group}' 只能选择一个标签")
        interval = {
            "start_time": start,
            "end_time": end,
            "group": resolved_group,
            "labels": labels,
            "label": " + ".join(labels),
            "color": self.schema.label_color(labels[0]),
            # Keep creation order separate from chronological display order so a
            # later single-select annotation reliably overwrites an older one.
            "_order": max(
                (int(item.get("_order", index)) for index, item in enumerate(self.intervals)),
                default=-1,
            ) + 1,
        }
        if df is not None and "timestamp" in df.columns:
            mask = (df["timestamp"] >= interval["start_time"]) & (
                df["timestamp"] <= interval["end_time"]
            )
            interval["frame_count"] = int(mask.sum())
        self.intervals.append(interval)
        self.intervals.sort(key=lambda item: item["start_time"])
        return interval

    def remove_interval(self, idx: int) -> None:
        if 0 <= idx < len(self.intervals):
            self.intervals.pop(idx)

    def clear(self) -> None:
        self.intervals.clear()

    def load_list(
        self,
        intervals: list[dict],
        df: pd.DataFrame | None = None,
        replace: bool = True,
    ) -> int:
        """Validate and restore serialized intervals atomically."""
        if not isinstance(intervals, list):
            raise ValueError("标注草稿中的 intervals 必须是列表")

        candidate = IntervalAnnotator(self.schema)
        source_items = ([] if replace else self.to_list()) + intervals
        for index, item in enumerate(source_items):
            if not isinstance(item, dict):
                raise ValueError(f"第 {index + 1} 个标注区间格式无效")
            labels = item.get("labels") or item.get("label")
            if not labels:
                raise ValueError(f"第 {index + 1} 个标注区间缺少标签")
            try:
                candidate.add_interval(
                    item.get("start_time"),
                    item.get("end_time"),
                    labels,
                    df,
                    group=item.get("group"),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"第 {index + 1} 个标注区间无效: {exc}") from exc

        self.intervals = candidate.intervals
        return len(intervals)

    def _interval_group(self, interval: dict) -> str:
        labels = self._interval_labels(interval)
        return str(
            interval.get("group")
            or (self.schema.label_group_for(labels[0]) if labels else None)
            or self.schema.default_label_group()
        )

    @staticmethod
    def _interval_labels(interval: dict) -> list[str]:
        if interval.get("labels"):
            return [str(item) for item in interval["labels"]]
        label = str(interval.get("label", "")).strip()
        return [label] if label else []

    def annotation_stats(self, df: pd.DataFrame, group: str | None = None) -> dict:
        """Return frame-level annotation coverage and per-label counts."""
        total_frames = len(df)
        if total_frames == 0:
            return {
                "total_frames": 0,
                "labeled_frames": 0,
                "ignore_frames": 0,
                "unlabeled_frames": 0,
                "assigned_ratio": 0.0,
                "label_counts": {},
            }

        resolved_group = group or self.schema.default_label_group()
        labeled_df = self.apply_to_dataframe(df)
        group_column = self.schema.label_column(resolved_group)
        labels = labeled_df.get(
            group_column, pd.Series("", index=labeled_df.index, dtype="object")
        ).fillna("").astype(str).str.strip()
        unlabeled_mask = labels.eq("")
        ignore_mask = labels.eq(str(self.ignore_label))
        labeled_mask = ~(unlabeled_mask | ignore_mask)
        label_counts: dict[str, int] = {}
        for value in labels[labeled_mask]:
            for item in str(value).split("|"):
                if item:
                    label_counts[item] = label_counts.get(item, 0) + 1
        labeled_frames = int(labeled_mask.sum())
        ignore_frames = int(ignore_mask.sum())
        unlabeled_frames = int(unlabeled_mask.sum())
        return {
            "total_frames": total_frames,
            "labeled_frames": labeled_frames,
            "ignore_frames": ignore_frames,
            "unlabeled_frames": unlabeled_frames,
            "assigned_ratio": (labeled_frames + ignore_frames) / total_frames,
            "label_counts": label_counts,
        }

    def next_unlabeled_interval(
        self, df: pd.DataFrame, after_time=None, group: str | None = None,
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Find the next contiguous unlabeled run, wrapping at the dataset end."""
        if df.empty or "timestamp" not in df.columns:
            return None

        labeled_df = self.apply_to_dataframe(df).reset_index(drop=True)
        timestamps = pd.to_datetime(labeled_df["timestamp"], errors="coerce")
        resolved_group = group or self.schema.default_label_group()
        labels = labeled_df[self.schema.label_column(resolved_group)].fillna("").astype(str).str.strip()
        unlabeled = labels.eq("") & timestamps.notna()
        positions = np.flatnonzero(unlabeled.to_numpy())
        if not len(positions):
            return None

        chosen = int(positions[0])
        if after_time is not None:
            after = pd.Timestamp(after_time)
            later_mask = (timestamps.iloc[positions] > after).to_numpy()
            later = positions[later_mask]
            if len(later):
                chosen = int(later[0])

        start_idx = chosen
        end_idx = chosen
        while start_idx > 0 and bool(unlabeled.iloc[start_idx - 1]):
            start_idx -= 1
        while end_idx + 1 < len(unlabeled) and bool(unlabeled.iloc[end_idx + 1]):
            end_idx += 1

        start_time = pd.Timestamp(timestamps.iloc[start_idx])
        end_time = pd.Timestamp(timestamps.iloc[end_idx])
        return start_time, end_time

    def quality_issues(
        self, df: pd.DataFrame, min_interval_frames: int = 3,
    ) -> list[dict]:
        """Find frame-level gaps and interval patterns that deserve review."""
        if df.empty or "timestamp" not in df.columns:
            return []
        min_frames = max(1, int(min_interval_frames))
        labeled_df = self.apply_to_dataframe(df).reset_index(drop=True)
        timestamps = pd.to_datetime(labeled_df["timestamp"], errors="coerce")
        issues: list[dict] = []

        for group_name in self.schema.label_group_names():
            column = self.schema.label_column(group_name)
            values = labeled_df.get(
                column, pd.Series("", index=labeled_df.index, dtype="object")
            ).fillna("").astype(str).str.strip()
            missing = values.eq("") & timestamps.notna()
            positions = np.flatnonzero(missing.to_numpy())
            if len(positions):
                run_start = int(positions[0])
                run_end = run_start
                for position in positions[1:]:
                    position = int(position)
                    if position == run_end + 1:
                        run_end = position
                        continue
                    issues.append({
                        "code": "unlabeled_run",
                        "severity": "warning",
                        "group": group_name,
                        "start_time": pd.Timestamp(timestamps.iloc[run_start]),
                        "end_time": pd.Timestamp(timestamps.iloc[run_end]),
                        "frame_count": run_end - run_start + 1,
                    })
                    run_start = run_end = position
                issues.append({
                    "code": "unlabeled_run",
                    "severity": "warning",
                    "group": group_name,
                    "start_time": pd.Timestamp(timestamps.iloc[run_start]),
                    "end_time": pd.Timestamp(timestamps.iloc[run_end]),
                    "frame_count": run_end - run_start + 1,
                })

        for index, interval in enumerate(self.intervals):
            mask = (
                (timestamps >= pd.Timestamp(interval["start_time"]))
                & (timestamps <= pd.Timestamp(interval["end_time"]))
            )
            frame_count = int(mask.sum())
            if frame_count < min_frames:
                issues.append({
                    "code": "short_interval",
                    "severity": "info",
                    "group": self._interval_group(interval),
                    "start_time": pd.Timestamp(interval["start_time"]),
                    "end_time": pd.Timestamp(interval["end_time"]),
                    "frame_count": frame_count,
                    "interval_index": index,
                    "labels": self._interval_labels(interval),
                })

        for group_name in self.schema.label_group_names():
            if self.schema.label_group_mode(group_name) != "single":
                continue
            group_intervals = [
                (index, interval)
                for index, interval in enumerate(self.intervals)
                if self._interval_group(interval) == group_name
            ]
            for left_pos, (left_index, left) in enumerate(group_intervals):
                for right_index, right in group_intervals[left_pos + 1:]:
                    overlap_start = max(left["start_time"], right["start_time"])
                    overlap_end = min(left["end_time"], right["end_time"])
                    if overlap_start >= overlap_end:
                        continue
                    if self._interval_labels(left) == self._interval_labels(right):
                        continue
                    overlap_frames = int(
                        ((timestamps >= overlap_start) & (timestamps <= overlap_end)).sum()
                    )
                    issues.append({
                        "code": "single_group_overlap",
                        "severity": "warning",
                        "group": group_name,
                        "start_time": pd.Timestamp(overlap_start),
                        "end_time": pd.Timestamp(overlap_end),
                        "frame_count": overlap_frames,
                        "interval_indexes": [left_index, right_index],
                        "labels": [
                            self._interval_labels(left)[0],
                            self._interval_labels(right)[0],
                        ],
                    })
        return issues

    def apply_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将所有区间标签按标签组填充到 DataFrame，并保留默认组的 ``label`` 兼容列。

        边界过渡区: 相邻区间交界处 ±margin 的帧标记为 Ignore。
        """
        result = df.copy()
        if "timestamp" not in result.columns:
            return result
        group_names = self.schema.label_group_names()
        if not group_names:
            group_names = [self.schema.default_label_group()]
        for group_name in group_names:
            result[self.schema.label_column(group_name)] = ""
        margin = pd.Timedelta(seconds=self.boundary_margin)

        for iv in sorted(
            self.intervals,
            key=lambda item: int(item.get("_order", self.intervals.index(item))),
        ):
            group_name = self._interval_group(iv)
            if group_name not in group_names:
                continue
            labels = self._interval_labels(iv)
            if not labels:
                continue
            group_column = self.schema.label_column(group_name)
            mask = (result["timestamp"] >= iv["start_time"]) & (
                result["timestamp"] <= iv["end_time"]
            )
            if self.schema.label_group_mode(group_name) == "multi":
                def merge_labels(current: str) -> str:
                    combined = [item for item in str(current).split("|") if item]
                    for item in labels:
                        if item not in combined:
                            combined.append(item)
                    return "|".join(combined)

                result.loc[mask, group_column] = result.loc[
                    mask, group_column
                ].apply(merge_labels)
            else:
                result.loc[mask, group_column] = labels[0]

        # 单选组分别计算边界过渡区，避免一个组覆盖另一个组的标签。
        if self.boundary_enabled and len(self.intervals) > 1:
            sample_step = (
                result["timestamp"].sort_values().diff().dropna().median()
                if len(result) > 1 else pd.Timedelta(0)
            )
            for group_name in group_names:
                if self.schema.label_group_mode(group_name) != "single":
                    continue
                group_intervals = sorted(
                    [
                        interval for interval in self.intervals
                        if self._interval_group(interval) == group_name
                    ],
                    key=lambda item: item["start_time"],
                )
                for i in range(len(group_intervals) - 1):
                    current = group_intervals[i]
                    following = group_intervals[i + 1]
                    if self._interval_labels(current) == self._interval_labels(following):
                        continue
                    gap = following["start_time"] - current["end_time"]
                    if gap > max(sample_step * 2, pd.Timedelta(milliseconds=1)):
                        continue
                    boundary_time = current["end_time"] + gap / 2
                    left = boundary_time - margin
                    right = boundary_time + margin
                    mask = (result["timestamp"] >= left) & (
                        result["timestamp"] <= right
                    )
                    result.loc[
                        mask, self.schema.label_column(group_name)
                    ] = self.ignore_label

        default_column = self.schema.label_column(self.schema.default_label_group())
        result["label"] = result.get(
            default_column, pd.Series("", index=result.index, dtype="object")
        )
        return result

    def to_list(self) -> list[dict]:
        """返回可 JSON 序列化的区间列表。"""
        out = []
        for iv in self.intervals:
            item = dict(iv)
            item.pop("_order", None)
            item["start_time"] = pd.Timestamp(iv["start_time"]).isoformat()
            item["end_time"] = pd.Timestamp(iv["end_time"]).isoformat()
            out.append(item)
        return out


# =============================================================================
# 6. LLM / VLM 辅助标注 (Module D)
# =============================================================================

class LLMAssistant:
    """
    调用 OpenAI 兼容 API 进行传感器数据 + 图像的自动地形预判。

    支持配置 Base URL 以使用兼容服务 (如 Azure, 本地 vLLM 等)。
    """

    def __init__(self, schema: SensorSchema):
        cfg = schema.llm_assistant or {}
        self.enabled = cfg.get("enabled", False)
        self.api_key = cfg.get("api_key", "")
        self.base_url = cfg.get("base_url", "") or None
        self.model = cfg.get("model", "gpt-4o")
        self.temperature = float(cfg.get("temperature", 0.2))
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "LLM 辅助需要 openai 库, 请运行: pip install openai"
            ) from e
        kwargs = {"api_key": self.api_key, "timeout": 60.0, "max_retries": 1}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def extract_features(
        self, df: pd.DataFrame, start_time, end_time
    ) -> dict:
        """提取某区间的传感器统计特征用于 LLM Prompt。"""
        if "timestamp" not in df.columns:
            return {}
        mask = (df["timestamp"] >= pd.Timestamp(start_time)) & (
            df["timestamp"] <= pd.Timestamp(end_time)
        )
        sub = df.loc[mask]
        features = {}
        for col in sub.columns:
            if col in ("timestamp", "label"):
                continue
            if pd.api.types.is_numeric_dtype(sub[col]):
                s = sub[col].dropna()
                if len(s) == 0:
                    continue
                features[col] = {
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "range": float(s.max() - s.min()),
                }
        return features

    def _encode_image(self, image_path: str) -> str:
        """将图像编码为 base64。"""
        from PIL import Image
        img = Image.open(image_path)
        # 缩放到合理尺寸以减少 token
        max_dim = 512
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            img = img.resize(
                (int(img.size[0] * ratio), int(img.size[1] * ratio))
            )
        buf = io.BytesIO()
        img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def predict_label(
        self,
        features: dict,
        image_path: str | None = None,
        candidate_labels: list[str] | None = None,
    ) -> dict:
        """
        调用 LLM/VLM 进行地形预判。

        Args:
            features: extract_features 返回的统计特征
            image_path: 可选的图像路径 (VLM 输入)
            candidate_labels: 候选标签列表

        Returns:
            {"label": str, "reasoning": str, "confidence": float}
        """
        if not self.enabled or not self.api_key:
            return {
                "label": None,
                "reasoning": "LLM 未启用或 API Key 未配置",
                "confidence": 0.0,
            }

        client = self._get_client()
        candidates = candidate_labels or ["平地", "坡道", "楼梯", "草地", "沙地"]

        # 构造文本 Prompt
        feature_text = json.dumps(features, ensure_ascii=False, indent=2)
        text_prompt = (
            "你是可穿戴机器人地形感知数据标注专家。\n"
            "以下是某时间区间内多个传感器的统计特征 (均值/标准差/范围):\n"
            f"{feature_text}\n\n"
            f"请从以下标签中选择最可能的一个: {candidates}\n"
            "请以 JSON 格式返回: "
            '{"label": "标签名", "reasoning": "推理理由", "confidence": 0-1的置信度}'
        )

        messages = []
        if image_path and os.path.exists(image_path):
            # VLM 多模态输入
            try:
                img_b64 = self._encode_image(image_path)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            },
                        },
                    ],
                })
            except Exception:
                # 图像处理失败则退化为纯文本
                messages.append({"role": "user", "content": text_prompt})
        else:
            messages.append({"role": "user", "content": text_prompt})

        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=500,
            )
            content = (resp.choices[0].message.content or "").strip()
            # 尝试解析 JSON (容错: 提取 {} 部分)
            import re
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                result = json.loads(m.group(0))
                label = result.get("label")
                if label not in candidates:
                    return {
                        "label": None,
                        "reasoning": f"模型返回了候选集之外的标签: {label}",
                        "confidence": 0.0,
                    }
                return {
                    "label": label,
                    "reasoning": result.get("reasoning", ""),
                    "confidence": min(1.0, max(0.0, float(result.get("confidence", 0.5)))),
                }
            return {
                "label": None,
                "reasoning": content,
                "confidence": 0.0,
            }
        except Exception as e:
            return {
                "label": None,
                "reasoning": f"API 调用失败: {e}",
                "confidence": 0.0,
            }


# =============================================================================
# 7. 统一导出 (Module E)
# =============================================================================

class Exporter:
    """将对齐且标注完毕的数据导出为 PyTorch 友好的格式。"""

    @staticmethod
    def to_hdf5(
        df: pd.DataFrame,
        intervals: list[dict],
        schema: SensorSchema,
        output_path: str,
    ) -> None:
        """
        导出为 HDF5 文件 (PyTorch 友好)。

        HDF5 结构:
        /aligned_data          对齐后的主表 (timestamp + 所有传感器数值 + label)
        /intervals            标注区间列表
        /schema               传感器 Schema (JSON)
        """
        import h5py

        with h5py.File(output_path, "w") as f:
            # 主表: 数值数据
            numeric_df = df.select_dtypes(include=[np.number, "bool"]).copy()
            default_group = schema.default_label_group()
            label_mapping = {
                name: index for index, name in enumerate(
                    schema.label_names(default_group)
                )
            }
            label_mapping.setdefault(schema.boundary.get("ignore_label", "Ignore"), -2)
            if "label" in df.columns:
                numeric_df["label_id"] = (
                    df["label"].map(label_mapping).fillna(-1).astype(np.int32)
                )
            group_mappings: dict[str, dict[str, int]] = {}
            for group_name in schema.label_group_names():
                if schema.label_group_mode(group_name) != "single":
                    continue
                group_column = schema.label_column(group_name)
                if group_column not in df.columns:
                    continue
                mapping = {
                    name: index for index, name in enumerate(
                        schema.label_names(group_name)
                    )
                }
                mapping.setdefault(schema.boundary.get("ignore_label", "Ignore"), -2)
                group_mappings[group_name] = mapping
                numeric_df[f"label_id__{group_name}"] = (
                    df[group_column].map(mapping).fillna(-1).astype(np.int32)
                )

            # 时间戳转为 float (Unix 秒)
            if "timestamp" in df.columns:
                timestamps = (
                    pd.to_datetime(df["timestamp"]).astype("int64").values
                    / 1e9
                )
            else:
                timestamps = np.arange(len(df), dtype=float)

            grp = f.create_group("aligned_data")
            grp.create_dataset("timestamp", data=timestamps)
            column_mapping: dict[str, str] = {"timestamp": "timestamp"}
            used_keys = {"timestamp", "label_str"}

            def dataset_key(column: Any) -> str:
                base = str(column).replace("/", "_").replace("\\", "_").replace("\x00", "")
                base = base or "unnamed"
                key = base
                suffix = 2
                while key in used_keys:
                    key = f"{base}_{suffix}"
                    suffix += 1
                used_keys.add(key)
                column_mapping[key] = str(column)
                return key

            for col in numeric_df.columns:
                grp.create_dataset(dataset_key(col), data=numeric_df[col].to_numpy())

            for group_name in schema.label_group_names():
                if schema.label_group_mode(group_name) != "multi":
                    continue
                group_column = schema.label_column(group_name)
                if group_column not in df.columns:
                    continue
                group_labels = schema.label_names(group_name)
                label_indexes = {name: index for index, name in enumerate(group_labels)}
                multi_hot = np.zeros((len(df), len(group_labels)), dtype=np.uint8)
                for row_index, value in enumerate(df[group_column].fillna("").astype(str)):
                    for label_name in value.split("|"):
                        if label_name in label_indexes:
                            multi_hot[row_index, label_indexes[label_name]] = 1
                multi_dataset = grp.create_dataset(
                    dataset_key(f"label_multihot__{group_name}"), data=multi_hot
                )
                multi_dataset.attrs["label_mapping"] = json.dumps(
                    label_indexes, ensure_ascii=False
                )

            # 图像文件名及其他字符串元数据也属于对齐结果，不能在导出时丢失。
            string_columns = [
                column for column in df.columns
                if column not in numeric_df.columns and column not in {"timestamp", "label"}
            ]
            string_dtype = h5py.string_dtype(encoding="utf-8")
            for column in string_columns:
                values = df[column].fillna("").astype(str).to_numpy(dtype=object)
                grp.create_dataset(dataset_key(column), data=values, dtype=string_dtype)

            # 标签字符串 (变长字符串)
            if "label" in df.columns:
                labels = df["label"].astype(str).values
                grp.create_dataset("label_str", data=labels, dtype=string_dtype)
                grp.attrs["label_mapping"] = json.dumps(
                    label_mapping, ensure_ascii=False
                )
            grp.attrs["label_group_mappings"] = json.dumps(
                group_mappings, ensure_ascii=False
            )
            grp.attrs["column_mapping"] = json.dumps(
                column_mapping, ensure_ascii=False
            )

            # 区间列表
            iv_grp = f.create_group("intervals")
            for i, iv in enumerate(intervals):
                ig = iv_grp.create_group(f"interval_{i}")
                ig.attrs["start_time"] = str(iv.get("start_time", ""))
                ig.attrs["end_time"] = str(iv.get("end_time", ""))
                ig.attrs["label"] = str(iv.get("label", ""))
                ig.attrs["group"] = str(iv.get("group", ""))
                ig.attrs["labels"] = json.dumps(
                    iv.get("labels", [iv.get("label", "")]), ensure_ascii=False
                )

            # Schema (JSON)
            schema_json = json.dumps(schema.to_dict(), ensure_ascii=False)
            f.attrs["schema"] = schema_json
            f.attrs["export_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def to_master_csv(
        df: pd.DataFrame, intervals: list[dict], output_path: str
    ) -> None:
        """导出为单张 master CSV (对齐后主表 + 区间标注)。"""
        out = df.copy()
        if "timestamp" in out.columns:
            out["timestamp"] = out["timestamp"].astype(str)
        out.to_csv(output_path, index=False)

        # 区间信息保存到同目录的 _intervals.json
        iv_path = os.path.splitext(output_path)[0] + "_intervals.json"
        with open(iv_path, "w", encoding="utf-8") as fp:
            json.dump(intervals, fp, ensure_ascii=False, indent=2)

    @staticmethod
    def export_schema(schema: SensorSchema, output_path: str) -> None:
        """导出传感器 Schema 到 YAML。"""
        schema.to_yaml(output_path)


# =============================================================================
# 8. 便捷聚合: DataEngine
# =============================================================================

class DataEngine:
    """
    聚合所有子模块的统一入口, 供 Streamlit app.py 调用。

    用法:
        engine = DataEngine(schema)
        engine.load_all(device_data_dict)
        aligned_df = engine.align()
        bkps = engine.detect_change_points("imu_thigh.accel_z")
        engine.annotator.add_interval(t1, t2, "坡道", aligned_df)
        Exporter.to_hdf5(aligned_df, engine.annotator.intervals, schema, path)
    """

    def __init__(self, schema: SensorSchema):
        self.schema = schema
        self.loader = DataLoader()
        self.alignment = AlignmentEngine(schema)
        self.cpd = ChangePointDetector(schema)
        self.annotator = IntervalAnnotator(schema)
        self.llm = LLMAssistant(schema)
        self.device_data: dict[str, pd.DataFrame] = {}
        self.aligned_df: pd.DataFrame | None = None

    def load_all(self, device_data: dict[str, pd.DataFrame]) -> None:
        """设置已加载的设备数据 (由 app.py 负责实际加载)。"""
        self.device_data = device_data

    def align(self, tolerance_seconds: float = 0.1) -> pd.DataFrame:
        """执行主轨对齐。"""
        self.aligned_df = self.alignment.align(
            self.device_data, tolerance_seconds
        )
        return self.aligned_df

    def detect_change_points(
        self, column: str, n_bkps: int | None = None,
        pen: float | None = None,
    ) -> list[int]:
        """变点检测。"""
        if self.aligned_df is None:
            raise RuntimeError("请先执行 align() 再做变点检测")
        return self.cpd.detect(self.aligned_df, column, n_bkps, pen)

    def auto_segment(
        self, column: str, n_bkps: int | None = None,
        pen: float | None = None,
    ) -> list[dict]:
        """自动变点分割为区间列表。"""
        if self.aligned_df is None:
            raise RuntimeError("请先执行 align()")
        bkps = self.detect_change_points(column, n_bkps, pen)
        return self.cpd.segments_to_intervals(self.aligned_df, bkps)

    def export_hdf5(self, output_path: str) -> None:
        """导出为 HDF5。"""
        if self.aligned_df is None:
            raise RuntimeError("请先执行 align()")
        labeled = self.annotator.apply_to_dataframe(self.aligned_df)
        Exporter.to_hdf5(
            labeled, self.annotator.to_list(), self.schema, output_path
        )

    def export_csv(self, output_path: str) -> None:
        """导出为 master CSV。"""
        if self.aligned_df is None:
            raise RuntimeError("请先执行 align()")
        labeled = self.annotator.apply_to_dataframe(self.aligned_df)
        Exporter.to_master_csv(
            labeled, self.annotator.to_list(), output_path
        )

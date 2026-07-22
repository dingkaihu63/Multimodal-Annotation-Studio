"""
timeline_ui.py
============================
多轨时间轴与画布 (Step 2)

根据 Schema 动态生成多轨 Plotly 交互折线图, 共享同一个 X 轴 (时间轴 T)。

核心特性:
- 动态多轨: 每个数值传感器设备生成一条独立子图, 共享 X 轴
- 播放指针 (Playhead): 红色垂直线表示当前时间点, 拖动时联动图像与数值
- 区间选择框 (Span Selector): 鼠标拖选时间范围 [T_start, T_end]
- 区间标注可视化: 在波形上叠加彩色半透明矩形显示已标注区间
- 变点标记: 在波形上叠加垂直虚线显示自动检测的变点
- 主轨网格: 以主设备时间戳为刻度
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_engine import SensorSchema


def _display_identifier(value: str) -> str:
    """Convert schema identifiers into readable IDE-style labels."""
    acronyms = {"imu", "fsr", "rgb", "rgbd", "lidar", "gps", "emg", "tof"}
    parts = str(value).replace("-", "_").split("_")
    return " ".join(
        part.upper() if part.lower() in acronyms else part[:1].upper() + part[1:]
        for part in parts if part
    ) or str(value)


class TimelineRenderer:
    """
    多轨时间轴渲染器。

    根据 Schema 动态构建 Plotly 子图, 每个数值设备的每个数值列对应一条折线。
    所有子图共享 X 轴 (时间), 支持缩放/平移联动。
    """

    def __init__(
        self,
        schema: SensorSchema,
        language: str = "zh-CN",
        line_width: float = 1.2,
        palette: str = "colorblind",
        show_range_slider: bool = True,
        friendly_names: bool = True,
    ):
        self.schema = schema
        self.language = language
        self.line_width = line_width
        self.show_range_slider = show_range_slider
        self.friendly_names = friendly_names
        palettes = {
            "colorblind": [
                "#0072B2", "#E69F00", "#009E73", "#D55E00",
                "#CC79A7", "#56B4E9", "#6B6B67", "#F0E442",
            ],
            "classic": [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            ],
            "monochrome": [
                "#1F6F50", "#3E8067", "#5E907D", "#7DA193",
                "#31594B", "#527064", "#73867D", "#222A27",
            ],
        }
        self.color_cycle = palettes.get(palette, palettes["colorblind"])

    def _label(self, value: str) -> str:
        return _display_identifier(value) if self.friendly_names else str(value)

    def _collect_tracks(self, df: pd.DataFrame) -> list[dict]:
        """
        从对齐后的 DataFrame 中收集所有需要绘制的轨道。

        返回: [{"device": str, "column": str, "full_name": str}, ...]
        """
        tracks = []
        for cfg in self.schema.devices:
            if cfg.device_type == "image":
                continue  # 图像设备不画折线
            for col in cfg.value_columns:
                full = f"{cfg.name}.{col}"
                if full in df.columns:
                    tracks.append({
                        "device": cfg.name,
                        "column": col,
                        "full_name": full,
                        "display_name": (
                            f"{self._label(cfg.name)}  /  {self._label(col)}"
                        ),
                    })
        # 兼容: 若没有带前缀的列, 用裸列名
        if not tracks:
            for col in df.columns:
                if col in ("timestamp", "label"):
                    continue
                if pd.api.types.is_numeric_dtype(df[col]):
                    tracks.append({
                        "device": "data",
                        "column": col,
                        "full_name": col,
                        "display_name": self._label(col),
                    })
        return tracks

    def render(
        self,
        df: pd.DataFrame,
        playhead_time=None,
        intervals: list[dict] | None = None,
        change_points: list[int] | None = None,
        selection: tuple | None = None,
        height_per_track: int = 180,
    ) -> go.Figure:
        """
        渲染多轨时间轴。

        Args:
            df: 对齐后的 DataFrame (含 timestamp 列)
            playhead_time: 当前播放指针时间 (datetime/Timestamp)
            intervals: 已标注区间列表
            change_points: 变点索引列表 (df 行索引)
            selection: 当前选中的时间范围 (start_time, end_time)
            height_per_track: 每条轨道高度 (像素)

        Returns:
            Plotly Figure 对象
        """
        tracks = self._collect_tracks(df)
        n = len(tracks)

        if n == 0:
            # 空状态
            fig = go.Figure()
            fig.update_layout(
                title=(
                    "No sensor data to display"
                    if self.language == "en-US"
                    else "暂无可绘制的传感器数据"
                ),
                xaxis={"visible": False},
                yaxis={"visible": False},
                height=300,
                template="plotly_white",
            )
            return fig

        # 创建共享 X 轴的子图
        titles = [t["display_name"] for t in tracks]
        fig = make_subplots(
            rows=n, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=titles,
        )

        ts = df["timestamp"] if "timestamp" in df.columns else df.iloc[:, 0]

        # 绘制每条轨道
        for i, track in enumerate(tracks, start=1):
            col = track["full_name"]
            display_name = track["display_name"]
            color = self.color_cycle[(i - 1) % len(self.color_cycle)]
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=df[col],
                    mode="lines",
                    name=display_name,
                    line={"color": color, "width": self.line_width},
                    hovertemplate=(
                        f"<b>{display_name}</b><br>"
                        + ("Time: %{x}<br>" if self.language == "en-US" else "时间: %{x}<br>")
                        + ("Value: %{y:.4f}" if self.language == "en-US" else "数值: %{y:.4f}")
                        + "<extra></extra>"
                    ),
                ),
                row=i, col=1,
            )

        # 叠加标注区间 (彩色半透明矩形)
        shapes = []
        if intervals:
            y_domains = self._compute_y_domains(n)
            for iv in intervals:
                color = iv.get("color", "#95A5A6")
                for row_idx in range(1, n + 1):
                    y0, y1 = y_domains[row_idx - 1]
                    shapes.append(dict(
                        type="rect",
                        xref="x" if row_idx == 1 else f"x{row_idx}",
                        yref="paper",
                        x0=iv["start_time"], x1=iv["end_time"],
                        y0=y0, y1=y1,
                        fillcolor=color,
                        opacity=0.18,
                        line={"width": 0},
                        layer="below",
                    ))

        # 叠加选中区间 (高亮框)
        if selection is not None and len(selection) == 2:
            t0, t1 = selection
            if t0 is not None and t1 is not None:
                y_domains = self._compute_y_domains(n)
                for row_idx in range(1, n + 1):
                    y0, y1 = y_domains[row_idx - 1]
                    shapes.append(dict(
                        type="rect",
                        xref="x" if row_idx == 1 else f"x{row_idx}",
                        yref="paper",
                        x0=min(t0, t1), x1=max(t0, t1),
                        y0=y0, y1=y1,
                        fillcolor="#FFC107",
                        opacity=0.25,
                        line={"color": "#FF6F00", "width": 1.5},
                        layer="above",
                    ))

        # 叠加变点 (垂直虚线)
        if change_points:
            y_domains = self._compute_y_domains(n)
            for bkp_idx in change_points:
                if 0 <= bkp_idx < len(df):
                    bkp_time = ts.iloc[bkp_idx]
                    for row_idx in range(1, n + 1):
                        y0, y1 = y_domains[row_idx - 1]
                        shapes.append(dict(
                            type="line",
                            xref="x" if row_idx == 1 else f"x{row_idx}",
                            yref="paper",
                            x0=bkp_time, x1=bkp_time,
                            y0=y0, y1=y1,
                            line={"color": "#E91E63", "width": 1.5,
                                  "dash": "dash"},
                            layer="above",
                        ))

        # 播放指针 (红色垂直线)
        if playhead_time is not None:
            y_domains = self._compute_y_domains(n)
            for row_idx in range(1, n + 1):
                y0, y1 = y_domains[row_idx - 1]
                shapes.append(dict(
                    type="line",
                    xref="x" if row_idx == 1 else f"x{row_idx}",
                    yref="paper",
                    x0=playhead_time, x1=playhead_time,
                    y0=y0, y1=y1,
                    line={"color": "#FF0000", "width": 2},
                    layer="above",
                ))

        # 布局配置
        total_height = max(400, n * height_per_track + 80)
        fig.update_layout(
            height=total_height,
            showlegend=False,
            template="plotly_white",
            margin={"l": 60, "r": 20, "t": 50, "b": 40},
            shapes=shapes,
            # 配置 dragmode 为 select 以支持区间选择
            dragmode="select",
            clickmode="event+select",
            uirevision="multimodal-timeline",
            hovermode="x unified",
            font={
                "family": "Segoe UI Variable Text, Segoe UI, Microsoft YaHei UI, sans-serif",
                "size": 11,
                "color": "#30302e",
            },
        )
        fig.update_annotations(font={"size": 11, "color": "#4f4f4b"})

        # 共享 X 轴配置 (只在最底层显示刻度)
        for i in range(1, n + 1):
            xaxis_key = "xaxis" if i == 1 else f"xaxis{i}"
            fig.update_layout(**{
                f"{xaxis_key}.type": "date",
                f"{xaxis_key}.rangeslider": {
                    "visible": i == n and self.show_range_slider,
                    "thickness": 0.08,
                } if i == n else {"visible": False},
                f"{xaxis_key}.showticklabels": i == n,
                f"{xaxis_key}.title": (
                    ("Time" if self.language == "en-US" else "时间")
                    if i == n else None
                ),
            })

        # Y 轴自适应
        for i in range(1, n + 1):
            yaxis_key = "yaxis" if i == 1 else f"yaxis{i}"
            fig.update_layout(**{
                f"{yaxis_key}.fixedrange": False,
                f"{yaxis_key}.title": "",
            })

        return fig

    def _compute_y_domains(self, n: int) -> list[tuple[float, float]]:
        """计算每个子图在 paper 坐标系中的 y 范围 (用于跨子图绘制形状)。"""
        # plotly 默认 vertical_spacing=0.02
        spacing = 0.02
        total_spacing = spacing * (n - 1)
        track_h = (1.0 - total_spacing) / n
        domains = []
        # 子图从上到下编号 1..n, paper 坐标 0 在底部
        for i in range(n):
            top = 1.0 - i * (track_h + spacing)
            bottom = top - track_h
            domains.append((bottom, top))
        return domains

    @staticmethod
    def extract_selection(
        plotly_event: dict | None,
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """
        从 Streamlit plotly 事件中提取选中的时间范围。

        Args:
            plotly_event: st.plotly_chart(..., on_select=...) 返回的 selection 字典

        Returns:
            (start_time, end_time) 或 None
        """
        if not plotly_event:
            return None
        selection = plotly_event.get("selection", {})
        ranges = selection.get("box", [])
        if not ranges:
            # 尝试 lasso
            ranges = selection.get("lasso", [])
        x_vals = []
        if ranges:
            x_vals = ranges[0].get("x", [])
        if len(x_vals) < 2:
            # 不同 Streamlit/Plotly 版本可能只返回被选中的点。
            x_vals = [
                point.get("x") for point in selection.get("points", [])
                if point.get("x") is not None
            ]
        if len(x_vals) < 2:
            return None
        try:
            times = [pd.Timestamp(value) for value in x_vals]
            return min(times), max(times)
        except Exception:
            return None

    @staticmethod
    def extract_click_time(
        plotly_event: dict | None,
    ) -> pd.Timestamp | None:
        """
        从点击事件中提取时间点 (用于移动播放指针)。

        Args:
            plotly_event: st.plotly_chart(..., on_select=...) 返回的字典

        Returns:
            Timestamp 或 None
        """
        if not plotly_event:
            return None
        points = plotly_event.get("selection", {}).get("points", [])
        if not points:
            return None
        x = points[0].get("x")
        if x is None:
            return None
        try:
            return pd.Timestamp(x)
        except Exception:
            return None


class ImagePreviewRenderer:
    """RGB / Depth 图像预览区渲染辅助。"""

    @staticmethod
    def get_image_path(
        df: pd.DataFrame,
        target_time,
        device: str,
        image_dir: str,
        image_column: str = "filename",
    ) -> str | None:
        """
        根据 target_time 找到最近的图像帧路径。

        Args:
            df: 对齐后的 DataFrame (含 timestamp 和 image 列)
            target_time: 目标时间
            device: 设备名前缀
            image_dir: 图像目录
            image_column: 图像文件名列名

        Returns:
            图像绝对路径或 None
        """
        if df.empty or "timestamp" not in df.columns:
            return None
        col = f"{device}.{image_column}"
        if col not in df.columns:
            col = image_column
        if col not in df.columns:
            return None

        ts = pd.to_datetime(df["timestamp"])
        target = pd.Timestamp(target_time)
        idx = (ts - target).abs().argmin()
        filename = df.iloc[idx][col]
        if pd.isna(filename):
            return None
        path = os.path.join(image_dir, str(filename))
        if os.path.exists(path):
            return path
        return None

    @staticmethod
    def depth_to_colormap(depth_path: str, max_depth: float = 5.0):
        """深度图转伪彩色图 (用于可视化)。"""
        import cv2
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            return None
        depth_f = depth.astype(np.float32)
        depth_f = np.clip(depth_f / (max_depth * 1000), 0, 1)
        depth_vis = (depth_f * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

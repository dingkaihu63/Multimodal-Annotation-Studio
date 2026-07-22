"""
app.py
============================
多模态数据对齐与区间标注 GUI 工具 (MVP) - Streamlit 前端 (Step 3)

集成 4 大核心区域:
1. 左上方: 多模态/视觉预览区 (Player Area) - RGB 图像 + Depth 伪彩 + 传感器数值卡片
2. 下半部分: 动态多轨道时间轴区 (Multi-track Timeline) - Plotly 交互折线图, 共享 X 轴
3. 右侧/顶部: 操作与控制面板 (Control Panel) - 对齐/打板/变点/标签/导出
4. 系统配置与 AI 助手面板 (Settings & LLM Panel) - 传感器动态配置 + LLM API

运行:
    streamlit run app.py

参考: PR/剪映 视频剪辑软件交互逻辑
"""

from __future__ import annotations

import os
import io
import json
import html
import hashlib
import zipfile
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml
from streamlit_shortcuts import add_shortcuts

from data_engine import (
    SensorSchema, SensorConfig, DataLoader, DataEngine,
    AlignmentEngine, ChangePointDetector, IntervalAnnotator,
    LLMAssistant, Exporter,
)
from timeline_ui import TimelineRenderer, ImagePreviewRenderer


# =============================================================================
# 页面配置 & 样式
# =============================================================================

st.set_page_config(
    page_title="Multimodal Studio",
    page_icon=":material/dataset:",
    layout="wide",
    initial_sidebar_state="auto",
)

UI_DEFAULTS = {
    "language": "zh-CN",
    "density": "comfortable",
    "friendly_device_names": True,
    "timeline_height": 160,
    "timeline_line_width": 1.2,
    "timeline_palette": "colorblind",
    "show_range_slider": True,
    "number_precision": 4,
    "preview_card_limit": 8,
    "alignment_tolerance": 0.1,
    "cpd_penalty": 10.0,
    "clear_selection_after_label": True,
}
if "ui_settings" not in st.session_state:
    st.session_state.ui_settings = dict(UI_DEFAULTS)
else:
    for setting_name, default_value in UI_DEFAULTS.items():
        st.session_state.ui_settings.setdefault(setting_name, default_value)

# Dialog submissions happen before the main workspace controls are created on
# the next run. Applying queued values here keeps both views in sync without
# mutating an already-instantiated Streamlit widget.
for widget_key, widget_value in st.session_state.pop("_settings_widget_sync", {}).items():
    st.session_state[widget_key] = widget_value


def tr(chinese: str, english: str) -> str:
    """根据当前界面语言返回文本。"""
    return english if st.session_state.ui_settings["language"] == "en-US" else chinese


def display_device_name(name: str) -> str:
    """把 Schema 标识符转成适合 IDE 资源列表展示的设备名。"""
    if not st.session_state.ui_settings.get("friendly_device_names", True):
        return str(name)
    acronyms = {"imu", "fsr", "rgb", "rgbd", "lidar", "gps", "emg", "tof"}
    words = str(name).replace("-", "_").split("_")
    return " ".join(
        word.upper() if word.lower() in acronyms else word[:1].upper() + word[1:]
        for word in words if word
    ) or str(name)


def device_type_label(device_type: str) -> str:
    labels = {
        "numeric": tr("数值", "Numeric"),
        "image": tr("图像", "Image"),
        "mixed": tr("混合", "Mixed"),
    }
    return labels.get(device_type, device_type)


def display_signal_name(column: str) -> str:
    if "." in str(column):
        device, channel = str(column).split(".", 1)
        return f"{display_device_name(device)} · {display_device_name(channel)}"
    return display_device_name(str(column))


def aligned_frame_filename(
    df: pd.DataFrame, frame_index: int, device: SensorConfig,
) -> str | None:
    """Return the aligned image filename for a frame and image device."""
    image_column = f"{device.name}.{device.image_column}"
    if image_column not in df.columns:
        image_column = device.image_column
    if image_column not in df.columns:
        return None
    filename = df.iloc[frame_index][image_column]
    return None if pd.isna(filename) else str(filename)


def aligned_frame_image_path(
    df: pd.DataFrame, frame_index: int, device: SensorConfig, *, depth: bool = False,
) -> str | None:
    """Resolve an RGB or depth image path for an aligned frame."""
    filename = aligned_frame_filename(df, frame_index, device)
    if filename is None:
        return None
    base_path = device.depth_path if depth else device.data_path
    if not base_path:
        return None
    candidates = [Path(base_path) / filename]
    if depth:
        candidates.insert(0, Path(base_path) / f"{Path(filename).stem}.png")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


density = st.session_state.ui_settings["density"]
content_gap = {"compact": "0.65rem", "comfortable": "1rem", "spacious": "1.35rem"}[density]

st.markdown(
    f"""
    <style>
    :root {{
        --studio-bg: #fbfbfa;
        --studio-sidebar: #f4f4f2;
        --studio-surface: #ffffff;
        --studio-border: #dededb;
        --studio-border-soft: #ececea;
        --studio-text: #20201e;
        --studio-muted: #6f6f6b;
        --studio-accent: #1f6f50;
    }}
    html, body, .stApp {{font-family: "Segoe UI Variable Text", "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;}}
    body *:not([data-testid="stIconMaterial"]):not([data-testid="stExpanderIcon"]):not([data-testid="stAlertDynamicIcon"]) {{font-family: "Segoe UI Variable Text", "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif !important;}}
    [data-testid="stIconMaterial"], [data-testid="stExpanderIcon"], [data-testid="stAlertDynamicIcon"] {{font-family: "Material Symbols Rounded" !important;}}
    code, pre {{font-family: "Segoe UI Variable Text", "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif !important;}}
    .stApp {{background: var(--studio-bg); color: var(--studio-text); font-size: .9rem;}}
    [data-testid="stSidebar"] {{background: var(--studio-sidebar); border-right: 1px solid var(--studio-border-soft);}}
    [data-testid="stAppDeployButton"] {{display: none;}}
    .block-container {{padding-top: 1.1rem; padding-bottom: 1.5rem;}}
    .stVerticalBlock {{gap: {content_gap};}}
    h1 {{font-size: 1.9rem !important; line-height: 1.2 !important; letter-spacing: 0 !important; margin-bottom: .15rem !important;}}
    h2 {{font-size: 1.28rem !important; letter-spacing: 0 !important;}}
    h3 {{font-size: 1rem !important; letter-spacing: 0 !important;}}
    p, label, button, input, textarea {{letter-spacing: 0 !important;}}
    [data-testid="stCaptionContainer"] {{color: var(--studio-muted);}}
    [data-testid="stHeader"] {{background: color-mix(in srgb, var(--studio-bg) 92%, transparent);}}
    [data-testid="stSidebar"] h2 {{font-size: 1rem !important; text-transform: none;}}
    [data-testid="stSidebar"] h3 {{font-size: .83rem !important; color: var(--studio-muted);}}
    [data-testid="stTabs"] [data-baseweb="tab-list"] {{gap: 1.25rem; border-bottom: 1px solid var(--studio-border-soft);}}
    [data-testid="stTabs"] [data-baseweb="tab"] {{height: 2.6rem; padding: 0 .15rem;}}
    [data-testid="stTabs"] [aria-selected="true"] {{color: var(--studio-text);}}
    .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{border-radius: 6px; border-color: var(--studio-border); box-shadow: none;}}
    .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{background: var(--studio-text); border-color: var(--studio-text); color: white;}}
    .stButton > button:hover, .stDownloadButton > button:hover {{border-color: #a8a8a3; color: var(--studio-text);}}
    [data-baseweb="input"] > div, [data-baseweb="select"] > div, [data-baseweb="textarea"] > div {{border-radius: 6px !important; border-color: var(--studio-border) !important;}}
    [data-testid="stExpander"] {{background: var(--studio-surface); border-color: var(--studio-border); border-radius: 6px;}}
    [data-testid="stMetric"] {{background: transparent; padding: .25rem 0;}}
    hr {{border-color: var(--studio-border-soft) !important;}}
    .sidebar-brand {{font-size: 1.02rem; font-weight: 650; color: var(--studio-text); padding: .25rem 0 0;}}
    .device-config-row {{display: flex; flex-direction: column; gap: .08rem; min-width: 0; padding: .15rem 0;}}
    .device-display-name {{font-family: "Segoe UI Variable Text", "Segoe UI", "Microsoft YaHei UI", sans-serif; font-size: .86rem; font-weight: 600; color: var(--studio-text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}}
    .device-meta {{font-size: .7rem; color: var(--studio-muted); text-transform: uppercase;}}
    .model-status {{display: flex; align-items: center; gap: .5rem; padding: .45rem 0;}}
    .model-status-dot {{width: .48rem; height: .48rem; border-radius: 50%; background: var(--studio-accent); flex: 0 0 auto;}}
    .model-status-title {{font-size: .82rem; font-weight: 600; color: var(--studio-text);}}
    .model-status-meta {{font-size: .7rem; color: var(--studio-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}}
    .st-key-clear_llm_api button {{width: 2.25rem; height: 2.25rem; padding: 0;}}
    .st-key-clear_llm_api button p {{font-size: 0;}}
    [data-testid="stDialog"] > div {{
        width: min(52rem, calc(100vw - 2rem)) !important;
        max-width: 52rem !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }}
    [role="dialog"] {{width: 100% !important; max-width: 100% !important; border-radius: 8px; overflow-x: hidden;}}
    [role="dialog"] [data-testid="stDialogBody"] {{padding-top: .25rem;}}
    [role="dialog"] [data-testid="stForm"] {{padding: 0; border: 0;}}
    [role="dialog"] [data-testid="stTabs"] [role="tablist"] {{
        display: flex;
        gap: .15rem;
        padding: .2rem;
        background: var(--studio-sidebar);
        border: 1px solid var(--studio-border-soft);
        border-radius: 6px;
    }}
    [role="dialog"] [data-testid="stTabs"] [data-testid="stTab"] {{
        flex: 1 1 0;
        min-width: 0;
        height: 2.25rem;
        padding: 0 .8rem;
        border-radius: 4px;
        justify-content: center;
    }}
    [role="dialog"] [data-testid="stTabs"] [role="tabpanel"] {{padding-top: 1rem;}}
    [role="dialog"] [data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {{background: var(--studio-surface); box-shadow: 0 1px 2px rgba(0,0,0,.06);}}
    [role="dialog"] [data-testid="stTabs"] .react-aria-SelectionIndicator {{display: none;}}
    @media (max-width: 640px) {{
        h1 {{font-size: 1.6rem !important;}}
        .block-container {{padding-left: 1rem; padding-right: 1rem;}}
        [data-testid="stDialog"] > div {{width: calc(100vw - 2rem) !important; max-width: calc(100vw - 2rem) !important;}}
        [role="dialog"] [data-testid="stTabs"] [role="tablist"] {{gap: 0; padding: .15rem;}}
        [role="dialog"] [data-testid="stTabs"] [data-testid="stTab"] {{padding: 0 .25rem;}}
        [role="dialog"] [data-testid="stTabs"] [data-testid="stTab"] p {{font-size: .78rem; white-space: nowrap;}}
        [role="dialog"] [data-testid="stHorizontalBlock"] {{flex-direction: column;}}
        [role="dialog"] [data-testid="stColumn"] {{width: 100% !important; flex: 1 1 auto !important;}}
    }}
    .interval-item {{
        padding: 0.4rem 0.6rem;
        margin: 0.2rem 0;
        border-left: 4px solid;
        background: var(--studio-sidebar);
        font-size: 0.85rem;
    }}
    .sensor-card {{
        background: var(--studio-surface);
        border: 1px solid var(--studio-border-soft);
        border-radius: 6px;
        padding: 0.5rem 0.7rem;
        text-align: center;
    }}
    .sensor-card .val {{font-size: 1rem; font-weight: 650; color: var(--studio-text);}}
    .sensor-card .name {{font-size: 0.72rem; color: var(--studio-muted);}}
    .st-key-alignment_current_frame [data-testid="stImage"] img {{max-height: 25rem; object-fit: contain; background: #111;}}
    .st-key-alignment_filmstrip [data-testid="stImage"] img {{width: 100%; aspect-ratio: 16 / 9; object-fit: cover; background: #111; border: 1px solid var(--studio-border-soft);}}
    .st-key-alignment_filmstrip [data-testid="stCaptionContainer"] {{font-size: .7rem; text-align: center;}}
    .st-key-alignment_previous_frame button, .st-key-alignment_next_frame button {{min-width: 2.5rem; height: 2.5rem; padding: 0;}}
    .st-key-alignment_previous_frame button p, .st-key-alignment_next_frame button p {{font-size: 0;}}
    .st-key-alignment_previous_frame [data-testid="stIconMaterial"], .st-key-alignment_next_frame [data-testid="stIconMaterial"] {{font-size: 1.2rem;}}
    .st-key-annotation_toolbar {{
        position: sticky;
        top: 3rem;
        z-index: 8;
        padding: .65rem .75rem .55rem;
        background: color-mix(in srgb, var(--studio-surface) 96%, transparent);
        border: 1px solid var(--studio-border);
        border-radius: 6px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, .05);
        backdrop-filter: blur(8px);
    }}
    .st-key-annotation_toolbar [data-testid="stMetric"] {{padding: 0;}}
    [class*="st-key-label_save_"] button, [class*="st-key-label_delete_"] button, [class*="st-key-group_save_"] button, [class*="st-key-group_delete_"] button, .st-key-annotation_manage_labels button {{min-width: 2.5rem; height: 2.5rem; padding: 0;}}
    [class*="st-key-label_save_"] button p, [class*="st-key-label_delete_"] button p, [class*="st-key-group_save_"] button p, [class*="st-key-group_delete_"] button p, .st-key-annotation_manage_labels button p {{font-size: 0;}}
    @media (max-width: 640px) {{
        .st-key-annotation_toolbar {{position: static; padding: .6rem;}}
        .st-key-annotation_status [data-testid="stHorizontalBlock"] {{
            flex-direction: row !important;
            align-items: center;
            gap: .5rem;
        }}
        .st-key-annotation_status [data-testid="stColumn"] {{
            width: auto !important;
            min-width: 0 !important;
            flex: 1 1 0 !important;
        }}
        .st-key-annotation_status [data-testid="stColumn"]:first-child {{flex: 3 1 0 !important;}}
        .st-key-annotation_status [data-testid="stMetricValue"] {{font-size: 1.25rem;}}
        .st-key-annotation_status [data-testid="stMetricLabel"] p {{font-size: .7rem;}}
        .st-key-alignment_filmstrip {{overflow-x: auto; padding-bottom: .35rem;}}
        .st-key-alignment_filmstrip [data-testid="stHorizontalBlock"] {{min-width: 42rem;}}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Session State 初始化
# =============================================================================

def init_state():
    """初始化 Streamlit session state。"""
    if "schema" not in st.session_state:
        # 默认加载示例 Schema
        default_path = os.path.join("config", "sensors_schema.yaml")
        if os.path.exists(default_path):
            st.session_state.schema = SensorSchema.from_yaml(default_path)
        else:
            st.session_state.schema = SensorSchema()
    if "engine" not in st.session_state:
        st.session_state.engine = DataEngine(st.session_state.schema)
    if "device_data" not in st.session_state:
        st.session_state.device_data = {}
    if "aligned_df" not in st.session_state:
        st.session_state.aligned_df = None
    if "playhead_time" not in st.session_state:
        st.session_state.playhead_time = None
    if "selection" not in st.session_state:
        st.session_state.selection = None
    if "change_points" not in st.session_state:
        st.session_state.change_points = []
    if "intervals" not in st.session_state:
        st.session_state.intervals = []
    if "annotation_undo_stack" not in st.session_state:
        st.session_state.annotation_undo_stack = []
    if "alignment_frame_index" not in st.session_state:
        st.session_state.alignment_frame_index = 0
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "对齐"
    if "schema_signature" not in st.session_state:
        st.session_state.schema_signature = _schema_signature(
            st.session_state.schema, include_llm=False
        )


def _schema_signature(schema: SensorSchema, include_llm: bool = True) -> str:
    data = schema.to_dict()
    if not include_llm:
        data.pop("llm_assistant", None)
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def sync_engine(clear_data: bool = False, clear_alignment: bool = False):
    """当 Schema 变化时, 重建 engine 并同步 annotator/llm。"""
    schema = st.session_state.schema
    old_intervals = [] if clear_alignment else st.session_state.intervals
    if clear_data:
        st.session_state.device_data = {}
    if clear_alignment:
        st.session_state.aligned_df = None
        st.session_state.playhead_time = None
        st.session_state.selection = None
        st.session_state.change_points = []
        st.session_state.intervals = []
        st.session_state.annotation_undo_stack = []
        st.session_state.alignment_frame_index = 0
    engine = DataEngine(schema)
    engine.load_all(st.session_state.device_data)
    engine.aligned_df = st.session_state.aligned_df
    engine.annotator.intervals = old_intervals
    st.session_state.engine = engine
    st.session_state.schema_signature = _schema_signature(schema, include_llm=False)


def safe_extract_zip(payload: bytes, target_dir: str) -> tuple[str, str | None]:
    """安全解压图像数据，并返回图像目录与可选时间戳 CSV。"""
    root = Path(target_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            destination = (root / member.filename).resolve()
            if root != destination and root not in destination.parents:
                raise ValueError(f"ZIP 包含非法路径: {member.filename}")
        archive.extractall(root)
    csv_files = sorted(root.rglob("*.csv"))
    image_files = sorted(
        path for pattern in ("*.png", "*.jpg", "*.jpeg")
        for path in root.rglob(pattern)
    )
    if not image_files:
        raise ValueError("ZIP 中未找到 PNG/JPG 图像")
    common_dir = Path(os.path.commonpath([str(path.parent) for path in image_files]))
    return str(common_dir), str(csv_files[0]) if csv_files else None


def enable_annotation_shortcuts() -> None:
    """启用 I/O 快捷键，同时避免在输入框中键入文字时误触。"""
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const win = window.parent.window;
        if (!win.__annotationShortcutInputGuard) {
          win.__annotationShortcutInputGuard = (event) => {
            const target = event.target;
            if (target && (target.matches('input, textarea, select') || target.isContentEditable)) {
              event.stopImmediatePropagation();
            }
          };
          doc.addEventListener('keydown', win.__annotationShortcutInputGuard, true);
        }
        </script>
        """,
        height=0,
        width=0,
    )
    add_shortcuts(mark_in="i", mark_out="o")


def export_path(filename: str, expected_suffix: str) -> tuple[str, str]:
    """将用户文件名限制在 exports 目录，并补齐扩展名。"""
    safe_name = Path(filename).name.strip()
    if not safe_name:
        raise ValueError("导出文件名不能为空")
    if Path(safe_name).suffix.lower() != expected_suffix:
        safe_name = f"{Path(safe_name).stem}{expected_suffix}"
    out_dir = Path("exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / safe_name), safe_name


def remember_annotation(interval: dict) -> None:
    """Record an applied interval so the latest annotation can be undone."""
    st.session_state.annotation_undo_stack.append((
        pd.Timestamp(interval["start_time"]),
        pd.Timestamp(interval["end_time"]),
        str(interval.get("group", st.session_state.schema.default_label_group())),
        tuple(interval.get("labels", [interval.get("label", "")])),
    ))


def undo_last_annotation() -> dict | None:
    """Remove the most recently applied interval that still exists."""
    annotator = st.session_state.engine.annotator
    while st.session_state.annotation_undo_stack:
        target = st.session_state.annotation_undo_stack.pop()
        for index in range(len(annotator.intervals) - 1, -1, -1):
            interval = annotator.intervals[index]
            identity = (
                pd.Timestamp(interval["start_time"]),
                pd.Timestamp(interval["end_time"]),
                str(interval.get("group", st.session_state.schema.default_label_group())),
                tuple(interval.get("labels", [interval.get("label", "")])),
            )
            legacy_identity = (
                identity[0], identity[1], str(interval.get("label", ""))
            )
            if identity == target or legacy_identity == target:
                removed = dict(interval)
                annotator.remove_interval(index)
                st.session_state.intervals = annotator.intervals
                return removed
    return None


def update_schema_label(index: int, name: str, color: str, group: str) -> None:
    """Update a label definition and migrate existing interval references."""
    schema = st.session_state.schema
    clean_name = name.strip()
    if not clean_name:
        raise ValueError(tr("标签名称不能为空", "Label name cannot be empty"))
    existing_names = [
        item["name"] for item_index, item in enumerate(schema.label_classes)
        if item_index != index
    ]
    if clean_name in existing_names:
        raise ValueError(tr("标签名称不能重复", "Label names must be unique"))

    old_name = str(schema.label_classes[index]["name"])
    old_group = str(
        schema.label_classes[index].get("group", schema.default_label_group())
    )
    if group not in schema.label_group_names():
        raise ValueError(tr("请选择有效的标签组", "Select a valid label group"))
    label_in_use = any(
        old_name in interval.get("labels", [interval.get("label", "")])
        for interval in st.session_state.engine.annotator.intervals
    )
    if group != old_group and label_in_use:
        raise ValueError(tr(
            "该标签已被区间使用；跨组移动前请先删除相关区间",
            "This label is in use; delete its intervals before moving it to another group",
        ))
    schema.label_classes[index] = {
        "name": clean_name, "color": color, "group": group,
    }
    for interval in st.session_state.engine.annotator.intervals:
        interval_labels = list(interval.get("labels", [interval.get("label", "")]))
        if old_name in interval_labels:
            interval_labels = [
                clean_name if item == old_name else item for item in interval_labels
            ]
            interval["labels"] = interval_labels
            interval["label"] = " + ".join(interval_labels)
            interval["group"] = old_group
            interval["color"] = st.session_state.schema.label_color(
                interval_labels[0]
            )
    st.session_state.intervals = st.session_state.engine.annotator.intervals
    migrated_stack = []
    for item in st.session_state.annotation_undo_stack:
        if len(item) == 4:
            start, end, item_group, labels = item
            migrated_labels = tuple(
                clean_name if label == old_name else label for label in labels
            )
            migrated_stack.append((
                start, end, old_group if old_name in labels else item_group,
                migrated_labels,
            ))
        else:
            start, end, label = item
            migrated_stack.append((
                start, end, old_group,
                (clean_name if label == old_name else label,),
            ))
    st.session_state.annotation_undo_stack = migrated_stack
    llm_result = st.session_state.get("llm_result")
    if llm_result and llm_result.get("label") == old_name:
        llm_result["label"] = clean_name
    st.session_state.schema_signature = _schema_signature(schema, include_llm=False)


def delete_schema_label(index: int) -> None:
    """Delete an unused label definition."""
    schema = st.session_state.schema
    label_name = str(schema.label_classes[index]["name"])
    if any(
        label_name in interval.get("labels", [interval.get("label", "")])
        for interval in st.session_state.engine.annotator.intervals
    ):
        raise ValueError(tr(
            "该标签仍被标注区间使用，请先删除区间或重命名标签",
            "This label is still used by intervals; delete those intervals or rename it first",
        ))
    schema.label_classes.pop(index)
    llm_result = st.session_state.get("llm_result")
    if llm_result and llm_result.get("label") == label_name:
        st.session_state.llm_result = None
    st.session_state.schema_signature = _schema_signature(schema, include_llm=False)


def update_label_group(index: int, name: str, mode: str) -> None:
    """Update a label group and migrate labels and intervals."""
    schema = st.session_state.schema
    clean_name = name.strip()
    if not clean_name:
        raise ValueError(tr("标签组名称不能为空", "Group name cannot be empty"))
    if mode not in {"single", "multi"}:
        raise ValueError(tr("标签组模式无效", "Invalid label group mode"))
    other_names = [
        group["name"] for group_index, group in enumerate(schema.label_groups)
        if group_index != index
    ]
    if clean_name in other_names:
        raise ValueError(tr("标签组名称不能重复", "Group names must be unique"))
    old_name = str(schema.label_groups[index]["name"])
    if mode == "single" and any(
        len(interval.get("labels", [interval.get("label", "")])) > 1
        for interval in st.session_state.engine.annotator.intervals
        if interval.get("group", schema.default_label_group()) == old_name
    ):
        raise ValueError(tr(
            "该组已有多标签区间，不能直接切换为单选",
            "This group contains multi-label intervals and cannot be changed to single-select",
        ))
    schema.label_groups[index] = {"name": clean_name, "mode": mode}
    for label_config in schema.label_classes:
        if label_config.get("group", old_name) == old_name:
            label_config["group"] = clean_name
    for interval in st.session_state.engine.annotator.intervals:
        if interval.get("group", old_name) == old_name:
            interval["group"] = clean_name
    migrated_stack = []
    for item in st.session_state.annotation_undo_stack:
        if len(item) == 4:
            start, end, item_group, labels = item
            migrated_stack.append((
                start,
                end,
                clean_name if item_group == old_name else item_group,
                labels,
            ))
        else:
            migrated_stack.append(item)
    st.session_state.annotation_undo_stack = migrated_stack
    st.session_state.schema_signature = _schema_signature(schema, include_llm=False)


def delete_label_group(index: int) -> None:
    """Delete an empty, unused label group."""
    schema = st.session_state.schema
    group_name = str(schema.label_groups[index]["name"])
    if any(label.get("group", group_name) == group_name for label in schema.label_classes):
        raise ValueError(tr(
            "请先移动或删除该组中的标签",
            "Move or delete the labels in this group first",
        ))
    if any(interval.get("group") == group_name for interval in st.session_state.engine.annotator.intervals):
        raise ValueError(tr("该标签组仍被区间使用", "This label group is still in use"))
    schema.label_groups.pop(index)
    st.session_state.schema_signature = _schema_signature(schema, include_llm=False)


init_state()
pending_notice = st.session_state.pop("pending_notice", None)
if pending_notice:
    st.toast(pending_notice)


@st.dialog(tr("设置", "Settings"), width="large")
def show_settings() -> None:
    """全局界面与标注基础设置。"""
    settings = st.session_state.ui_settings
    schema = st.session_state.schema
    cpd_config = schema.change_point_detection
    boundary_config = schema.boundary

    with st.form("ui_settings_form", border=False):
        general_tab, timeline_tab, processing_tab, annotation_tab = st.tabs([
            tr("常规", "General"),
            tr("时间轴", "Timeline"),
            tr("处理", "Processing"),
            tr("标注", "Annotation"),
        ])

        with general_tab:
            general_left, general_right = st.columns(2)
            with general_left:
                language = st.selectbox(
                    tr("界面语言", "Language"),
                    ["zh-CN", "en-US"],
                    index=["zh-CN", "en-US"].index(settings["language"]),
                    format_func=lambda value: "简体中文" if value == "zh-CN" else "English",
                )
            with general_right:
                density_options = ["compact", "comfortable", "spacious"]
                density_labels = {
                    "compact": tr("紧凑", "Compact"),
                    "comfortable": tr("标准", "Comfortable"),
                    "spacious": tr("宽松", "Spacious"),
                }
                interface_density = st.selectbox(
                    tr("界面密度", "Interface density"),
                    density_options,
                    index=density_options.index(settings["density"]),
                    format_func=lambda value: density_labels[value],
                )
            friendly_device_names = st.toggle(
                tr("将设备标识符显示为可读名称", "Show readable device names"),
                value=bool(settings["friendly_device_names"]),
            )

        with timeline_tab:
            timeline_left, timeline_right = st.columns(2)
            with timeline_left:
                timeline_height = st.slider(
                    tr("轨道高度", "Track height"),
                    120, 220, int(settings["timeline_height"]), 10,
                )
                timeline_line_width = st.slider(
                    tr("波形线宽", "Waveform line width"),
                    0.8, 2.4, float(settings["timeline_line_width"]), 0.2,
                )
                number_precision = st.slider(
                    tr("数值小数位", "Number precision"),
                    2, 6, int(settings["number_precision"]),
                )
            with timeline_right:
                palette_options = ["colorblind", "classic", "monochrome"]
                palette_labels = {
                    "colorblind": tr("色觉友好", "Colorblind safe"),
                    "classic": tr("经典多色", "Classic"),
                    "monochrome": tr("单色 IDE", "Monochrome IDE"),
                }
                timeline_palette = st.selectbox(
                    tr("轨道配色", "Track palette"),
                    palette_options,
                    index=palette_options.index(settings["timeline_palette"]),
                    format_func=lambda value: palette_labels[value],
                )
                preview_card_limit = st.number_input(
                    tr("数值卡片上限", "Value card limit"),
                    min_value=4,
                    max_value=24,
                    value=int(settings["preview_card_limit"]),
                    step=1,
                )
                show_range_slider = st.toggle(
                    tr("显示时间缩略轴", "Show timeline range slider"),
                    value=bool(settings["show_range_slider"]),
                )

        with processing_tab:
            processing_left, processing_right = st.columns(2)
            with processing_left:
                alignment_tolerance = st.number_input(
                    tr("默认对齐容差（秒）", "Default alignment tolerance (seconds)"),
                    min_value=0.01,
                    max_value=1.0,
                    value=float(settings["alignment_tolerance"]),
                    step=0.01,
                )
                cpd_penalty = st.number_input(
                    tr("默认变点惩罚项", "Default change-point penalty"),
                    min_value=0.1,
                    max_value=1000.0,
                    value=float(settings["cpd_penalty"]),
                    step=1.0,
                )
            with processing_right:
                cpd_min_size = st.number_input(
                    tr("最短区间样本数", "Minimum segment samples"),
                    min_value=2,
                    max_value=10000,
                    value=int(cpd_config.get("min_size", 30)),
                    step=1,
                )
                cpd_jump = st.number_input(
                    tr("变点检测步长", "Change-point detection jump"),
                    min_value=1,
                    max_value=1000,
                    value=int(cpd_config.get("jump", 5)),
                    step=1,
                )

        with annotation_tab:
            annotation_left, annotation_right = st.columns(2)
            with annotation_left:
                boundary_enabled = st.toggle(
                    tr("启用 Ignore 过渡区", "Enable Ignore transition region"),
                    value=bool(boundary_config.get("enabled", True)),
                )
                boundary_margin = st.number_input(
                    tr("过渡半径（秒）", "Transition margin (seconds)"),
                    min_value=0.0,
                    max_value=5.0,
                    value=float(boundary_config.get("margin_seconds", 0.5)),
                    step=0.1,
                    disabled=not boundary_enabled,
                )
            with annotation_right:
                ignore_label = st.text_input(
                    tr("Ignore 标签名称", "Ignore label name"),
                    value=str(boundary_config.get("ignore_label", "Ignore")),
                )
                clear_selection_after_label = st.toggle(
                    tr("应用标签后清除选区", "Clear selection after applying a label"),
                    value=bool(settings["clear_selection_after_label"]),
                )

        st.divider()
        save_col, reset_col = st.columns([3, 1])
        submitted = save_col.form_submit_button(
            tr("保存设置", "Save settings"),
            type="primary",
            use_container_width=True,
            icon=":material/check:",
        )
        reset = reset_col.form_submit_button(
            tr("恢复默认", "Restore defaults"),
            use_container_width=True,
            icon=":material/restart_alt:",
        )

    if reset:
        settings.clear()
        settings.update(UI_DEFAULTS)
        boundary_config.update({
            "enabled": True,
            "margin_seconds": 0.5,
            "ignore_label": "Ignore",
        })
        cpd_config.update({"min_size": 30, "jump": 5})
        st.session_state.engine.annotator.boundary_enabled = True
        st.session_state.engine.annotator.boundary_margin = 0.5
        st.session_state.engine.annotator.ignore_label = "Ignore"
        st.session_state.engine.cpd = ChangePointDetector(schema)
        st.session_state.schema_signature = _schema_signature(schema, include_llm=False)
        st.session_state._settings_widget_sync = {
            "alignment_tolerance_control": UI_DEFAULTS["alignment_tolerance"],
            "cpd_penalty_control": UI_DEFAULTS["cpd_penalty"],
        }
        st.rerun()

    if submitted:
        settings.update({
            "language": language,
            "density": interface_density,
            "friendly_device_names": friendly_device_names,
            "timeline_height": timeline_height,
            "timeline_line_width": timeline_line_width,
            "timeline_palette": timeline_palette,
            "show_range_slider": show_range_slider,
            "number_precision": number_precision,
            "preview_card_limit": preview_card_limit,
            "alignment_tolerance": alignment_tolerance,
            "cpd_penalty": cpd_penalty,
            "clear_selection_after_label": clear_selection_after_label,
        })
        boundary_config.update({
            "enabled": boundary_enabled,
            "margin_seconds": boundary_margin,
            "ignore_label": ignore_label.strip() or "Ignore",
        })
        cpd_config.update({"min_size": int(cpd_min_size), "jump": int(cpd_jump)})
        st.session_state.engine.annotator.boundary_enabled = boundary_enabled
        st.session_state.engine.annotator.boundary_margin = boundary_margin
        st.session_state.engine.annotator.ignore_label = boundary_config["ignore_label"]
        st.session_state.engine.cpd = ChangePointDetector(schema)
        st.session_state.schema_signature = _schema_signature(schema, include_llm=False)
        st.session_state._settings_widget_sync = {
            "alignment_tolerance_control": alignment_tolerance,
            "cpd_penalty_control": cpd_penalty,
        }
        st.rerun()


@st.dialog(tr("标签管理", "Label manager"), width="large")
def show_label_manager() -> None:
    """Manage label groups and their label definitions."""
    schema = st.session_state.schema

    groups_tab, labels_tab = st.tabs([
        tr("标签组", "Groups"), tr("标签", "Labels")
    ])

    with groups_tab:
        for index, group_config in enumerate(list(schema.label_groups)):
            group_name = str(group_config.get("name", ""))
            group_mode = str(group_config.get("mode", "single"))
            row = st.columns([3, 1.5, .55, .55], vertical_alignment="bottom")
            edited_group_name = row[0].text_input(
                tr("标签组名称", "Group name"),
                value=group_name,
                key=f"group_name_{index}_{group_name}",
            )
            edited_group_mode = row[1].selectbox(
                tr("选择方式", "Selection mode"),
                ["single", "multi"],
                index=["single", "multi"].index(group_mode),
                format_func=lambda value: (
                    tr("单选", "Single") if value == "single" else tr("多选", "Multiple")
                ),
                key=f"group_mode_{index}_{group_name}",
            )
            save_group = row[2].button(
                tr("保存", "Save"), icon=":material/check:",
                help=tr("保存标签组", "Save group"),
                key=f"group_save_{index}_{group_name}", use_container_width=True,
            )
            group_in_use = any(
                label.get("group", group_name) == group_name
                for label in schema.label_classes
            ) or any(
                interval.get("group") == group_name
                for interval in st.session_state.engine.annotator.intervals
            )
            delete_group = row[3].button(
                tr("删除", "Delete"), icon=":material/delete:",
                help=(
                    tr("请先清空组内标签", "Remove labels from this group first")
                    if group_in_use else tr("删除标签组", "Delete group")
                ),
                key=f"group_delete_{index}_{group_name}",
                use_container_width=True, disabled=group_in_use,
            )
            if save_group:
                try:
                    update_label_group(index, edited_group_name, edited_group_mode)
                    st.toast(tr("标签组已更新", "Group updated"))
                    st.rerun(scope="fragment")
                except ValueError as error:
                    st.error(str(error))
            if delete_group:
                try:
                    delete_label_group(index)
                    st.toast(tr("标签组已删除", "Group deleted"))
                    st.rerun(scope="fragment")
                except ValueError as error:
                    st.error(str(error))

        st.divider()
        with st.form("add_group_form", border=False, clear_on_submit=True):
            add_group_columns = st.columns([3, 1.5, 1.1], vertical_alignment="bottom")
            new_group_name = add_group_columns[0].text_input(
                tr("新标签组名称", "New group name")
            )
            new_group_mode = add_group_columns[1].selectbox(
                tr("选择方式", "Selection mode"), ["single", "multi"],
                format_func=lambda value: (
                    tr("单选", "Single") if value == "single" else tr("多选", "Multiple")
                ),
            )
            add_group = add_group_columns[2].form_submit_button(
                tr("添加", "Add"), icon=":material/add:", use_container_width=True,
            )
        if add_group:
            clean_group_name = new_group_name.strip()
            if not clean_group_name:
                st.error(tr("请输入标签组名称", "Enter a group name"))
            elif clean_group_name in schema.label_group_names():
                st.error(tr("标签组名称不能重复", "Group names must be unique"))
            else:
                schema.label_groups.append({
                    "name": clean_group_name, "mode": new_group_mode,
                })
                st.session_state.schema_signature = _schema_signature(
                    schema, include_llm=False
                )
                st.toast(tr("标签组已添加", "Group added"))
                st.rerun(scope="fragment")

    with labels_tab:
        if not schema.label_classes:
            st.info(tr("当前项目还没有标签", "This project has no labels yet"))
        group_names = schema.label_group_names()
        for index, label_config in enumerate(list(schema.label_classes)):
            label_name = str(label_config.get("name", ""))
            label_color = str(label_config.get("color", "#95A5A6"))
            label_group = str(label_config.get("group", schema.default_label_group()))
            row = st.columns([2.4, 1.6, 1, .55, .55], vertical_alignment="bottom")
            edited_name = row[0].text_input(
                tr("标签名称", "Label name"), value=label_name,
                key=f"label_name_{index}_{label_name}",
            )
            edited_group = row[1].selectbox(
                tr("标签组", "Group"), group_names,
                index=group_names.index(label_group),
                key=f"label_group_{index}_{label_name}",
            )
            edited_color = row[2].color_picker(
                tr("颜色", "Color"), value=label_color,
                key=f"label_color_{index}_{label_name}",
            )
            save_label = row[3].button(
                tr("保存", "Save"), icon=":material/check:",
                help=tr("保存标签", "Save label"),
                key=f"label_save_{index}_{label_name}", use_container_width=True,
            )
            label_in_use = any(
                label_name in interval.get("labels", [interval.get("label", "")])
                for interval in st.session_state.engine.annotator.intervals
            )
            delete_label = row[4].button(
                tr("删除", "Delete"), icon=":material/delete:",
                help=(
                    tr("标签正在使用，不能删除", "Label is in use and cannot be deleted")
                    if label_in_use else tr("删除标签", "Delete label")
                ),
                key=f"label_delete_{index}_{label_name}",
                use_container_width=True, disabled=label_in_use,
            )
            if save_label:
                try:
                    update_schema_label(index, edited_name, edited_color, edited_group)
                    st.toast(tr("标签已更新", "Label updated"))
                    st.rerun(scope="fragment")
                except ValueError as error:
                    st.error(str(error))
            if delete_label:
                try:
                    delete_schema_label(index)
                    st.toast(tr("标签已删除", "Label deleted"))
                    st.rerun(scope="fragment")
                except ValueError as error:
                    st.error(str(error))

        st.divider()
        with st.form("add_label_form", border=False, clear_on_submit=True):
            add_columns = st.columns([2.4, 1.6, 1, 1.1], vertical_alignment="bottom")
            new_label_name = add_columns[0].text_input(
                tr("新标签名称", "New label name")
            )
            new_label_group = add_columns[1].selectbox(
                tr("标签组", "Group"), group_names,
                disabled=not group_names,
            )
            new_label_color = add_columns[2].color_picker(
                tr("颜色", "Color"), "#2F7D5C", key="new_label_color"
            )
            add_label = add_columns[3].form_submit_button(
                tr("添加", "Add"), icon=":material/add:",
                use_container_width=True, disabled=not group_names,
            )
        if add_label:
            clean_name = new_label_name.strip()
            if not clean_name:
                st.error(tr("请输入标签名称", "Enter a label name"))
            elif clean_name in schema.label_names():
                st.error(tr("标签名称不能重复", "Label names must be unique"))
            else:
                schema.label_classes.append({
                    "name": clean_name,
                    "color": new_label_color,
                    "group": new_label_group,
                })
                st.session_state.schema_signature = _schema_signature(
                    schema, include_llm=False
                )
                st.toast(tr("标签已添加", "Label added"))
                st.rerun(scope="fragment")

    if st.button(
        tr("完成", "Done"), type="primary",
        icon=":material/check:", use_container_width=True,
    ):
        st.rerun()


@st.dialog(tr("模型 API", "Model API"), width="small")
def show_llm_config() -> None:
    """在独立弹窗中配置 OpenAI 兼容的模型接口。"""
    llm_config = st.session_state.schema.llm_assistant
    existing_key = str(llm_config.get("api_key", ""))
    with st.form("llm_api_form"):
        model_name = st.text_input(
            tr("模型名称", "Model name"),
            value=str(llm_config.get("model", "gpt-4o")),
            placeholder="gpt-4o",
        )
        api_url = st.text_input(
            tr("API 地址", "API URL"),
            value=str(llm_config.get("base_url", "") or "https://api.openai.com/v1"),
            placeholder="https://api.openai.com/v1",
        )
        api_key_input = st.text_input(
            "API Key",
            value="",
            type="password",
            placeholder=(
                tr("已配置，留空则保持不变", "Configured; leave blank to keep it")
                if existing_key else "sk-..."
            ),
        )
        submitted = st.form_submit_button(
            tr("保存 API 配置", "Save API configuration"),
            type="primary",
            icon=":material/check:",
            use_container_width=True,
        )

    if not submitted:
        return
    resolved_key = api_key_input.strip() or existing_key
    parsed_url = urlparse(api_url.strip())
    if not model_name.strip():
        st.error(tr("请输入模型名称", "Enter a model name"))
        return
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        st.error(tr("请输入有效的 HTTP(S) API 地址", "Enter a valid HTTP(S) API URL"))
        return
    if not resolved_key:
        st.error(tr("请输入 API Key", "Enter an API key"))
        return

    llm_config.update({
        "enabled": True,
        "model": model_name.strip(),
        "base_url": api_url.strip().rstrip("/"),
        "api_key": resolved_key,
        "temperature": float(llm_config.get("temperature", 0.2)),
    })
    st.session_state.engine.llm = LLMAssistant(st.session_state.schema)
    st.session_state.llm_result = None
    st.rerun()


# =============================================================================
# 标题栏
# =============================================================================

st.title(tr("多模态标注工作台", "Multimodal Annotation Studio"))


# =============================================================================
# 侧边栏: 系统配置与 AI 助手面板 (Settings & LLM Panel)
# =============================================================================

with st.sidebar:
    st.markdown('<div class="sidebar-brand">Multimodal Studio</div>', unsafe_allow_html=True)

    # ---------- Schema 加载/保存 ----------
    st.subheader(tr("项目配置", "Project configuration"))
    schema_file = st.file_uploader(
        tr("导入 YAML Schema", "Import YAML schema"), type=["yaml", "yml"],
        help=tr("上传文件以替换当前传感器配置", "Upload a file to replace the current sensor configuration"),
    )
    if schema_file is not None:
        try:
            payload = schema_file.getvalue()
            upload_digest = hashlib.sha256(payload).hexdigest()
            if upload_digest != st.session_state.get("schema_upload_digest"):
                data = yaml.safe_load(payload.decode("utf-8"))
                schema_from_file = SensorSchema.from_dict(data or {})
                schema_from_file.validate()
                st.session_state.schema = schema_from_file
                st.session_state.schema_upload_digest = upload_digest
                sync_engine(clear_data=True, clear_alignment=True)
                st.success(
                    tr(
                        f"Schema 已加载（{len(st.session_state.schema.devices)} 个设备）",
                        f"Schema loaded ({len(st.session_state.schema.devices)} devices)",
                    )
                )
        except Exception as e:
            st.error(tr(f"加载失败：{e}", f"Load failed: {e}"))

    # ---------- 传感器动态配置 (Sensor Schema Editor) ----------
    st.subheader(tr("传感器", "Sensors"))

    with st.expander(tr("管理设备", "Manage devices"), expanded=False, icon=":material/sensors:"):
        schema = st.session_state.schema

        # 显示已有设备
        to_remove = None
        for i, dev in enumerate(schema.devices):
            with st.container():
                cols = st.columns([3, 1])
                cols[0].markdown(
                    '<div class="device-config-row">'
                    f'<div class="device-display-name">{html.escape(display_device_name(dev.name))}</div>'
                    f'<div class="device-meta">{html.escape(device_type_label(dev.device_type))} · {dev.frequency_hz:g} Hz</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                if cols[1].button(
                    tr("删除", "Delete"), key=f"del_{i}",
                    help=tr(f"删除 {dev.name}", f"Delete {dev.name}"),
                    icon=":material/delete:",
                ):
                    to_remove = i

                with st.expander(
                    tr(
                        f"编辑 {display_device_name(dev.name)}",
                        f"Edit {display_device_name(dev.name)}",
                    ),
                    expanded=False,
                ):
                    old_name = dev.name
                    dev.name = st.text_input(
                        tr("设备名称", "Device name"), dev.name, key=f"name_{i}"
                    )
                    if schema.master_device == old_name and dev.name != old_name:
                        schema.master_device = dev.name
                    dev.device_type = st.selectbox(
                        tr("设备类型", "Device type"), ["numeric", "image", "mixed"],
                        index=["numeric", "image", "mixed"].index(dev.device_type),
                        key=f"type_{i}",
                    )
                    dev.frequency_hz = st.number_input(
                        tr("采样频率 (Hz)", "Sample rate (Hz)"), value=dev.frequency_hz,
                        min_value=0.1, step=1.0, key=f"freq_{i}",
                    )
                    dev.data_path = st.text_input(
                        tr("数据路径（CSV/图像目录）", "Data path (CSV/image directory)"), dev.data_path,
                        key=f"path_{i}",
                    )
                    dev.timestamp_column = st.text_input(
                        tr("时间戳列名", "Timestamp column"), dev.timestamp_column,
                        key=f"ts_{i}",
                    )
                    if dev.device_type in ("image", "mixed"):
                        dev.timestamp_file = st.text_input(
                            tr("时间戳文件（CSV，可选）", "Timestamp file (optional CSV)"), dev.timestamp_file,
                            key=f"ts_file_{i}",
                        )
                        dev.image_column = st.text_input(
                            tr("图像文件名列", "Image filename column"), dev.image_column,
                            key=f"img_{i}",
                        )
                        dev.depth_path = st.text_input(
                            tr("深度图目录（可选）", "Depth directory (optional)"), dev.depth_path,
                            key=f"depth_{i}",
                        )
                    val_str = st.text_input(
                        tr("数值列（逗号分隔）", "Value columns (comma separated)"), ", ".join(dev.value_columns),
                        key=f"vals_{i}",
                    )
                    dev.value_columns = [
                        v.strip() for v in val_str.split(",") if v.strip()
                    ]
                    dev.interpolation = st.selectbox(
                        tr("插值策略", "Interpolation"), ["linear", "nearest", "forward_fill"],
                        index=["linear", "nearest", "forward_fill"].index(
                            dev.interpolation
                        ) if dev.interpolation in ["linear", "nearest", "forward_fill"] else 0,
                        key=f"interp_{i}",
                    )

        if to_remove is not None:
            removed = schema.devices.pop(to_remove)
            if schema.master_device == removed.name:
                schema.master_device = schema.devices[0].name if schema.devices else ""
            sync_engine(clear_data=True, clear_alignment=True)
            st.rerun()

        # 添加新设备
        st.markdown(f"**{tr('添加设备', 'Add device')}**")
        with st.form("add_device_form", clear_on_submit=True):
            new_name = st.text_input(tr("设备名称", "Device name"), "new_sensor")
            new_type = st.selectbox(tr("类型", "Type"), ["numeric", "image", "mixed"])
            new_freq = st.number_input(tr("采样频率 (Hz)", "Sample rate (Hz)"), value=50.0, step=1.0)
            new_path = st.text_input(tr("数据路径", "Data path"), "")
            new_ts = st.text_input(tr("时间戳列名", "Timestamp column"), "timestamp")
            new_vals = st.text_input(tr("数值列（逗号分隔）", "Value columns (comma separated)"), "")
            new_interp = st.selectbox(
                tr("插值策略", "Interpolation"), ["linear", "nearest", "forward_fill"]
            )
            if st.form_submit_button(tr("添加设备", "Add device"), icon=":material/add:"):
                schema.devices.append(SensorConfig(
                    name=new_name, device_type=new_type,
                    frequency_hz=new_freq, data_path=new_path,
                    timestamp_column=new_ts,
                    value_columns=[
                        v.strip() for v in new_vals.split(",") if v.strip()
                    ],
                    interpolation=new_interp,
                ))
                if not schema.master_device:
                    schema.master_device = new_name
                schema.validate()
                sync_engine(clear_data=True, clear_alignment=True)
                st.success(tr(f"已添加设备：{new_name}", f"Device added: {new_name}"))
                st.rerun()

    # 主设备选择
    st.subheader(tr("主轨", "Master track"))
    device_names = [d.name for d in st.session_state.schema.devices]
    if device_names:
        cur_idx = device_names.index(
            st.session_state.schema.master_device
        ) if st.session_state.schema.master_device in device_names else 0
        new_master = st.selectbox(
            tr("主设备", "Master device"), device_names,
            index=cur_idx, help=tr("其时间戳作为统一对齐网格", "Its timestamps define the shared alignment grid"),
            format_func=display_device_name,
        )
        if new_master != st.session_state.schema.master_device:
            st.session_state.schema.master_device = new_master
            sync_engine(clear_alignment=True)

    # ---------- 标签配置 ----------
    st.subheader(tr("标签", "Labels"))
    if st.button(
        tr(
            f"管理标签（{len(st.session_state.schema.label_classes)}）",
            f"Manage labels ({len(st.session_state.schema.label_classes)})",
        ),
        icon=":material/label:",
        use_container_width=True,
        key="sidebar_manage_labels",
    ):
        show_label_manager()

    # ---------- LLM API 配置 ----------
    st.subheader(tr("模型助手", "Model assistant"))
    llm_cfg = st.session_state.schema.llm_assistant
    llm_configured = bool(
        llm_cfg.get("enabled")
        and llm_cfg.get("model")
        and llm_cfg.get("base_url")
        and llm_cfg.get("api_key")
    )
    if llm_configured:
        api_host = urlparse(str(llm_cfg.get("base_url"))).netloc
        st.markdown(
            '<div class="model-status">'
            '<span class="model-status-dot"></span>'
            '<div style="min-width:0">'
            f'<div class="model-status-title">{html.escape(str(llm_cfg.get("model")))}</div>'
            f'<div class="model-status-meta">{html.escape(api_host)}</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        model_config_col, model_clear_col = st.columns([4, 1])
        if model_config_col.button(
            tr("修改 API 配置", "Edit API configuration"),
            icon=":material/tune:",
            use_container_width=True,
            key="configure_llm_api",
        ):
            show_llm_config()
        if model_clear_col.button(
            tr("清除", "Clear"),
            icon=":material/link_off:",
            help=tr("清除 API Key", "Clear API key"),
            key="clear_llm_api",
        ):
            llm_cfg["enabled"] = False
            llm_cfg["api_key"] = ""
            st.session_state.engine.llm = LLMAssistant(st.session_state.schema)
            st.session_state.llm_result = None
            st.rerun()
    elif st.button(
        tr("配置模型 API", "Configure model API"),
        icon=":material/key:",
        use_container_width=True,
        key="configure_llm_api",
    ):
        show_llm_config()

    # ---------- Schema 保存 ----------
    st.subheader(tr("工作区", "Workspace"))
    if st.button(
        tr("导出 Schema", "Export schema"),
        icon=":material/download:",
        use_container_width=True,
    ):
        out = os.path.join("config", "sensors_schema_export.yaml")
        os.makedirs("config", exist_ok=True)
        st.session_state.schema.to_yaml(out)
        st.success(tr(f"已保存到 {out}", f"Saved to {out}"))

    if st.button(
        tr("设置", "Settings"),
        icon=":material/settings:",
        use_container_width=True,
        key="open_settings",
    ):
        show_settings()

    # 文本控件会直接修改 dataclass；结构变化后在同一轮重建引擎，避免旧配置残留。
    try:
        st.session_state.schema.validate()
        current_signature = _schema_signature(
            st.session_state.schema, include_llm=False
        )
        if current_signature != st.session_state.schema_signature:
            sync_engine(clear_data=True, clear_alignment=True)
        else:
            # LLM 配置不影响已对齐数据，但需要即时刷新客户端。
            st.session_state.engine.llm = LLMAssistant(st.session_state.schema)
    except ValueError as exc:
        st.error(tr(f"Schema 配置无效：{exc}", f"Invalid schema: {exc}"))


# =============================================================================
# 主区域: Tabs 分区
# =============================================================================

tab_data, tab_align, tab_annotate, tab_export = st.tabs([
    tr("数据", "Data"),
    tr("对齐与分割", "Align & segment"),
    tr("标注", "Annotate"),
    tr("导出", "Export"),
])


# -----------------------------------------------------------------------------
# Tab 1: 数据加载
# -----------------------------------------------------------------------------

with tab_data:
    st.header(tr("数据源", "Data sources"))

    schema = st.session_state.schema
    device_data = {}

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(tr("设备数据", "Device data"))
        loaded_count = 0
        for dev in schema.devices:
            device_label = display_device_name(dev.name)
            with st.expander(
                f"{device_label}  ·  {device_type_label(dev.device_type)}  ·  {dev.frequency_hz:g} Hz"
            ):
                st.text(tr(f"路径：{dev.data_path}", f"Path: {dev.data_path}"))
                # 允许用户上传文件覆盖配置路径
                uploaded = st.file_uploader(
                    tr(f"上传 {device_label} 数据文件", f"Upload data for {device_label}"),
                    type=["csv"] if dev.device_type == "numeric" else ["csv", "zip"],
                    key=f"upload_{dev.name}",
                )
                if uploaded is not None:
                    tmp_dir = os.path.join("data", dev.name)
                    os.makedirs(tmp_dir, exist_ok=True)
                    payload = uploaded.getvalue()
                    if uploaded.name.lower().endswith(".zip"):
                        image_dir, timestamp_file = safe_extract_zip(payload, tmp_dir)
                        dev.data_path = image_dir
                        if timestamp_file:
                            dev.timestamp_file = timestamp_file
                    else:
                        tmp_path = os.path.join(tmp_dir, os.path.basename(uploaded.name))
                        with open(tmp_path, "wb") as f:
                            f.write(payload)
                        if dev.device_type == "numeric" or (
                            dev.device_type == "mixed" and dev.value_columns
                        ):
                            dev.data_path = tmp_path
                        else:
                            dev.timestamp_file = tmp_path
                    st.success(tr(f"已加载：{uploaded.name}", f"Loaded: {uploaded.name}"))

                # 尝试加载
                has_source = (
                    bool(dev.data_path and os.path.exists(dev.data_path))
                    or bool(dev.timestamp_file and os.path.exists(dev.timestamp_file))
                )
                if has_source:
                    try:
                        df = DataLoader.load(dev)
                        if not df.empty:
                            device_data[dev.name] = df
                            st.success(
                                tr(
                                    f"{device_label}：{len(df)} 行，{df.shape[1]} 列",
                                    f"{device_label}: {len(df)} rows, {df.shape[1]} columns",
                                )
                            )
                            with st.expander(tr(f"预览 {device_label}", f"Preview {device_label}")):
                                st.dataframe(df.head(10), use_container_width=True)
                            loaded_count += 1
                        else:
                            st.warning(tr(f"{device_label}：数据为空", f"{device_label}: no data"))
                    except Exception as e:
                        st.error(tr(f"{device_label} 加载失败：{e}", f"Failed to load {device_label}: {e}"))
                else:
                    st.info(tr(
                        f"请配置 {device_label} 的数据路径或上传文件",
                        f"Configure a data path or upload a file for {device_label}",
                    ))

    with col2:
        st.subheader(tr("状态", "Status"))
        st.metric(tr("已配置设备", "Configured devices"), len(schema.devices))
        st.metric(tr("已加载数据", "Loaded sources"), loaded_count)
        st.metric(
            tr("主设备", "Master device"),
            display_device_name(schema.master_device) if schema.master_device else tr("未设置", "Not set"),
        )

        if st.button(
            tr("重新加载", "Reload"), type="primary",
            icon=":material/refresh:", use_container_width=True,
        ):
            st.rerun()

        st.divider()
        st.subheader(tr("演示数据", "Demo data"))
        if st.button(
            tr("生成演示数据", "Generate demo data"),
            icon=":material/science:", use_container_width=True,
        ):
            from generate_sample_data import generate_all
            generate_all()
            st.success(tr("演示数据已生成，请重新加载", "Demo data generated. Reload to continue."))
            st.rerun()

    # 保存到 session
    if loaded_count > 0:
        st.session_state.device_data = device_data
        st.session_state.engine.load_all(device_data)
        with st.expander(tr("加载详情", "Load details")):
            st.json({k: {"rows": len(v), "cols": list(v.columns)}
                     for k, v in device_data.items()})


# -----------------------------------------------------------------------------
# Tab 2: 对齐 & 变点检测
# -----------------------------------------------------------------------------

with tab_align:
    st.header(tr("对齐与自动分割", "Alignment and auto-segmentation"))

    if not st.session_state.device_data:
        st.warning(tr("请先在“数据”中加载设备数据", "Load device data in the Data tab first"), icon=":material/info:")
    else:
        ctrl_col, info_col = st.columns([1, 2])

        with ctrl_col:
            st.subheader(tr("处理", "Processing"))

            # 一键对齐
            tol = st.slider(
                tr("对齐容差（秒）", "Alignment tolerance (seconds)"),
                0.01, 1.0, float(st.session_state.ui_settings["alignment_tolerance"]), 0.01,
                help=tr("超出容差的样本不会参与对齐", "Samples beyond this tolerance are left unmatched"),
                key="alignment_tolerance_control",
            )
            if st.button(
                tr("对齐数据", "Align data"), type="primary",
                icon=":material/sync:", use_container_width=True,
            ):
                with st.spinner(tr("正在对齐设备时间戳…", "Aligning device timestamps…")):
                    try:
                        aligned = st.session_state.engine.align(tol)
                        st.session_state.aligned_df = aligned
                        # 初始化播放指针到起始
                        if "timestamp" in aligned.columns and len(aligned) > 0:
                            st.session_state.playhead_time = aligned["timestamp"].iloc[0]
                        st.session_state.alignment_frame_index = 0
                        st.success(
                            tr(
                                f"对齐完成：{len(aligned)} 帧，{aligned.shape[1]} 列",
                                f"Alignment complete: {len(aligned)} frames, {aligned.shape[1]} columns",
                            )
                        )
                    except Exception as e:
                        st.error(tr(f"对齐失败：{e}", f"Alignment failed: {e}"))

            st.divider()

            # 打板对齐
            st.markdown(f"**{tr('冲击峰值校正', 'Impact peak calibration')}**")
            impact_devs = [
                d.name for d in schema.devices
                if d.device_type != "image" and d.name in st.session_state.device_data
            ]
            if impact_devs:
                impact_device = st.selectbox(
                    tr("检测设备", "Detection device"), impact_devs,
                    help=tr("选择包含物理冲击峰值的设备", "Select a device containing a physical impact peak"),
                    format_func=display_device_name,
                )
                impact_cfg = schema.get_device(impact_device)
                impact_cols = impact_cfg.value_columns if impact_cfg else []
                if impact_cols:
                    impact_col = st.selectbox(tr("冲击信号列", "Impact signal"), impact_cols)
                    if st.button(
                        tr("校正时间偏移", "Calibrate offsets"),
                        icon=":material/adjust:", use_container_width=True,
                    ):
                        with st.spinner(tr("正在检测冲击峰值…", "Detecting impact peaks…")):
                            offsets = st.session_state.engine.alignment.clapperboard_align(
                                st.session_state.device_data,
                                impact_device, impact_col,
                                apply=True,
                            )
                        st.session_state.clapperboard_offsets = offsets
                        if offsets:
                            st.session_state.engine.load_all(
                                st.session_state.device_data
                            )
                            aligned = st.session_state.engine.align(tol)
                            st.session_state.aligned_df = aligned
                            st.session_state.playhead_time = aligned["timestamp"].iloc[0]
                            st.session_state.alignment_frame_index = 0
                            st.session_state.selection = None
                            st.session_state.change_points = []
                            st.json(offsets)
                            st.success(tr("时间偏移已应用并重新对齐", "Offsets applied and data realigned"))
                        else:
                            st.warning(tr("未检测到可用冲击峰值", "No usable impact peak detected"))

            st.divider()

            # 变点检测
            st.markdown(f"**{tr('自动分割', 'Auto-segmentation')}**")
            cpd_cfg = schema.change_point_detection
            cpd_cfg["algorithm"] = st.selectbox(
                tr("变点算法", "Change-point algorithm"), ["pelt", "binseg", "window", "bottomup"],
                index=["pelt", "binseg", "window", "bottomup"].index(
                    cpd_cfg.get("algorithm", "pelt")
                ),
            )
            cpd_cfg["model"] = st.selectbox(
                tr("代价模型", "Cost model"), ["rbf", "l1", "l2", "normal", "ar"],
                index=["rbf", "l1", "l2", "normal", "ar"].index(
                    cpd_cfg.get("model", "rbf")
                ),
            )
            if cpd_cfg["algorithm"] == "pelt":
                pen = st.number_input(
                    tr("惩罚项", "Penalty"),
                    value=float(st.session_state.ui_settings["cpd_penalty"]),
                    step=1.0,
                    help=tr("值越大，检测到的变点越少", "Higher values produce fewer change points"),
                    key="cpd_penalty_control",
                )
                n_bkps_input = None
            else:
                n_bkps_input = st.number_input(
                    tr("变点数", "Number of change points"), value=5, min_value=1, step=1,
                )
                pen = None

        with info_col:
            if st.session_state.aligned_df is None:
                st.info(tr("运行对齐后将在此显示结果", "Run alignment to inspect the result here"), icon=":material/info:")
            else:
                df = st.session_state.aligned_df
                st.subheader(tr("对齐结果", "Aligned result"))
                st.dataframe(df.head(20), use_container_width=True)

                # 数据质量概览
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                data_columns = [column for column in df.columns if column != "timestamp"]
                timestamp_values = (
                    pd.to_datetime(df["timestamp"], errors="coerce")
                    if "timestamp" in df.columns
                    else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
                )
                duration_seconds = (
                    max(0.0, float((timestamp_values.max() - timestamp_values.min()).total_seconds()))
                    if len(timestamp_values) and timestamp_values.notna().any() else 0.0
                )
                missing_cells = int(df[data_columns].isna().sum().sum()) if data_columns else 0
                duplicate_timestamps = int(timestamp_values.duplicated().sum())
                quality_cols = st.columns(4)
                quality_cols[0].metric(tr("数据时长", "Duration"), f"{duration_seconds:.2f} s")
                quality_cols[1].metric(tr("信号列", "Signal columns"), len(data_columns))
                quality_cols[2].metric(tr("缺失值", "Missing values"), missing_cells)
                quality_cols[3].metric(tr("重复时间戳", "Duplicate timestamps"), duplicate_timestamps)

                with st.expander(tr("数据质量明细", "Data quality details"), icon=":material/fact_check:"):
                    quality_rows = []
                    for column in data_columns:
                        missing_count = int(df[column].isna().sum())
                        quality_rows.append({
                            tr("列", "Column"): display_signal_name(column),
                            tr("类型", "Type"): str(df[column].dtype),
                            tr("缺失数", "Missing"): missing_count,
                            tr("缺失率", "Missing rate"): (
                                f"{missing_count / len(df):.1%}" if len(df) else "0.0%"
                            ),
                            tr("唯一值", "Unique values"): int(df[column].nunique(dropna=True)),
                        })
                    st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)

                if numeric_cols:
                    with st.expander(tr("数值列统计", "Numeric summary"), icon=":material/query_stats:"):
                        st.dataframe(df[numeric_cols].describe(), use_container_width=True)

                # 变点检测执行
                st.divider()
                st.subheader(tr("变点检测", "Change-point detection"))
                detect_col = None
                if numeric_cols:
                    detect_col = st.selectbox(
                        tr("检测列", "Signal column"), numeric_cols,
                        help=tr("分析该信号以寻找区间边界", "Analyze this signal to find segment boundaries"),
                    )
                else:
                    st.info(tr("没有可用于变点检测的数值列", "No numeric column is available for change-point detection"))
                if detect_col and st.button(
                    tr("检测变点", "Detect change points"), type="primary",
                    icon=":material/search:",
                ):
                    with st.spinner(tr(
                        f"正在使用 {cpd_cfg['algorithm']} 检测…",
                        f"Detecting with {cpd_cfg['algorithm']}…",
                    )):
                        try:
                            st.session_state.engine.cpd = ChangePointDetector(schema)
                            bkps = st.session_state.engine.detect_change_points(
                                detect_col, n_bkps=n_bkps_input, pen=pen,
                            )
                            st.session_state.change_points = bkps
                            # 自动分割为区间
                            intervals = st.session_state.engine.cpd.segments_to_intervals(
                                df, bkps
                            )
                            st.success(tr(
                                f"检测到 {len(bkps)} 个变点，得到 {len(intervals)} 个区间",
                                f"Detected {len(bkps)} change points and {len(intervals)} segments",
                            ))
                            with st.expander(tr("变点与区间", "Change points and segments")):
                                st.write(tr("变点索引：", "Change-point indices:"), bkps)
                                st.json([
                                    {"start": str(iv["start_time"]),
                                     "end": str(iv["end_time"])}
                                    for iv in intervals
                                ])
                        except Exception as e:
                            st.error(tr(f"变点检测失败：{e}", f"Detection failed: {e}"))

        if st.session_state.aligned_df is not None and not st.session_state.aligned_df.empty:
            aligned_df = st.session_state.aligned_df
            frame_count = len(aligned_df)
            st.divider()
            st.subheader(tr("逐帧对齐检查", "Frame alignment inspector"))

            if st.session_state.alignment_frame_index >= frame_count:
                st.session_state.alignment_frame_index = frame_count - 1

            frame_controls = st.columns([1, 8, 1], vertical_alignment="bottom")
            previous_frame = frame_controls[0].button(
                tr("上一帧", "Previous"),
                icon=":material/skip_previous:",
                help=tr("上一帧", "Previous frame"),
                use_container_width=True,
                disabled=st.session_state.alignment_frame_index <= 0,
                key="alignment_previous_frame",
            )
            next_frame = frame_controls[2].button(
                tr("下一帧", "Next"),
                icon=":material/skip_next:",
                help=tr("下一帧", "Next frame"),
                use_container_width=True,
                disabled=st.session_state.alignment_frame_index >= frame_count - 1,
                key="alignment_next_frame",
            )
            if previous_frame:
                st.session_state.alignment_frame_index -= 1
            if next_frame:
                st.session_state.alignment_frame_index += 1
            with frame_controls[1]:
                frame_index = st.slider(
                    tr("帧位置", "Frame position"),
                    min_value=0,
                    max_value=frame_count - 1,
                    step=1,
                    key="alignment_frame_index",
                )

            frame_time = pd.Timestamp(aligned_df.iloc[frame_index]["timestamp"])
            st.markdown(
                f"**{tr('当前帧', 'Current frame')} #{frame_index}**  ·  "
                f"{frame_time.strftime('%H:%M:%S.%f')[:-3]}"
            )

            image_device = next(
                (
                    device for device in st.session_state.schema.devices
                    if device.device_type in ("image", "mixed")
                    and aligned_frame_filename(aligned_df, frame_index, device) is not None
                ),
                None,
            )
            media_col, values_col = st.columns([3, 2])
            with media_col:
                with st.container(key="alignment_current_frame"):
                    if image_device:
                        rgb_path = aligned_frame_image_path(
                            aligned_df, frame_index, image_device
                        )
                        depth_path = aligned_frame_image_path(
                            aligned_df, frame_index, image_device, depth=True
                        )
                        visual_columns = st.columns(2) if rgb_path and depth_path else [st.container()]
                        if rgb_path:
                            visual_columns[0].image(
                                rgb_path,
                                caption=f"RGB · #{frame_index}",
                                use_container_width=True,
                            )
                        if depth_path:
                            colored_depth = ImagePreviewRenderer.depth_to_colormap(depth_path)
                            if colored_depth is not None:
                                from PIL import Image
                                visual_columns[-1].image(
                                    Image.fromarray(colored_depth),
                                    caption=f"Depth · #{frame_index}",
                                    use_container_width=True,
                                )
                    else:
                        st.info(tr(
                            "当前数据没有可用图像帧",
                            "No image frame is available for the current data",
                        ))

            with values_col:
                st.markdown(f"**{tr('同帧传感器值', 'Synchronized sensor values')}**")
                frame_row = aligned_df.iloc[frame_index]
                precision = int(st.session_state.ui_settings["number_precision"])
                value_rows = []
                for column in aligned_df.select_dtypes(include=[np.number]).columns:
                    value = frame_row[column]
                    value_rows.append({
                        tr("信号", "Signal"): display_signal_name(column),
                        tr("数值", "Value"): (
                            f"{float(value):.{precision}f}" if pd.notna(value) else "N/A"
                        ),
                    })
                st.dataframe(
                    pd.DataFrame(value_rows),
                    use_container_width=True,
                    hide_index=True,
                    height=min(420, 38 + max(1, len(value_rows)) * 35),
                )

            if image_device:
                strip_size = min(7, frame_count)
                strip_start = max(
                    0,
                    min(frame_index - strip_size // 2, frame_count - strip_size),
                )
                strip_indices = list(range(strip_start, strip_start + strip_size))
                st.markdown(f"**{tr('帧带', 'Filmstrip')}**")
                with st.container(key="alignment_filmstrip"):
                    strip_columns = st.columns(strip_size)
                    for column, strip_index in zip(strip_columns, strip_indices):
                        strip_path = aligned_frame_image_path(
                            aligned_df, strip_index, image_device
                        )
                        if strip_path:
                            column.image(strip_path, use_container_width=True)
                        strip_time = pd.Timestamp(
                            aligned_df.iloc[strip_index]["timestamp"]
                        ).strftime("%H:%M:%S.%f")[:-3]
                        frame_label = (
                            tr("当前", "Current")
                            if strip_index == frame_index else f"#{strip_index}"
                        )
                        column.caption(f"{frame_label} · {strip_time}")


# -----------------------------------------------------------------------------
# Tab 3: 区间标注 (核心 - 多轨时间轴)
# -----------------------------------------------------------------------------

with tab_annotate:
    st.header(tr("区间标注", "Interval annotation"))

    if st.session_state.aligned_df is None:
        st.warning(tr("请先完成数据对齐", "Align the data before annotating"), icon=":material/info:")
    else:
        df = st.session_state.aligned_df
        renderer = TimelineRenderer(
            st.session_state.schema,
            language=st.session_state.ui_settings["language"],
            line_width=float(st.session_state.ui_settings["timeline_line_width"]),
            palette=st.session_state.ui_settings["timeline_palette"],
            show_range_slider=bool(st.session_state.ui_settings["show_range_slider"]),
            friendly_names=bool(st.session_state.ui_settings["friendly_device_names"]),
        )

        # ============ 标注工具栏 ============
        with st.container(key="annotation_toolbar"):
            group_names = st.session_state.schema.label_group_names()
            toolbar = st.columns(
                [1.25, 2.35, .9, .9, .8, .42], vertical_alignment="bottom"
            )
            selected_group = toolbar[0].selectbox(
                tr("标签组", "Label group"),
                group_names,
                index=0 if group_names else None,
                placeholder=tr("请先添加标签组", "Add a group first"),
                key="annotation_active_group",
            )
            group_mode = (
                st.session_state.schema.label_group_mode(selected_group)
                if selected_group else "single"
            )
            label_names = (
                st.session_state.schema.label_names(selected_group)
                if selected_group else []
            )
            if group_mode == "multi":
                selected_labels = toolbar[1].multiselect(
                    tr("标签（可多选）", "Labels (multiple)"),
                    label_names,
                    placeholder=tr("选择一个或多个标签", "Select one or more labels"),
                    key=f"annotation_labels_{selected_group}",
                )
            else:
                selected_label = toolbar[1].selectbox(
                    tr("标签", "Label"),
                    label_names,
                    index=0 if label_names else None,
                    placeholder=tr("请先添加标签", "Add a label first"),
                    key=f"annotation_label_{selected_group}",
                )
                selected_labels = [selected_label] if selected_label else []

            apply_label = toolbar[2].button(
                tr("应用", "Apply"), type="primary", icon=":material/label:",
                use_container_width=True,
                disabled=not selected_labels or selected_group is None,
                key="annotation_apply_label",
            )
            next_unlabeled = toolbar[3].button(
                tr("下一空白", "Next gap"), icon=":material/skip_next:",
                use_container_width=True, key="annotation_next_unlabeled",
            )
            undo_annotation = toolbar[4].button(
                tr("撤销", "Undo"), icon=":material/undo:",
                use_container_width=True, key="annotation_undo",
            )
            if toolbar[5].button(
                tr("管理标签", "Manage labels"), icon=":material/edit:",
                help=tr("管理标签组和标签", "Manage groups and labels"),
                use_container_width=True, key="annotation_manage_labels",
            ):
                show_label_manager()

            if apply_label:
                sel = st.session_state.selection
                if sel and sel[0] and sel[1]:
                    iv = st.session_state.engine.annotator.add_interval(
                        sel[0], sel[1], selected_labels, df, group=selected_group,
                    )
                    remember_annotation(iv)
                    st.session_state.intervals = (
                        st.session_state.engine.annotator.intervals
                    )
                    st.session_state.pending_notice = tr(
                        f"已标注 [{iv['label']}]，共 {iv.get('frame_count', '?')} 帧",
                        f"Applied [{iv['label']}] to {iv.get('frame_count', '?')} frames",
                    )
                    if st.session_state.ui_settings["clear_selection_after_label"]:
                        st.session_state.selection = None
                    st.rerun()
                else:
                    st.warning(tr(
                        "请先在时间轴拖选区间，或使用 I/O 设置入点和出点",
                        "Select a timeline range or set in and out points first",
                    ))

            if next_unlabeled:
                next_interval = st.session_state.engine.annotator.next_unlabeled_interval(
                    df, st.session_state.playhead_time, selected_group,
                )
                if next_interval:
                    st.session_state.selection = next_interval
                    st.session_state.playhead_time = next_interval[0]
                    st.rerun()
                else:
                    st.info(tr(
                        "当前标签组没有未标注区间",
                        "The active group has no unlabeled interval",
                    ))

            if undo_annotation:
                removed = undo_last_annotation()
                if removed:
                    st.session_state.selection = (
                        removed["start_time"], removed["end_time"]
                    )
                    st.session_state.playhead_time = removed["start_time"]
                    st.session_state.pending_notice = tr(
                        f"已撤销 [{removed['label']}]",
                        f"Undid [{removed['label']}]",
                    )
                    st.rerun()
                else:
                    st.info(tr("没有可撤销的标注", "No annotation is available to undo"))

            annotation_stats = st.session_state.engine.annotator.annotation_stats(
                df, selected_group
            )
            with st.container(key="annotation_status"):
                status_columns = st.columns(
                    [3.5, 1, 1, 1], vertical_alignment="center"
                )
                status_columns[0].progress(
                    annotation_stats["assigned_ratio"],
                    text=tr(
                        f"{selected_group or '-'} · 覆盖 {annotation_stats['assigned_ratio']:.1%}",
                        f"{selected_group or '-'} · {annotation_stats['assigned_ratio']:.1%} covered",
                    ),
                )
                status_columns[1].metric(
                    tr("已标注", "Labeled"), annotation_stats["labeled_frames"]
                )
                status_columns[2].metric(
                    tr("未标注", "Unlabeled"), annotation_stats["unlabeled_frames"]
                )
                status_columns[3].metric("Ignore", annotation_stats["ignore_frames"])

        # ============ 左上方: 预览区 ============
        st.subheader(tr("多模态预览", "Multimodal preview"))

        preview_col, control_col = st.columns([3, 1])

        with control_col:
            st.markdown(f"**{tr('播放指针', 'Playhead')}**")
            ts = df["timestamp"] if "timestamp" in df.columns else df.iloc[:, 0]
            t_min, t_max = ts.min(), ts.max()

            # 时间滑块 (Scrubber)
            current_time = st.slider(
                tr("时间指针", "Timeline position"),
                min_value=float(t_min.timestamp()),
                max_value=float(t_max.timestamp()),
                value=float(
                    st.session_state.playhead_time.timestamp()
                ) if st.session_state.playhead_time else float(t_min.timestamp()),
                step=0.01,
                format="%.2f",
            )
            playhead_ts = pd.Timestamp(current_time, unit="s")
            st.session_state.playhead_time = playhead_ts

            # 当前时间显示
            st.metric(tr("当前时间", "Current time"), playhead_ts.strftime("%H:%M:%S.%f")[:-3])

            # 找当前帧索引
            ts_series = pd.to_datetime(df["timestamp"])
            frame_idx = int((ts_series - playhead_ts).abs().argmin())
            st.metric(tr("帧索引", "Frame index"), frame_idx)

        with preview_col:
            # 显示 RGB 图像 (若有)
            image_device = None
            for dev in st.session_state.schema.devices:
                if dev.device_type in ("image", "mixed"):
                    image_device = dev
                    break

            img_cols = st.columns([2, 1]) if image_device else [st.container()]

            if image_device:
                filename = None
                with img_cols[0]:
                    img_col_name = f"{image_device.name}.{image_device.image_column}"
                    if img_col_name not in df.columns:
                        img_col_name = image_device.image_column
                    if img_col_name in df.columns:
                        filename = df.iloc[frame_idx][img_col_name]
                        if pd.notna(filename):
                            img_path = os.path.join(image_device.data_path, str(filename))
                            if os.path.exists(img_path):
                                from PIL import Image
                                st.image(Image.open(img_path),
                                         caption=f"RGB: {filename}",
                                         use_container_width=True)
                            else:
                                st.info(tr(f"图像文件不存在：{img_path}", f"Image file not found: {img_path}"))
                        else:
                            st.info(tr("当前帧无图像数据", "No image for the current frame"))

                with img_cols[1]:
                    # Depth 伪彩
                    if image_device.depth_path and filename is not None:
                        depth_name = f"{Path(str(filename)).stem}.png"
                        depth_file = os.path.join(image_device.depth_path, depth_name)
                        if not os.path.exists(depth_file):
                            depth_file = os.path.join(
                                image_device.depth_path, str(filename)
                            )
                        if os.path.exists(depth_file):
                            colored = ImagePreviewRenderer.depth_to_colormap(depth_file)
                            if colored is not None:
                                from PIL import Image
                                st.image(Image.fromarray(colored),
                                         caption=tr("Depth（伪彩）", "Depth (colorized)"),
                                         use_container_width=True)
            else:
                st.info(tr("未配置图像设备，仅显示传感器数值", "No image device configured; showing sensor values only"))

            # 传感器数值卡片
            st.markdown(f"**{tr('当前帧数值', 'Current frame values')}**")
            current_row = df.iloc[frame_idx]
            cards = []
            for col in df.columns:
                if col in ("timestamp", "label"):
                    continue
                if pd.api.types.is_numeric_dtype(df[col]):
                    val = current_row[col]
                    precision = int(st.session_state.ui_settings["number_precision"])
                    cards.append((
                        display_signal_name(col),
                        f"{val:.{precision}f}" if pd.notna(val) else "N/A",
                    ))
            # 分行显示卡片
            card_limit = int(st.session_state.ui_settings["preview_card_limit"])
            visible_cards = cards[:card_limit]
            card_cols = st.columns(min(4, max(1, len(visible_cards))))
            for i, (name, val) in enumerate(visible_cards):
                with card_cols[i % len(card_cols)]:
                    st.markdown(
                        f'<div class="sensor-card"><div class="val">{val}</div>'
                        f'<div class="name">{name}</div></div>',
                        unsafe_allow_html=True,
                    )

        st.divider()

        # ============ 下半部分: 多轨时间轴 ============
        st.subheader(tr("多轨时间轴", "Multi-track timeline"))

        # 控制按钮行
        btn_cols = st.columns([1, 1, 1, 1, 2])
        with btn_cols[0]:
            if st.button(
                tr("设为入点 (I)", "Set in point (I)"),
                key="mark_in", icon=":material/first_page:",
            ):
                if st.session_state.playhead_time:
                    sel = st.session_state.selection
                    if sel:
                        st.session_state.selection = (
                            st.session_state.playhead_time, sel[1]
                        )
                    else:
                        st.session_state.selection = (
                            st.session_state.playhead_time, None
                        )
                    st.success(tr(
                        f"入点：{st.session_state.playhead_time}",
                        f"In point: {st.session_state.playhead_time}",
                    ))
        with btn_cols[1]:
            if st.button(
                tr("设为出点 (O)", "Set out point (O)"),
                key="mark_out", icon=":material/last_page:",
            ):
                if st.session_state.playhead_time:
                    sel = st.session_state.selection
                    if sel:
                        st.session_state.selection = (
                            sel[0], st.session_state.playhead_time
                        )
                    else:
                        st.session_state.selection = (
                            None, st.session_state.playhead_time
                        )
                    st.success(tr(
                        f"出点：{st.session_state.playhead_time}",
                        f"Out point: {st.session_state.playhead_time}",
                    ))
        with btn_cols[2]:
            if st.button(tr("清除选区", "Clear selection"), icon=":material/close:"):
                st.session_state.selection = None
        with btn_cols[3]:
            if st.button(tr("隐藏变点", "Hide change points"), icon=":material/hide_source:"):
                st.session_state.change_points = []

        enable_annotation_shortcuts()

        # 渲染 Plotly 多轨图
        fig = renderer.render(
            df,
            playhead_time=st.session_state.playhead_time,
            intervals=st.session_state.intervals,
            change_points=st.session_state.change_points,
            selection=st.session_state.selection,
            height_per_track=int(st.session_state.ui_settings["timeline_height"]),
        )

        # 使用 on_select 捕获区间选择 (Streamlit 1.35+) 或退化为点击
        try:
            event = st.plotly_chart(
                fig, use_container_width=True,
                on_select="rerun", key="timeline_chart",
            )
        except TypeError:
            # 旧版本 Streamlit 不支持 on_select
            event = None
            st.plotly_chart(fig, use_container_width=True)

        # 处理选择事件
        if event:
            selection = TimelineRenderer.extract_selection(event)
            if selection:
                st.session_state.selection = selection
                st.info(
                    tr(
                        f"已选择：{selection[0]} → {selection[1]}",
                        f"Selected: {selection[0]} → {selection[1]}",
                    )
                )
            else:
                click_time = TimelineRenderer.extract_click_time(event)
                if click_time is not None:
                    st.session_state.playhead_time = click_time

        # 显示当前选择状态
        if st.session_state.selection:
            sel = st.session_state.selection
            if sel[0] and sel[1]:
                st.success(
                    tr("选区：", "Selection: ")
                    + f"{sel[0].strftime('%H:%M:%S.%f')[:-3]} → "
                    + f"{sel[1].strftime('%H:%M:%S.%f')[:-3]}"
                )

        if annotation_stats["label_counts"]:
            with st.expander(tr("标签分布", "Label distribution"), icon=":material/bar_chart:"):
                distribution = pd.DataFrame([
                    {
                        tr("标签", "Label"): label,
                        tr("帧数", "Frames"): count,
                        tr("占比", "Share"): f"{count / annotation_stats['total_frames']:.1%}",
                    }
                    for label, count in annotation_stats["label_counts"].items()
                ])
                st.dataframe(distribution, use_container_width=True, hide_index=True)

        support_columns = st.columns(2)
        with support_columns[0]:
            with st.expander(
                tr("质量检查", "Quality checks"), icon=":material/fact_check:"
            ):
                minimum_frames = st.number_input(
                    tr("过短区间阈值（帧）", "Short interval threshold (frames)"),
                    min_value=1, max_value=max(1, len(df)), value=min(3, max(1, len(df))),
                    step=1, key="quality_minimum_frames",
                )
                quality_issues = st.session_state.engine.annotator.quality_issues(
                    df, int(minimum_frames)
                )
                warning_count = sum(
                    issue["severity"] == "warning" for issue in quality_issues
                )
                quality_metrics = st.columns(3)
                quality_metrics[0].metric(
                    tr("待检查", "To review"), len(quality_issues)
                )
                quality_metrics[1].metric(
                    tr("警告", "Warnings"), warning_count
                )
                quality_metrics[2].metric(
                    tr("提示", "Info"), len(quality_issues) - warning_count
                )
                quality_filter_options = [tr("全部组", "All groups"), *group_names]
                quality_group = st.selectbox(
                    tr("筛选标签组", "Filter by group"),
                    quality_filter_options,
                    key="quality_group_filter",
                )
                filtered_issues = [
                    issue for issue in quality_issues
                    if quality_group == quality_filter_options[0]
                    or issue.get("group") == quality_group
                ]
                if not filtered_issues:
                    st.success(tr(
                        "当前范围未发现需要复核的问题",
                        "No review issues found in the current scope",
                    ))
                for issue_index, issue in enumerate(filtered_issues[:30]):
                    issue_code = issue["code"]
                    if issue_code == "unlabeled_run":
                        issue_text = tr("未标注片段", "Unlabeled run")
                    elif issue_code == "short_interval":
                        labels_text = " + ".join(issue.get("labels", []))
                        issue_text = tr(
                            f"过短区间 · {labels_text}",
                            f"Short interval · {labels_text}",
                        )
                    else:
                        labels_text = " / ".join(issue.get("labels", []))
                        issue_text = tr(
                            f"单选组重叠 · {labels_text}",
                            f"Single-select overlap · {labels_text}",
                        )
                    issue_row = st.columns([4, 1], vertical_alignment="center")
                    issue_row[0].markdown(
                        f"**{issue.get('group', '-')} · {issue_text}**  "
                        f"\n{issue['start_time'].strftime('%H:%M:%S.%f')[:-3]} → "
                        f"{issue['end_time'].strftime('%H:%M:%S.%f')[:-3]} · "
                        f"{issue.get('frame_count', 0)} {tr('帧', 'frames')}"
                    )
                    if issue_row[1].button(
                        tr("跳转", "Go"), icon=":material/play_arrow:",
                        key=f"quality_jump_{issue_index}_{issue_code}_{issue.get('group')}",
                        use_container_width=True,
                    ):
                        st.session_state.selection = (
                            issue["start_time"], issue["end_time"]
                        )
                        st.session_state.playhead_time = issue["start_time"]
                        st.rerun()
                if len(filtered_issues) > 30:
                    st.caption(tr(
                        f"仅显示前 30 项，当前筛选共 {len(filtered_issues)} 项",
                        f"Showing the first 30 of {len(filtered_issues)} issues",
                    ))

        with support_columns[1]:
            with st.expander(
                tr("草稿与恢复", "Draft and recovery"), icon=":material/save:"
            ):
                draft_payload = {
                    "format": "multimodal-annotation-draft",
                    "version": 1,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "schema_signature": _schema_signature(
                        st.session_state.schema, include_llm=False
                    ),
                    "dataset": {
                        "frames": len(df),
                        "start_time": str(df["timestamp"].iloc[0]),
                        "end_time": str(df["timestamp"].iloc[-1]),
                    },
                    "intervals": st.session_state.engine.annotator.to_list(),
                }
                st.download_button(
                    tr("下载标注草稿", "Download annotation draft"),
                    data=json.dumps(
                        draft_payload, ensure_ascii=False, indent=2
                    ).encode("utf-8"),
                    file_name=f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    icon=":material/download:",
                    use_container_width=True,
                )
                draft_file = st.file_uploader(
                    tr("恢复 JSON 草稿", "Restore JSON draft"),
                    type=["json"], key="annotation_draft_file",
                )
                restore_mode = st.radio(
                    tr("恢复方式", "Restore mode"),
                    [tr("替换当前标注", "Replace current"), tr("合并到当前标注", "Merge")],
                    horizontal=True, key="annotation_restore_mode",
                )
                if st.button(
                    tr("恢复草稿", "Restore draft"), icon=":material/upload:",
                    disabled=draft_file is None, use_container_width=True,
                    key="restore_annotation_draft",
                ):
                    try:
                        payload = json.loads(
                            draft_file.getvalue().decode("utf-8-sig")
                        )
                        draft_intervals = (
                            payload.get("intervals") if isinstance(payload, dict)
                            else payload
                        )
                        restored_count = st.session_state.engine.annotator.load_list(
                            draft_intervals,
                            df,
                            replace=restore_mode == tr(
                                "替换当前标注", "Replace current"
                            ),
                        )
                        st.session_state.intervals = (
                            st.session_state.engine.annotator.intervals
                        )
                        st.session_state.annotation_undo_stack = []
                        st.session_state.pending_notice = tr(
                            f"已恢复 {restored_count} 个标注区间",
                            f"Restored {restored_count} annotation intervals",
                        )
                        st.rerun()
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                        st.error(tr(
                            f"草稿恢复失败：{exc}", f"Draft restore failed: {exc}"
                        ))

                st.divider()
                confirm_clear = st.checkbox(
                    tr("我确认清空当前全部标注", "I confirm clearing all annotations"),
                    key="confirm_clear_annotations",
                )
                if st.button(
                    tr("清空全部标注", "Clear all annotations"),
                    icon=":material/delete_sweep:", use_container_width=True,
                    disabled=not confirm_clear, key="clear_all_annotations",
                ):
                    st.session_state.engine.annotator.clear()
                    st.session_state.intervals = []
                    st.session_state.annotation_undo_stack = []
                    st.session_state.pending_notice = tr(
                        "已清空所有标注", "All annotations cleared"
                    )
                    st.rerun()

        # LLM 辅助预判
        st.divider()
        st.subheader(tr("模型建议", "Model suggestion"))

        llm_cols = st.columns([1, 1, 2])
        with llm_cols[0]:
            if st.button(
                tr("分析选区", "Analyze selection"),
                icon=":material/auto_awesome:", use_container_width=True,
                disabled=not label_names,
            ):
                sel = st.session_state.selection
                if sel and sel[0] and sel[1]:
                    features = st.session_state.engine.llm.extract_features(
                        df, sel[0], sel[1]
                    )
                    # 尝试获取选区中间帧的图像
                    img_path = None
                    if image_device:
                        mid_time = sel[0] + (sel[1] - sel[0]) / 2
                        img_col_name = f"{image_device.name}.{image_device.image_column}"
                        if img_col_name not in df.columns:
                            img_col_name = image_device.image_column
                        if img_col_name in df.columns:
                            mid_idx = int(
                                (pd.to_datetime(df["timestamp"]) - mid_time)
                                .abs().argmin()
                            )
                            fname = df.iloc[mid_idx][img_col_name]
                            if pd.notna(fname):
                                p = os.path.join(image_device.data_path, str(fname))
                                if os.path.exists(p):
                                    img_path = p

                    with st.spinner(tr("模型正在分析…", "Model is analyzing…")):
                        result = st.session_state.engine.llm.predict_label(
                            features, img_path,
                            candidate_labels=label_names,
                        )
                    result["group"] = selected_group
                    st.session_state.llm_result = result
                else:
                    st.warning(tr("请先选择区间", "Select an interval first"))

        with llm_cols[1]:
            if st.button(
                tr("采纳建议", "Accept suggestion"),
                icon=":material/check:", use_container_width=True,
            ):
                result = st.session_state.get("llm_result")
                sel = st.session_state.selection
                result_is_current = bool(
                    result
                    and result.get("group") == selected_group
                    and result.get("label") in label_names
                )
                if result_is_current and sel and sel[0] and sel[1]:
                    iv = st.session_state.engine.annotator.add_interval(
                        sel[0], sel[1], result["label"], df,
                        group=selected_group,
                    )
                    remember_annotation(iv)
                    st.session_state.intervals = (
                        st.session_state.engine.annotator.intervals
                    )
                    st.session_state.pending_notice = tr(
                        f"已采纳标签：{result['label']}",
                        f"Accepted label: {result['label']}",
                    )
                    if st.session_state.ui_settings["clear_selection_after_label"]:
                        st.session_state.selection = None
                    st.rerun()
                else:
                    st.warning(tr("没有可采纳的建议", "No suggestion is available"))

        with llm_cols[2]:
            result = st.session_state.get("llm_result")
            if result:
                result_group = result.get("group", selected_group)
                st.markdown(
                    f"**{tr('标签组', 'Group')}:** `{result_group}` · "
                    f"**{tr('建议标签', 'Suggested label')}:** `{result.get('label', 'N/A')}` · "
                    f"**{tr('置信度', 'Confidence')}:** {result.get('confidence', 0):.0%}"
                )
                st.text(tr("说明：", "Reasoning: ") + result.get("reasoning", ""))

        # 已标注区间列表
        st.divider()
        st.subheader(tr("已标注区间", "Annotated intervals"))

        if st.session_state.intervals:
            for i, iv in enumerate(st.session_state.intervals):
                with st.container():
                    cols = st.columns([3, 2, 1, 1])
                    cols[0].markdown(
                        f'<div class="interval-item" '
                        f'style="border-left-color: {iv.get("color", "#95A5A6")}">'
                        f'#{i} {iv["start_time"].strftime("%H:%M:%S.%f")[:-3]} → '
                        f'{iv["end_time"].strftime("%H:%M:%S.%f")[:-3]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    cols[1].markdown(
                        f'`{iv.get("group", st.session_state.schema.default_label_group())}` · '
                        f'{iv["label"]} · {iv.get("frame_count", "?")} '
                        f'{tr("帧", "frames")}'
                    )
                    if cols[2].button(tr("跳转", "Go"), key=f"jump_{i}", icon=":material/play_arrow:"):
                        st.session_state.playhead_time = iv["start_time"]
                        st.rerun()
                    if cols[3].button(tr("删除", "Delete"), key=f"rm_{i}", icon=":material/delete:"):
                        st.session_state.engine.annotator.remove_interval(i)
                        st.session_state.intervals = (
                            st.session_state.engine.annotator.intervals
                        )
                        st.rerun()
        else:
            st.info(tr("暂无标注区间", "No annotated intervals yet"))


# -----------------------------------------------------------------------------
# Tab 4: 导出
# -----------------------------------------------------------------------------

with tab_export:
    st.header(tr("导出数据集", "Export dataset"))

    if st.session_state.aligned_df is None:
        st.warning(tr("请先完成数据对齐", "Align the data before exporting"), icon=":material/info:")
    else:
        df = st.session_state.aligned_df
        intervals = st.session_state.intervals

        st.subheader(tr("概览", "Overview"))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(tr("对齐帧数", "Aligned frames"), len(df))
        m2.metric(tr("数据列数", "Data columns"), df.shape[1])
        m3.metric(tr("标注区间", "Intervals"), len(intervals))
        group_stats = [
            (group_name, st.session_state.engine.annotator.annotation_stats(df, group_name))
            for group_name in st.session_state.schema.label_group_names()
        ]
        average_coverage = (
            sum(stats["assigned_ratio"] for _, stats in group_stats) / len(group_stats)
            if group_stats else 0.0
        )
        m4.metric(
            tr("平均组覆盖", "Average group coverage"),
            f"{average_coverage:.1%}",
        )
        if group_stats:
            coverage_rows = [
                {
                    tr("标签组", "Group"): group_name,
                    tr("模式", "Mode"): tr("多选", "Multiple")
                    if st.session_state.schema.label_group_mode(group_name) == "multi"
                    else tr("单选", "Single"),
                    tr("已标注帧", "Labeled frames"): stats["labeled_frames"],
                    tr("未标注帧", "Unlabeled frames"): stats["unlabeled_frames"],
                    tr("覆盖率", "Coverage"): f"{stats['assigned_ratio']:.1%}",
                }
                for group_name, stats in group_stats
            ]
            st.dataframe(coverage_rows, use_container_width=True, hide_index=True)
        incomplete_groups = [
            f"{group_name} ({stats['unlabeled_frames']})"
            for group_name, stats in group_stats if stats["unlabeled_frames"]
        ]
        if incomplete_groups:
            st.warning(tr(
                "以下标签组仍有未标注帧：" + "、".join(incomplete_groups)
                + "。导出时将保留空标签。",
                "Unlabeled frames remain in: " + ", ".join(incomplete_groups)
                + ". Empty labels will be preserved in the export.",
            ))

        st.divider()

        # 预览带标签的数据
        st.subheader(tr("数据预览", "Data preview"))
        labeled_df = st.session_state.engine.annotator.apply_to_dataframe(df)
        st.dataframe(labeled_df.head(30), use_container_width=True)

        st.divider()

        # 导出选项
        st.subheader(tr("格式", "Formats"))
        export_cols = st.columns(3)

        with export_cols[0]:
            st.markdown("**HDF5**")
            st.caption(tr("包含对齐数据、区间与 Schema", "Aligned data, intervals, and schema"))
            h5_name = st.text_input(tr("文件名", "Filename"), "dataset.h5", key="h5_name")
            if st.button(
                tr("生成 HDF5", "Generate HDF5"), type="primary",
                icon=":material/archive:", use_container_width=True,
            ):
                try:
                    out_path, h5_name = export_path(h5_name, ".h5")
                    with st.spinner("导出中..."):
                        st.session_state.engine.export_hdf5(out_path)
                    st.success(tr(f"已导出：{out_path}", f"Exported: {out_path}"))
                    with open(out_path, "rb") as f:
                        st.download_button(
                            tr("下载 HDF5", "Download HDF5"), f.read(),
                            file_name=h5_name,
                            mime="application/octet-stream",
                        )
                except Exception as e:
                    st.error(tr(f"导出失败：{e}", f"Export failed: {e}"))

        with export_cols[1]:
            st.markdown("**Master CSV**")
            st.caption(tr("主表与区间 JSON", "Master table with interval JSON"))
            csv_name = st.text_input(tr("文件名", "Filename"), "master.csv", key="csv_name")
            if st.button(
                tr("生成 CSV", "Generate CSV"), type="primary",
                icon=":material/table_view:", use_container_width=True,
            ):
                try:
                    out_path, csv_name = export_path(csv_name, ".csv")
                    with st.spinner("导出中..."):
                        st.session_state.engine.export_csv(out_path)
                    st.success(tr(f"已导出：{out_path}", f"Exported: {out_path}"))
                    with open(out_path, "rb") as f:
                        st.download_button(
                            tr("下载 CSV", "Download CSV"), f.read(),
                            file_name=csv_name,
                            mime="text/csv",
                        )
                except Exception as e:
                    st.error(tr(f"导出失败：{e}", f"Export failed: {e}"))

        with export_cols[2]:
            st.markdown("**Schema YAML**")
            st.caption(tr("当前传感器配置", "Current sensor configuration"))
            yaml_name = st.text_input(tr("文件名", "Filename"), "schema.yaml", key="yaml_name")
            if st.button(
                tr("生成 Schema", "Generate schema"), type="primary",
                icon=":material/code:", use_container_width=True,
            ):
                try:
                    out_path, yaml_name = export_path(yaml_name, ".yaml")
                    st.session_state.schema.to_yaml(out_path)
                    st.success(tr(f"已导出：{out_path}", f"Exported: {out_path}"))
                    with open(out_path, "rb") as f:
                        st.download_button(
                            tr("下载 YAML", "Download YAML"), f.read(),
                            file_name=yaml_name,
                            mime="application/x-yaml",
                        )
                except Exception as e:
                    st.error(tr(f"导出失败：{e}", f"Export failed: {e}"))

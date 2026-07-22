"""
generate_sample_data.py
============================
示例数据生成脚本

生成与 config/sensors_schema.yaml 路径匹配的示例多模态数据:
- RGB-D 相机: 时间戳 CSV (30Hz) + 伪 RGB 图像 + 深度图
- FSR 足底压力: CSV (100Hz)
- IMU 惯性测量单元: CSV (200Hz)
- 超声波测距: CSV (50Hz)

模拟可穿戴机器人地形感知场景, 包含 平地/坡道/楼梯 三种地形切换。
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from PIL import Image


# 输出目录
BASE_DIR = "data"
RGB_DIR = os.path.join(BASE_DIR, "rgb")
DEPTH_DIR = os.path.join(BASE_DIR, "depth")


def _gen_timestamps(duration: float, freq: float, start: float = 0.0):
    """生成时间戳数组 (秒)。"""
    n = int(duration * freq)
    return np.linspace(start, start + duration, n, endpoint=False)


def generate_rgb_camera(duration: float = 10.0, freq: float = 30.0):
    """生成 RGB-D 相机时间戳表 + 伪图像。"""
    os.makedirs(RGB_DIR, exist_ok=True)
    os.makedirs(DEPTH_DIR, exist_ok=True)

    timestamps = _gen_timestamps(duration, freq)
    filenames = []
    print(f"生成 RGB 图像: {len(timestamps)} 帧 @ {freq}Hz")

    for i, t in enumerate(timestamps):
        # 根据 t 判断地形 (用于生成不同颜色的伪图像)
        if t < duration / 3:
            terrain = "flat"        # 绿色 平地
            base_color = (46, 204, 113)
        elif t < 2 * duration / 3:
            terrain = "slope"       # 橙色 坡道
            base_color = (243, 156, 18)
        else:
            terrain = "stairs"      # 红色 楼梯
            base_color = (231, 76, 60)

        # 生成 320x240 带噪点的伪图像
        img = np.full((240, 320, 3), base_color, dtype=np.uint8)
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # 添加帧编号文字区域
        img[10:40, 10:200] = (0, 0, 0)
        fname = f"frame_{i:05d}.jpg"
        Image.fromarray(img).save(os.path.join(RGB_DIR, fname), quality=85)
        filenames.append(fname)

        # 深度图 (16-bit PNG, 模拟 0.5-5m)
        depth = np.random.uniform(500, 5000, (240, 320)).astype(np.uint16)
        depth_path = os.path.join(DEPTH_DIR, fname.replace(".jpg", ".png"))
        Image.fromarray(depth).save(depth_path)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "filename": filenames,
    })
    out = os.path.join(BASE_DIR, "rgb_timestamps.csv")
    df.to_csv(out, index=False)
    print(f"[OK] RGB 时间戳表: {out} ({len(df)} 行)")
    return df


def generate_fsr(duration: float = 10.0, freq: float = 100.0):
    """生成足底压力传感器 FSR 数据 (100Hz)。"""
    timestamps = _gen_timestamps(duration, freq)
    n = len(timestamps)
    rng = np.random.default_rng(42)

    # 模拟足底压力: 平地均匀, 坡道左右不对称, 楼梯周期性冲击
    heel_l = np.zeros(n)
    heel_r = np.zeros(n)
    mid_l = np.zeros(n)
    mid_r = np.zeros(n)
    fore_l = np.zeros(n)
    fore_r = np.zeros(n)

    step_period = 0.6  # 步态周期 (秒)
    for i, t in enumerate(timestamps):
        phase = (t % step_period) / step_period  # 0-1
        # 步态包络 (正弦平方模拟足底接触)
        env = np.sin(np.pi * phase) ** 2
        if t < duration / 3:
            # 平地: 左右对称
            heel_l[i] = 300 + 200 * env + rng.normal(0, 15)
            heel_r[i] = 300 + 200 * env + rng.normal(0, 15)
            fore_l[i] = 200 + 150 * env + rng.normal(0, 10)
            fore_r[i] = 200 + 150 * env + rng.normal(0, 10)
        elif t < 2 * duration / 3:
            # 坡道: 前脚掌压力增大
            fore_l[i] = 350 + 200 * env + rng.normal(0, 15)
            fore_r[i] = 350 + 200 * env + rng.normal(0, 15)
            heel_l[i] = 150 + 100 * env + rng.normal(0, 10)
            heel_r[i] = 150 + 100 * env + rng.normal(0, 10)
        else:
            # 楼梯: 冲击更大, 周期变化
            heel_l[i] = 400 + 300 * env + rng.normal(0, 20)
            heel_r[i] = 400 + 300 * env + rng.normal(0, 20)
            fore_l[i] = 100 + 80 * env + rng.normal(0, 10)
            fore_r[i] = 100 + 80 * env + rng.normal(0, 10)
        mid_l[i] = 100 + 50 * env + rng.normal(0, 8)
        mid_r[i] = 100 + 50 * env + rng.normal(0, 8)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "heel_left": heel_l, "heel_right": heel_r,
        "mid_left": mid_l, "mid_right": mid_r,
        "fore_left": fore_l, "fore_right": fore_r,
    })
    out = os.path.join(BASE_DIR, "fsr_insole.csv")
    df.to_csv(out, index=False)
    print(f"[OK] FSR 足底压力: {out} ({len(df)} 行 @ {freq}Hz)")
    return df


def generate_imu(duration: float = 10.0, freq: float = 200.0):
    """生成 IMU 惯性测量单元数据 (200Hz)。"""
    timestamps = _gen_timestamps(duration, freq)
    n = len(timestamps)
    rng = np.random.default_rng(123)

    accel_x = np.zeros(n)
    accel_y = np.zeros(n)
    accel_z = np.zeros(n)
    gyro_x = np.zeros(n)
    gyro_y = np.zeros(n)
    gyro_z = np.zeros(n)

    for i, t in enumerate(timestamps):
        # 重力分量 (z 轴 ~9.8)
        accel_z[i] = 9.8 + rng.normal(0, 0.1)
        if t < duration / 3:
            # 平地: 水平加速度小
            accel_x[i] = rng.normal(0, 0.3)
            accel_y[i] = rng.normal(0, 0.3)
            gyro_z[i] = rng.normal(0, 0.1)
        elif t < 2 * duration / 3:
            # 坡道: 倾角变化, x 轴加速度增大
            tilt = 12 * np.pi / 180  # 12 度
            accel_x[i] = 9.8 * np.sin(tilt) + rng.normal(0, 0.4)
            accel_z[i] = 9.8 * np.cos(tilt) + rng.normal(0, 0.1)
            gyro_y[i] = rng.normal(0.05, 0.1)  # 持续俯仰角速度
        else:
            # 楼梯: 周期性冲击 + 角速度变化
            step_phase = np.sin(2 * np.pi * t / 0.8)
            accel_x[i] = 2 * step_phase + rng.normal(0, 0.5)
            gyro_y[i] = 1.5 * step_phase + rng.normal(0, 0.2)
            gyro_z[i] = 0.3 * step_phase + rng.normal(0, 0.1)
        gyro_x[i] = rng.normal(0, 0.05)
        accel_y[i] = accel_y[i] + rng.normal(0, 0.2)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "accel_x": accel_x, "accel_y": accel_y, "accel_z": accel_z,
        "gyro_x": gyro_x, "gyro_y": gyro_y, "gyro_z": gyro_z,
    })
    out = os.path.join(BASE_DIR, "imu_thigh.csv")
    df.to_csv(out, index=False)
    print(f"[OK] IMU 惯性单元: {out} ({len(df)} 行 @ {freq}Hz)")
    return df


def generate_ultrasonic(duration: float = 10.0, freq: float = 50.0):
    """生成超声波测距数据 (50Hz)。"""
    timestamps = _gen_timestamps(duration, freq)
    n = len(timestamps)
    rng = np.random.default_rng(7)

    distance = np.zeros(n)
    for i, t in enumerate(timestamps):
        if t < duration / 3:
            # 平地: 稳定 ~1.0m
            distance[i] = 1.0 + rng.normal(0, 0.02)
        elif t < 2 * duration / 3:
            # 坡道: 渐变
            distance[i] = 1.0 + 0.5 * (t - duration / 3) / (duration / 3) \
                + rng.normal(0, 0.03)
        else:
            # 楼梯: 周期性变化 (上下台阶)
            distance[i] = 1.2 + 0.3 * np.sin(2 * np.pi * t / 0.8) \
                + rng.normal(0, 0.04)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "distance": distance,
    })
    out = os.path.join(BASE_DIR, "ultrasonic.csv")
    df.to_csv(out, index=False)
    print(f"[OK] 超声波测距: {out} ({len(df)} 行 @ {freq}Hz)")
    return df


def generate_all(duration: float = 10.0):
    """生成所有示例数据。"""
    os.makedirs(BASE_DIR, exist_ok=True)
    print("=" * 50)
    print("开始生成示例多模态数据...")
    print("=" * 50)
    generate_rgb_camera(duration, 30.0)
    generate_fsr(duration, 100.0)
    generate_imu(duration, 200.0)
    generate_ultrasonic(duration, 50.0)
    print("=" * 50)
    print(f"[OK] 全部完成! 数据保存在 {BASE_DIR}/ 目录")
    print(f"  - RGB 图像: {RGB_DIR}/ ({int(duration*30)} 帧)")
    print(f"  - Depth 深度: {DEPTH_DIR}/ ({int(duration*30)} 帧)")
    print(f"  - rgb_timestamps.csv (30Hz)")
    print(f"  - fsr_insole.csv (100Hz)")
    print(f"  - imu_thigh.csv (200Hz)")
    print(f"  - ultrasonic.csv (50Hz)")
    print("=" * 50)


if __name__ == "__main__":
    generate_all()

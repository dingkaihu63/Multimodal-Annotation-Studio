# 多模态数据标注辅助工具

简体中文 | [English](README_EN.md)

面向可穿戴机器人、地形感知和人体运动学数据的离线对齐与区间标注 MVP。应用使用动态 YAML Schema 描述设备，不在代码中绑定传感器名称。

## 运行环境

项目使用现有 Anaconda `python3.10` 环境：

```powershell
conda run -n python3.10 python -m pip install -r requirements.txt
conda run -n python3.10 streamlit run app.py
```

也可以创建独立环境：

```powershell
conda env create -f environment.yml
conda activate multimodal-annotation-studio
streamlit run app.py
```

浏览器默认访问 `http://localhost:8501`。

侧边栏的“设置”按钮可切换简体中文/English，并调整界面密度、时间轴轨道高度、数值精度、默认对齐容差和边界过渡区。

标签管理支持自定义标签组。地形、动作、步态阶段等互斥类别可设为单选组，事件和属性可设为多选组；同一帧可同时保留多个组的结果，多选组的重叠区间会自动合并标签。标签和标签组均可新增、重命名、修改或删除。完整操作流程见 [使用说明.md](使用说明.md)。

标注工作区采用时间轴优先布局：标签和常用操作固定在顶部工具栏；质量检查会定位未标注片段、过短区间和单选冲突；JSON 草稿支持下载、替换恢复或合并恢复，错误草稿不会破坏当前标注。

模型助手通过“配置模型 API”弹窗设置模型名称、API 地址和 API Key。API Key 仅用于当前运行会话，不会写入导出的 YAML 或 HDF5 文件。

## 基本流程

1. 在侧边栏导入或编辑传感器 Schema，选择主设备。
2. 在“数据加载”中使用配置路径或上传 CSV/图像 ZIP；首次体验可点击“生成演示数据”。
3. 执行主轨对齐。需要时选择冲击设备和信号做打板校正，校正后会自动重对齐。
4. 运行变点检测，进入多轨时间轴拖选区间，或用 `I`、`O` 设置区间端点。
5. 选择标签组；单选组选择一个标签，多选组可同时选择多个标签。
6. 应用人工标签，或配置 OpenAI 兼容接口获取当前组的 LLM/VLM 标签建议。
7. 导出 HDF5、Master CSV/区间 JSON 和当前 YAML Schema。

默认配置位于 `config/sensors_schema.yaml`，内置地形、动作、步态阶段和事件四个标签组。旧版 Schema 未声明标签组时会自动兼容为一个“类别”单选组。数值 CSV 至少包含配置的时间戳列和数值列；时间戳支持 Unix 秒、毫秒、微秒、纳秒及 ISO 8601 字符串。图像设备使用时间戳 CSV（时间戳列 + 文件名列）和图像目录。

演示数据由 `generate_sample_data.py` 本地生成，不写入 Git 仓库。可在应用“数据”页点击“生成演示数据”，或运行：

```powershell
conda run -n python3.10 python generate_sample_data.py
```

## 验证

```powershell
conda run -n python3.10 python -m unittest -v
```

测试覆盖 Schema 兼容与校验、时间戳解析、主轨对齐、打板校正、分组多标签、草稿原子恢复、质量检查、边界标签以及 HDF5 单选/多热导出。

## 致谢

本项目建立在多个优秀的开源项目之上。感谢以下项目及其贡献者：

- [Streamlit](https://github.com/streamlit/streamlit) 提供交互式应用框架；
- [Plotly.py](https://github.com/plotly/plotly.py) 提供多轨时间序列可视化；
- [pandas](https://github.com/pandas-dev/pandas) 和 [NumPy](https://github.com/numpy/numpy) 提供数据处理、时间戳对齐与数值计算能力；
- [ruptures](https://github.com/deepcharles/ruptures) 提供时间序列变点检测算法；
- [h5py](https://github.com/h5py/h5py) 提供 HDF5 数据集读写能力；
- [OpenCV](https://github.com/opencv/opencv)、[Pillow](https://github.com/python-pillow/Pillow) 和 [PyYAML](https://github.com/yaml/pyyaml) 分别支持图像处理、图像预览和 Schema 配置；
- [OpenAI Python SDK](https://github.com/openai/openai-python) 提供 OpenAI 兼容模型接口客户端。

项目的工作流和界面设计还参考了 [Label Studio](https://github.com/HumanSignal/label-studio)、[CVAT](https://github.com/cvat-ai/cvat)、[BORIS](https://github.com/olivierfriard/BORIS) 和 [PlotJuggler](https://github.com/PlotJuggler/PlotJuggler)。这些项目在通用数据标注、视觉标注、行为事件记录和时间序列分析方面提供了重要启发。

上述项目的版权归各自作者所有，并分别遵循其自身的开源协议。本项目对它们的致谢不表示这些项目的作者为本项目提供背书。

## 引用

如果本项目对研究或工程工作有帮助，请在论文、报告或项目文档中引用：

```bibtex
@misc{dingkaihu63_2026_multimodal_annotation_studio,
  author       = {{dingkaihu63}},
  title        = {Multimodal Annotation Studio},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/dingkaihu63/Multimodal-Annotation-Studio},
  note         = {MIT licensed software}
}
```

引用具体发布版本时，建议同时注明所使用的 Git 提交哈希或 GitHub Release 版本号，以保证结果可复现。

## 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。

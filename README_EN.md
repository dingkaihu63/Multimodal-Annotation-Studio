# Multimodal Annotation Studio

[简体中文](README.md) | English

An offline alignment and interval annotation workspace for wearable robotics,
terrain perception, gait analysis, and human-motion datasets. Devices and data
channels are described by a dynamic YAML schema instead of being hard-coded in
the application.

## Highlights

- Batch-import datasets by `session_id` from a long-form CSV manifest or a ZIP
  package, then align RGB, depth, and numeric sensors on one shared timeline.
- Store `planned_label` and `verified_label` separately. Effective training
  labels prefer verified annotations and fall back to planned annotations.
- Copy video-level labels to a full Session and record start, confirm, stop,
  and cancel events at the playhead.
- Mark terrain/surface boundaries plus validity and quality-anomaly intervals.
- Generate planned foot-contact and gait-phase annotations from FSR signals.
- Audit edits with annotator, source, confidence, version, and change history.
- Export stable training fields and inspect balance by class, subject, and scene.
- Align RGB-D images and numeric sensors sampled at different frequencies to a
  configurable master timeline.
- Correct device clock offsets using detected physical impact peaks.
- Inspect synchronized RGB, depth, IMU, pressure, and distance data frame by
  frame in a video-editor-style workspace.
- Detect candidate segment boundaries with `ruptures` change-point algorithms.
- Organize labels into single-select and multi-select groups. A frame can keep
  labels from multiple groups, while overlapping multi-select intervals are
  merged automatically.
- Find unlabeled runs, suspiciously short intervals, and conflicting overlaps
  in single-select groups with a navigable quality-check queue.
- Download and restore atomic JSON annotation drafts without risking partial
  replacement from an invalid draft.
- Export aligned datasets to CSV and training-friendly HDF5 files, including
  single-label IDs and multi-hot matrices.
- Optionally request label suggestions from an OpenAI-compatible LLM/VLM API.

## Requirements

- Anaconda or Miniconda
- Python 3.10

Create the provided environment:

```powershell
conda env create -f environment.yml
conda activate multimodal-annotation-studio
streamlit run app.py
```

Alternatively, use an existing Python 3.10 environment:

```powershell
conda run -n python3.10 python -m pip install -r requirements.txt
conda run -n python3.10 streamlit run app.py
```

Open `http://localhost:8501` in a browser.

## Demo Data

Generated demo data is intentionally excluded from Git history. Create a local
RGB, depth, FSR, IMU, and ultrasonic sample dataset from the **Data** tab, or
run:

```powershell
conda run -n python3.10 python generate_sample_data.py
```

The generated files are written to `data/`, match the default schema in
`config/sensors_schema.yaml`, and include `data/session_manifest.csv` for the
Session workflow.

## Typical Workflow

1. Import or edit a sensor schema and select the master device.
2. Download the manifest template from **Sessions**, fill one row per device,
   and import the CSV or a ZIP containing the manifest and referenced files.
3. Enter the annotator ID, select a Session, and load it onto the shared master
   timeline. Apply impact-peak clock correction when required.
4. Inspect individual frames and synchronized sensor values.
5. Select a time range on the multi-track timeline, or set in/out points with
   `I` and `O`.
6. Select a planned or verified layer and apply interval labels. Copy a label
   across the full Session or add operation events at the playhead when needed.
7. Generate gait and missing-data prelabels, then verify planned intervals.
8. Review gaps and edit history. Export the direct training CSV after checking
   class, subject, and scene balance, or export HDF5/Master CSV/Schema.

## Session Manifest

The manifest is a long-form CSV with one `session_id + device` per row. Required
columns are `session_id`, `subject_id`, `scene_id`, `device`, and `data_path`.
Optional columns are `timestamp_file`, `depth_path`, and `planned_label`.
Relative paths are resolved from the manifest directory. Device names must
exist in the active sensor schema.

## Training Export

The training CSV contains session metadata, frame/timestamp fields, RGB/depth
paths, planned/verified/effective fields for terrain, surface, action, gait,
events, validity, and quality, foot-contact flags, terrain/surface boundaries,
four operation-event flags, annotation version/annotator, and raw sensor fields
prefixed with `sensor__`.

## Grouped Labels

The default schema demonstrates seven groups:

- `地形` (terrain), single-select
- `表面` (surface), single-select
- `动作` (action), single-select
- `步态阶段` (gait phase), single-select
- `操作事件` (operation event), multi-select
- `有效性` (validity), single-select
- `质量异常` (quality anomaly), multi-select

CSV exports contain one string column per group, such as `label__地形` and
`label__事件`. Multi-select values use `|` as a separator. HDF5 exports store
single-select groups as `label_id__<group>` and multi-select groups as
`label_multihot__<group>` datasets with label mappings in their attributes.

Schemas created before grouped labels were introduced remain compatible: their
labels are assigned to an automatically created single-select group named
`类别`.

## Model API Configuration

Open **Configure Model API** in the sidebar and enter the model name, an
OpenAI-compatible API URL, and an API key. The key is kept only in the current
runtime session and is excluded from exported YAML and HDF5 files.

## Tests

```powershell
conda run -n python3.10 python -m unittest -v
```

The test suite covers schema compatibility and validation, timestamp parsing,
master-track alignment, Session manifest validation, planned/verified layers,
history, gait/quality prelabels, stable training fields, balance reports, atomic
draft restoration, quality checks, and HDF5 label encoding.

## Documentation

The complete Chinese user guide is available in [使用说明.md](使用说明.md).

## Acknowledgements

This project is built on the work of many open-source communities. We thank the
following projects and their contributors:

- [Streamlit](https://github.com/streamlit/streamlit) for the interactive
  application framework;
- [Plotly.py](https://github.com/plotly/plotly.py) for multi-track time-series
  visualization;
- [pandas](https://github.com/pandas-dev/pandas) and
  [NumPy](https://github.com/numpy/numpy) for data processing, timestamp
  alignment, and numerical computation;
- [ruptures](https://github.com/deepcharles/ruptures) for time-series
  change-point detection;
- [h5py](https://github.com/h5py/h5py) for HDF5 dataset support;
- [OpenCV](https://github.com/opencv/opencv),
  [Pillow](https://github.com/python-pillow/Pillow), and
  [PyYAML](https://github.com/yaml/pyyaml) for image processing, image preview,
  and schema configuration;
- [OpenAI Python SDK](https://github.com/openai/openai-python) for the
  OpenAI-compatible model client.

The workflow and interface design were also informed by
[Label Studio](https://github.com/HumanSignal/label-studio),
[CVAT](https://github.com/cvat-ai/cvat),
[BORIS](https://github.com/olivierfriard/BORIS), and
[PlotJuggler](https://github.com/PlotJuggler/PlotJuggler). Their work on general
data labeling, visual annotation, behavioral event logging, and time-series
analysis provided valuable inspiration.

Copyright in the projects listed above remains with their respective authors,
and each project is distributed under its own license. Acknowledgement does not
imply endorsement of this project by those authors or organizations.

## Citation

If this project supports your research or engineering work, please cite it in
papers, reports, or project documentation:

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

For reproducible work, include the Git commit hash or GitHub Release version
used in your experiments.

## License

This project is released under the [MIT License](LICENSE).

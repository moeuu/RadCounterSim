# RadCounterSim 実装仕様書

## 0. この仕様書の目的

本仕様書は、NVIDIA Isaac Sim 上に、ロボット行動と放射線場変化を連成させた閉ループ型放射線源対策シミュレーター **RadCounterSim** を実装するための Codex 向け設計仕様である。

実装対象は次の全機能を含む。

- 移動ロボットによる環境内移動
- ロボットアームによる遮蔽材、障害物、汚染物体の把持・移動・設置・撤去
- 面線源、点線源、汚染物体に付随する線源の表現
- 除染による線源強度の局所的減少
- 遮蔽による線源―検出器間の伝達率の変化
- 汚染物体の移動・撤去による線源位置・有無の変化
- 非指向性検出器、回転遮蔽体付き検出器、エネルギービン付きカウント計測
- 線量率マップと検出器カウントの高速再計算
- 放射線源位置・強度・不確かさの推定
- 対策前予測、対策実行、対策後再計測、予測―実測残差、再推定、再計画の閉ループ
- 除染残り、遮蔽材ずれ、未発見線源、推定スケール誤差の故障モード生成・識別
- 計測時間、作業時間、対策回数、遮蔽材量、ロボット稼働時間を考慮する行動評価
- GUI、ヘッドレス実験実行、ログ、再現性、単体試験、統合試験、性能ベンチマーク
- ROS 2 / MoveIt 2 / Nav2 連携を追加できる拡張インターフェース

## 1. 最重要設計判断

### 1.1 OceanSim を直接フォークしない

OceanSim は設計参考とし、RadCounterSim は独立した Isaac Sim extension として実装する。理由は次の通り。

- 水中カメラ・ソナー用コードと放射線輸送・対策コードを混在させない。
- Isaac Sim の将来版への追従を容易にする。
- 放射線カーネルを Isaac Sim 非依存で単体試験できるようにする。
- 論文上の貢献を独立したシミュレーション基盤として示せる。

### 1.2 Truth と Belief を完全分離する

シミュレーション内部には次の二つの状態を持たせる。

- **TruthState**: 実際の線源分布、実際の除染効率、実際の遮蔽位置、未発見線源、検出器誤差など。
- **BeliefState**: 推定された線源分布、推定不確かさ、計画時に仮定した対策効果。

Estimator と Planner は TruthState を参照してはならない。利用可能なのは公開環境形状、ロボット状態、計測結果、対策完了通知のみとする。

### 1.3 放射線計算を物理ステップから分離する

放射線場全体を毎 physics tick 再計算しない。計算を行うイベントは以下に限定する。

- 計測要求
- 線量マップ更新要求
- 対策候補評価
- 遮蔽・汚染物体・環境形状の pose 変更完了
- 除染による activity 更新
- 閉ループの各ステップ

### 1.4 二つの実行モードを持つ

- **Deterministic action mode**: ナビゲーション・把持の成功を高レベルで判定し、最終 pose と作用だけを適用する。閉ループアルゴリズム評価と CI に使用。
- **Physics action mode**: PhysX、コントローラ、グリッパ、MoveIt 2 等を使って実際にロボットを動かす。実機接続前の評価に使用。

両モードで同じ `CountermeasureAction` と `ActionResult` を使用する。

## 2. 対象バージョンと開発環境

### 2.1 推奨固定環境

- Ubuntu 24.04
- NVIDIA Isaac Sim 6.0.1
- ROS 2 Jazzy
- Python は Isaac Sim 同梱環境
- C++17 以上
- Intel Embree 4 系
- NumPy、SciPy、Pydantic、PyYAML、pandas、pyarrow
- pytest

`requirements-lock.txt` と `environment_manifest.json` にバージョンを固定すること。Isaac Sim 5.0 を使う必要がある場合は別ブランチを切り、同一コード内で大量の version conditional を書かない。

### 2.2 開発方式

Isaac Sim source workspace で extension template を生成し、以下の三層に分割する。

1. `radcounter.core`: Isaac Sim 非依存の Python モデル、推定、計画、ログ。
2. `radcounter.radiation.native`: Embree を使う C++/pybind11 backend。
3. `radcounter.isaac`: USD、UI、ロボット、物理、ROS 2 との接続。

## 3. リポジトリ構成

```text
RadCounterSim/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements-lock.txt
├── environment_manifest.json
├── docs/
│   ├── architecture.md
│   ├── radiation_model.md
│   ├── usd_metadata.md
│   ├── scenarios.md
│   ├── validation.md
│   └── CHANGELOG.md
├── configs/
│   ├── materials/
│   │   ├── lead.yaml
│   │   ├── steel.yaml
│   │   ├── concrete.yaml
│   │   └── air.yaml
│   ├── isotopes/
│   │   └── example_isotope.yaml
│   ├── detectors/
│   │   ├── omni_counter.yaml
│   │   └── rotating_shield_counter.yaml
│   ├── robots/
│   │   ├── measurement_robot.yaml
│   │   └── countermeasure_robot.yaml
│   └── scenarios/
│       ├── analytic_free_space.yaml
│       ├── shield_demo.yaml
│       ├── decon_demo.yaml
│       ├── movable_source_demo.yaml
│       └── closed_loop_demo.yaml
├── assets/
│   ├── environments/
│   ├── shields/
│   ├── tools/
│   ├── contaminated_objects/
│   └── robots/
├── source/extensions/
│   ├── radcounter.radiation.native/
│   │   ├── config/extension.toml
│   │   ├── premake5.lua
│   │   ├── include/radcounter_radiation/
│   │   ├── src/
│   │   ├── bindings/
│   │   ├── radcounter/radiation/native/__init__.py
│   │   └── tests/
│   └── radcounter.isaac/
│       ├── config/extension.toml
│       ├── data/
│       ├── docs/
│       ├── radcounter/isaac/
│       │   ├── extension.py
│       │   ├── scenario.py
│       │   ├── ui.py
│       │   ├── app_controller.py
│       │   ├── usd/
│       │   ├── robots/
│       │   ├── visualization/
│       │   ├── ros2/
│       │   └── tests/
│       └── premake5.lua
├── radcounter/
│   └── core/
│       ├── models/
│       ├── radiation/
│       ├── sensors/
│       ├── actions/
│       ├── estimation/
│       ├── planning/
│       ├── workflow/
│       ├── experiments/
│       └── logging/
├── ros2_ws/src/
│   ├── radcounter_msgs/
│   └── radcounter_bringup/
├── scripts/
│   ├── run_gui.py
│   ├── run_headless.py
│   ├── validate_scenario.py
│   ├── build_transfer_matrix.py
│   ├── benchmark_radiation.py
│   └── export_run_report.py
├── experiments/
│   ├── baselines/
│   ├── sweeps/
│   └── notebooks/
└── tests/
    ├── unit/
    ├── integration/
    ├── regression/
    └── data/
```

## 4. extension の役割

### 4.1 `radcounter.radiation.native`

責務:

- Embree device、scene、geometry、instance の生成・破棄
- USD から抽出済みの三角形 mesh を受け取る
- source point と detector point の segment ray tracing
- 材料別通過長または energy 別 transmission のバッチ計算
- 動的 object transform の更新
- GIL を解放した並列計算
- Embree が使えないときに明示的なエラーを返す

この extension は UI や USD API を直接呼ばない。

### 4.2 `radcounter.isaac`

責務:

- UI extension と Examples Browser 登録
- scenario load/reset/clear
- USD mesh、transform、custom attribute の読み取り
- robot、sensor、shield、tool、contaminated object の生成
- Physics callback と event subscription
- visualization
- optional ROS 2 bridge

### 4.3 `radcounter.core`

責務:

- TruthState、BeliefState、Action、Measurement の型
- 放射線源、材料、検出器モデル
- 放射線演算の高レベル API
- activity map と cache 管理
- 線源推定
- residual diagnosis
- planner
- closed-loop state machine
- experiment runner と logger

## 5. 基本型と状態モデル

すべての public data model は Pydantic v2 または frozen dataclass で定義する。単位を field 名に含める。

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping
import numpy as np

class SourceType(str, Enum):
    POINT = "point"
    SURFACE = "surface"
    VOLUME = "volume"

@dataclass(frozen=True)
class EmissionLine:
    energy_keV: float
    photons_per_decay: float

@dataclass(frozen=True)
class IsotopeSpec:
    isotope_id: str
    emission_lines: tuple[EmissionLine, ...]

@dataclass
class PointSourceState:
    source_id: str
    position_world_m: np.ndarray
    activity_bq: float
    isotope_id: str
    enabled: bool = True
    attached_prim_path: str | None = None

@dataclass
class SurfaceSourceState:
    source_id: str
    prim_path: str
    triangle_indices: np.ndarray          # [K]
    activity_bq_per_triangle: np.ndarray # [K]
    isotope_id: str
    enabled: bool = True

@dataclass(frozen=True)
class MaterialSpec:
    material_id: str
    energies_keV: np.ndarray
    linear_attenuation_m_inv: np.ndarray
    geometry_mode: str                    # "solid" | "thin_sheet"
    explicit_thickness_m: float | None

@dataclass(frozen=True)
class DetectorSpec:
    detector_id: str
    energy_bin_edges_keV: np.ndarray
    efficiency_energy_keV: np.ndarray
    intrinsic_efficiency: np.ndarray
    background_cps_per_bin: np.ndarray
    dead_time_s: float
    dose_conversion: np.ndarray | None

@dataclass(frozen=True)
class RadiationMeasurement:
    measurement_id: str
    detector_id: str
    timestamp_sim_s: float
    duration_s: float
    position_world_m: np.ndarray
    orientation_world_wxyz: np.ndarray
    counts_per_bin: np.ndarray
    expected_background_counts: np.ndarray
    dose_rate_sv_h: float | None
    covariance: np.ndarray
    scene_revision: int
```

### 5.1 Revision 管理

```python
@dataclass
class RevisionState:
    geometry_revision: int = 0
    material_revision: int = 0
    source_pose_revision: int = 0
    source_activity_revision: int = 0
    detector_revision: int = 0
```

更新規則:

- 遮蔽材、障害物、汚染物体の pose 変更: `geometry_revision += 1`
- 材質・厚さ変更: `material_revision += 1`
- 線源付き物体移動: `source_pose_revision += 1` と必要に応じて `geometry_revision += 1`
- 除染: `source_activity_revision += 1` のみ
- detector calibration 変更: `detector_revision += 1`

除染時は transfer matrix を再 ray trace せず、既存行列と新 activity vector の積だけで再計算できる設計にする。

## 6. USD metadata 実装

最初の版では独自 USD schema plugin を作らず、namespaced custom attribute を使用する。巨大な per-face activity array は `.npz` sidecar に保存し、USD には URI と checksum を格納する。

### 6.1 共通属性

```text
rad:role                        token
rad:enabled                     bool
rad:objectId                    string
```

`rad:role` の候補:

```text
source
attenuator
shield
contaminated_surface
contaminated_object
detector
decon_tool
disposal_zone
robot
obstacle
```

### 6.2 線源属性

```text
rad:source:type                 token    point|surface|volume
rad:source:isotopeId            string
rad:source:activityBq           double
rad:source:surfaceActivityBqM2  double
rad:source:activityMapUri       asset
rad:source:activityMapSha256    string
rad:source:hiddenFromEstimator  bool
rad:source:movableWithPrim      bool
```

### 6.3 遮蔽物属性

```text
rad:material:id                 string
rad:material:mode               token    solid|thin_sheet
rad:material:thicknessM         double
rad:material:attenuationUri     asset
rad:shield:movable              bool
rad:shield:resourceUnits        double
```

### 6.4 除染対象属性

```text
rad:decon:enabled               bool
rad:decon:activityMapUri        asset
rad:decon:efficiencyMean        double
rad:decon:efficiencyStd         double
rad:decon:minToolDwellS         double
rad:decon:surfaceId             string
```

### 6.5 操作対象属性

```text
rad:manipulation:movable        bool
rad:manipulation:removable      bool
rad:manipulation:graspFrame     token
rad:manipulation:massKg         double
rad:manipulation:disposalClass  string
```

### 6.6 `UsdRadiationRegistry`

実装ファイル:

```text
radcounter/isaac/usd/radiation_registry.py
```

public API:

```python
class UsdRadiationRegistry:
    def scan_stage(self, stage) -> "SceneDescriptor": ...
    def register_prim(self, prim_path: str) -> None: ...
    def unregister_prim(self, prim_path: str) -> None: ...
    def on_objects_changed(self, notice) -> None: ...
    def get_source_descriptors(self) -> list[SourceDescriptor]: ...
    def get_attenuator_descriptors(self) -> list[AttenuatorDescriptor]: ...
    def get_detector_descriptors(self) -> list[DetectorDescriptor]: ...
    def get_decon_surfaces(self) -> list[DeconSurfaceDescriptor]: ...
```

USD change notice を購読し、transform と radiation attribute の変更を分類して revision を更新する。変更ごとに全 stage を再走査しない。

## 7. USD mesh 抽出

実装ファイル:

```text
radcounter/isaac/usd/mesh_extractor.py
```

### 7.1 出力型

```python
@dataclass(frozen=True)
class MeshGeometry:
    mesh_id: str
    prim_path: str
    vertices_local_m: np.ndarray      # float32 [N,3]
    triangles: np.ndarray             # uint32 [M,3]
    world_transform: np.ndarray       # float64 [4,4]
    material_id_per_triangle: np.ndarray # int32 [M]
    is_closed_volume: bool
    geometry_mode: str
    explicit_thickness_m: float | None
    dynamic: bool
```

### 7.2 実装要件

- `UsdGeom.Mesh` の points、faceVertexCounts、faceVertexIndices を三角形化する。
- quad と n-gon は fan triangulation ではなく、USD triangulation utility または安定な ear clipping を使う。
- metersPerUnit を読み、すべて meter に変換する。
- world transform は double で保持し、Embree 入力時に float32 へ変換する。
- negative scale、non-uniform scale、instance、prototype を扱う。
- material binding と `rad:material:id` を triangle ごとに解決する。
- collision mesh と visual mesh が異なる場合、放射線計算用 mesh を明示属性で指定できるようにする。
- `rad:transportMesh=true` を優先し、なければ render mesh を使用する。
- source surface の triangle index と transport mesh の triangle index が一致するよう、抽出後の index map を保存する。

## 8. Embree backend

### 8.1 Python public interface

```python
from typing import Protocol

class RayTransportBackend(Protocol):
    def build_scene(
        self,
        meshes: list[MeshGeometry],
        material_table: "MaterialTable",
    ) -> None: ...

    def update_instance_transform(
        self,
        mesh_id: str,
        world_transform: np.ndarray,
    ) -> None: ...

    def remove_geometry(self, mesh_id: str) -> None: ...

    def commit_updates(self) -> int: ...

    def trace_transmission(
        self,
        origins_m: np.ndarray,       # [R,3]
        targets_m: np.ndarray,       # [R,3]
        energies_keV: np.ndarray,    # [E]
    ) -> np.ndarray:                 # [R,E]
        ...

    def trace_path_lengths(
        self,
        origins_m: np.ndarray,
        targets_m: np.ndarray,
    ) -> "PathLengthBatch": ...
```

### 8.2 C++ class

```cpp
class EmbreeTransportScene {
public:
    EmbreeTransportScene();
    ~EmbreeTransportScene();

    GeometryHandle addTriangleMesh(
        std::string meshId,
        py::array_t<float> vertices,
        py::array_t<uint32_t> triangles,
        py::array_t<int32_t> materialIds,
        GeometryMode mode,
        std::optional<float> thicknessM,
        bool dynamic);

    void updateTransform(const std::string& meshId,
                         py::array_t<double> transform44);
    void removeGeometry(const std::string& meshId);
    uint64_t commit();

    py::array_t<float> traceTransmission(
        py::array_t<float> origins,
        py::array_t<float> targets,
        py::array_t<float> energies,
        py::array_t<float> attenuationTable);

    PathLengthBatch tracePathLengths(...);
};
```

### 8.3 scene 構成

- 静的 environment mesh は一つまたは material ごとの少数 geometry にまとめる。
- 可動 shield、汚染物体、障害物は instance として登録し、transform update 後に commit する。
- ray query は C++ 内で並列化する。
- pybind11 binding は計算中 GIL を解放する。
- scene commit と trace を同時実行しない。read-write lock を使用する。

### 8.4 segment ray tracing

各 ray は source sample から detector までの有限 segment とする。

```text
origin = source + eps * dir
tnear = eps
tfar  = distance - eps
```

最近接 hit を反復取得し、`tnear = hit_t + eps` として全交差を収集する。最大 hit 数を設定し、超過時は error flag を返す。

### 8.5 solid geometry の通過長

- hit を距離順に保持する。
- geom ID ごとに entry/exit を判定する。
- normal orientation が信頼できる場合は `dot(ray_dir, geometric_normal)` で entry/exit を判断する。
- orientation が不安定な mesh は geom ごとの hit を pair にして長さを合計する。
- odd number の hit は invalid geometry として警告し、設定に応じて conservative または zero attenuation とする。
- nested material は active geometry の material をすべて加算する。

### 8.6 thin sheet

一枚板、フィルム、簡略化した遮蔽板は `thin_sheet` とし、交差一回につき

```text
effective_thickness = thickness / max(abs(dot(ray_dir, normal)), cos_limit)
```

を加算する。grazing angle で無限大にならないよう上限を設定する。

### 8.7 transmission

```text
T(E) = exp(-sum_m mu_m(E) * length_m)
```

数値 underflow を避けるため exponent を下限 clamp する。`trace_path_lengths` は debug・検証用、`trace_transmission` は高速通常経路とする。

### 8.8 fallback backend

`AnalyticTransportBackend` を必ず実装する。

- 遮蔽物なし
- 単一平板の解析式
- CI で Embree 未インストール時にも core test を実行可能

## 9. 放射線源モデル

### 9.1 point source

各 isotope emission line に対して

```text
photon_rate_s = activity_bq * photons_per_decay
fluence_rate = photon_rate_s / (4*pi*r^2)
```

を計算する。

### 9.2 surface source

triangle ごとに activity を持たせる。

```text
activity_triangle_bq = surface_activity_bq_m2 * triangle_area_m2
```

quadrature mode:

- `centroid`: triangle centroid 一点
- `stratified_n`: triangle 内に n 点
- `adaptive`: detector との距離と triangle サイズで n を調整

各 sample は local coordinate と weight を持つ。

```python
@dataclass(frozen=True)
class SourceSampleBatch:
    positions_world_m: np.ndarray  # [S,3]
    activity_bq: np.ndarray        # [S]
    isotope_index: np.ndarray      # [S]
    source_id_index: np.ndarray    # [S]
    triangle_index: np.ndarray     # [S], point source は -1
```

可動 object に付随する sample は local position を cache し、object transform 変更時に world position だけ更新する。

### 9.3 volume source

初回論文に必須ではないが interface は用意する。voxel center と voxel activity の sample batch に変換する。

### 9.4 activity repository

```python
class SourceRepository:
    def get_truth_samples(self) -> SourceSampleBatch: ...
    def get_belief_basis_samples(self) -> SourceSampleBatch: ...
    def scale_triangle_activity(self, source_id, triangle_ids, factors): ...
    def move_attached_sources(self, prim_path, transform): ...
    def deactivate_source(self, source_id): ...
```

Truth と Belief の repository instance は別にする。

## 10. 放射線 forward model

```python
class RadiationForwardModel:
    def predict_count_rate(
        self,
        detector_poses: np.ndarray,
        source_samples: SourceSampleBatch,
        detector: DetectorSpec,
        scene_snapshot: SceneSnapshot,
    ) -> CountRatePrediction: ...

    def predict_dose_rate(...): ...
    def build_transfer_matrix(...): ...
```

energy line `e`、source sample `s`、detector pose `d` の寄与:

```text
lambda_sde = A_s * Y_e * G(r_sd) * T_sd(E_e) * epsilon_d(E_e, theta_sd)
```

- `G(r)=1/(4*pi*max(r,r_min)^2)`
- `T` は Embree attenuation
- `epsilon` は detector response interpolation
- energy bin に集約
- background を加える
- dead time model を適用

### 10.1 direct/scatter plugin

```python
class ScatterModel(Protocol):
    def add_scatter(self, direct_prediction, context) -> np.ndarray: ...
```

実装:

- `NoScatterModel`
- `EmpiricalBuildupModel`
- `TruthOnlyBiasModel`: ground truth 側だけに spatial bias、energy redistribution、background drift を与え、推定モデルとの mismatch を生成

散乱を未実装のまま暗黙に無視せず、設定ファイルとログに必ず model 名を保存する。

## 11. detector 実装

### 11.1 センサ階層

```python
class RadiationSensor:
    def start_measurement(self, duration_s: float) -> str: ...
    def update(self, sim_time_s: float) -> None: ...
    def cancel(self) -> None: ...
    def get_latest(self) -> RadiationMeasurement | None: ...

class OmnidirectionalCounter(RadiationSensor): ...
class RotatingShieldCounter(RadiationSensor): ...
class DoseRateMeter(RadiationSensor): ...
```

### 11.2 計測状態機械

```text
IDLE -> INTEGRATING -> FINALIZING -> READY
                  \-> CANCELLED
```

積算中に detector が動く場合は `trajectory_subsamples` 回だけ pose を採取して平均 rate を求める。初期設定は stationary measurement とする。

### 11.3 Poisson sampling

```text
expected_counts_bin = rate_cps_bin * duration_s
observed_counts_bin ~ Poisson(expected_counts_bin)
```

乱数 generator は run seed から detector ごとの child seed を作る。再現性試験で完全一致すること。

### 11.4 回転遮蔽体

二方式を実装する。

1. `physical_geometry`: 実際の遮蔽体 mesh を detector 周囲で回転し、Embree で減衰を計算。
2. `response_mask`: 事前計算した角度 response を掛ける高速モード。

回転角、回転速度、各角度の積算時間、encoder noise を記録する。

## 12. transfer matrix と cache

### 12.1 行列定義

```text
y = H x + b
```

- `x`: candidate source basis の activity
- `H`: detector pose × energy bin × candidate basis の unit-activity count response
- `b`: background

### 12.2 cache key

```python
@dataclass(frozen=True)
class TransferMatrixKey:
    detector_pose_hash: str
    basis_hash: str
    geometry_revision: int
    material_revision: int
    detector_revision: int
    energy_grid_hash: str
```

`source_activity_revision` は key に含めない。除染は `x` のみを変える。

### 12.3 partial invalidation

- shield pose 変更: geometry revision により全体無効化する MVP を実装。
- その後、shield bounding box と交差可能な ray の行のみ再計算する optional optimization。
- movable source pose 変更: source basis columns のみ更新。

### 12.4 chunk 計算

大規模 map は detector evaluation points と source samples を chunk し、最大一時メモリを設定値以下にする。

## 13. 線量マップ

```python
class DoseMapEvaluator:
    def create_planar_grid(bounds, z_m, resolution_m) -> EvaluationGrid: ...
    def create_3d_grid(bounds, resolution_m) -> EvaluationGrid: ...
    def mask_occupied_cells(grid, collision_scene) -> EvaluationGrid: ...
    def evaluate(grid, state, chunk_size) -> DoseMap: ...
```

出力:

```python
@dataclass(frozen=True)
class DoseMap:
    points_world_m: np.ndarray
    dose_rate_sv_h: np.ndarray
    standard_deviation: np.ndarray | None
    revision: RevisionState
```

## 14. visualization

実装ファイル:

```text
radcounter/isaac/visualization/
├── dose_map_visualizer.py
├── source_estimate_visualizer.py
├── residual_visualizer.py
├── ray_debug_visualizer.py
└── action_visualizer.py
```

要件:

- 2D heatmap は `UsdGeom.Points` または instancer を使い、cell ごとの cube を大量生成しない。
- color range は fixed、percentile、log の三方式。
- Truth source overlay は debug 権限かつ明示 toggle 時のみ表示。
- Belief source、uncertainty、predicted post-action、observed post-action、normalized residual を別 layer にする。
- selected ray の material path と通過長を表示できる。

## 15. robot abstraction

```python
class RobotController(Protocol):
    async def navigate_to(self, pose, timeout_s) -> "ExecutionStatus": ...
    async def move_end_effector(self, pose, timeout_s) -> "ExecutionStatus": ...
    async def execute_joint_trajectory(self, trajectory) -> "ExecutionStatus": ...
    async def grasp(self, target_prim_path) -> "ExecutionStatus": ...
    async def release(self) -> "ExecutionStatus": ...
    async def stop(self) -> None: ...

class DeterministicRobotController(RobotController): ...
class IsaacPhysicsRobotController(RobotController): ...
class Ros2RobotController(RobotController): ...
```

### 15.1 measurement robot

- differential drive または omnidirectional mobile base
- detector mast
- detector pose は robot base transform と sensor extrinsic から取得
- Nav2 接続は optional

### 15.2 countermeasure robot

- mobile base + manipulator + gripper を推奨
- 初期段階では fixed manipulator でもよいが、controller interface は mobile manipulator を前提にする
- shield、obstacle、contaminated object を操作できる
- decon tool を tool changer または固定 attachment として持つ

### 15.3 grasp 実装

二方式:

- deterministic: target の grasp frame 到達可能性と collision-free 条件を確認後、target prim を gripper prim に parent/constraint する。
- physics: surface gripper または fixed joint constraint を使用し、接触と相対 pose を確認する。

把持終了後に対象 pose を radiation registry に反映する。

## 16. action model

```python
class ActionType(str, Enum):
    MEASURE = "measure"
    DECONTAMINATE = "decontaminate"
    PLACE_SHIELD = "place_shield"
    MOVE_SHIELD = "move_shield"
    MOVE_OBJECT = "move_object"
    REMOVE_OBJECT = "remove_object"
    REPAIR_ACTION = "repair_action"

@dataclass(frozen=True)
class CountermeasureAction:
    action_id: str
    action_type: ActionType
    robot_id: str
    target_prim_path: str | None
    target_region: dict | None
    target_pose_world: np.ndarray | None
    parameters: dict
    predicted_duration_s: float
    resource_cost: dict[str, float]

@dataclass(frozen=True)
class ActionResult:
    action_id: str
    status: str
    started_sim_s: float
    completed_sim_s: float
    public_details: dict
    truth_details: dict | None
    before_revision: RevisionState
    after_revision: RevisionState
```

`truth_details` は experiment logger のみアクセス可能で、Estimator/Planner には渡さない。

## 17. 除染実装

### 17.1 activity map

surface source は triangle ごとの activity を `.npz` で持つ。

```text
triangle_indices
activity_bq
last_treated_step
cumulative_treatment_exposure
```

### 17.2 tool footprint

`DeconToolModel`:

```python
@dataclass(frozen=True)
class DeconToolSpec:
    footprint_sample_points_local_m: np.ndarray
    treatment_axis_local: np.ndarray
    max_contact_distance_m: float
    max_normal_angle_deg: float
    max_surface_speed_m_s: float
    rate_constant_s_inv: float
```

physics tick ごとに footprint sample ray を tool axis 方向に cast する。

有効接触条件:

- target surface までの距離が閾値以下
- tool axis と surface normal の角度が閾値以下
- end-effector の surface tangential speed が上限以下
- target prim が `rad:decon:enabled=true`

triangle ごとに exposure を積算する。

```text
E_i += contact_weight * dt
nominal_removal_fraction_i = 1 - exp(-k * E_i)
```

### 17.3 ground truth failure

Truth 側では

```text
actual_fraction_i = clamp(nominal_fraction_i * local_efficiency_i, 0, 1)
local_efficiency_i ~ spatially correlated random field
```

とし、未処理 spot、工具位置ずれ、効率ばらつきを生成できる。

### 17.4 removed activity

設定により二方式:

- `discard`: 除去 activity を scene から消す。
- `transfer_to_waste`: 除去 activity を waste container source に移す。

研究用には後者を推奨する。除染しただけで放射能が消滅したことにしない。

### 17.5 API

```python
class DecontaminationExecutor:
    async def execute(self, action, robot, truth_state) -> ActionResult: ...
    def preview_nominal_effect(self, action, belief_state) -> SourceStateDelta: ...
```

## 18. 遮蔽実装

### 18.1 shield asset

各 shield asset は以下を持つ。

- visual mesh
- collider
- mass/inertia
- grasp frame
- support/contact frame
- radiation transport mesh
- material ID
- solid または thin sheet mode
- nominal thickness
- resource units

### 18.2 shield action

```text
PLAN -> NAVIGATE_TO_SHIELD -> GRASP -> NAVIGATE_TO_TARGET
-> PLACE -> RELEASE -> WAIT_SETTLE -> COMMIT_RADIATION_SCENE -> COMPLETE
```

`WAIT_SETTLE` では線速度・角速度が閾値以下になるまで待つ。timeout 時は FAILED または PARTIAL。

### 18.3 遮蔽ずれ

Truth mode では target pose に対し平行移動・回転誤差を加える。

```text
actual_pose = target_pose * pose_error_transform
```

pose は USD から読み取った actual pose を radiation scene に反映する。Planner の predicted pose を直接使わない。

### 18.4 即時効果測定

shield placement 完了後:

1. geometry revision を更新
2. Embree instance transform 更新・commit
3. selected verification poses の predicted dose を計算
4. measurement robot を verification pose に移動
5. measurement 実行
6. residual を生成

## 19. 汚染物体・障害物の移動と撤去

### 19.1 contaminated object

source sample は object local frame に保持する。object pose 更新時に world sample pose を更新する。

### 19.2 obstacle

非汚染 obstacle は線源を持たないが、robot reachability、path planning、放射線 attenuation に影響し得る。`rad:material:id` が設定されていれば attenuation geometry として登録する。

### 19.3 move action

```text
NAVIGATE -> GRASP/PUSH -> MOVE -> RELEASE -> SETTLE -> UPDATE SOURCE/GEOMETRY
```

push と pick の action subtype を持つ。

### 19.4 remove action

単に source を API で無効化してはならない。対象が disposal zone に入ったことを検証した後に、次のいずれかを行う。

- disposal container の shielding を含めた状態で scene 内に残す
- evaluation domain 外へ搬出して source を deactivate

ログに除去前後の activity 保存先を記録する。

## 20. measurement・source estimation

### 20.1 candidate basis

最初の実装は二種類。

- 3D grid basis: unknown point/small voxel source 用
- surface triangle basis: 床・壁・物体表面の汚染分布用

### 20.2 Poisson sparse estimator

観測 count `y`、response matrix `H`、background `b` に対し

```text
minimize_x>=0  sum_i [(Hx+b)_i - y_i log((Hx+b)_i)]
              + lambda_l1 ||x||_1
              + lambda_tv TV(x)
```

を解く。

実装クラス:

```python
class SourceEstimator(Protocol):
    def fit(self, measurements, basis, forward_model) -> "SourceEstimate": ...
    def update(self, previous, new_measurements, action_context) -> "SourceEstimate": ...

class GridPoissonSparseEstimator(SourceEstimator): ...
class SurfacePoissonTVEstimator(SourceEstimator): ...
class PFPlusMLEEstimator(SourceEstimator): ...   # optional phase
```

### 20.3 solver

初期版は SciPy を使用し、次を実装する。

- nonnegative L-BFGS-B for unregularized MLE
- proximal gradient/FISTA for L1
- graph incidence matrix を使う TV proximal の簡略版、または split Bregman
- gradient test を finite difference で検証

### 20.4 local refinement

sparse grid で得た上位候補を連続座標 MLE で refinement する。

```text
coarse sparse grid -> connected components -> source seeds
-> continuous position/activity MLE
```

### 20.5 uncertainty

最低限、active set 上の Fisher information 近似を実装する。

```text
F = H_A^T diag(1/max(lambda, eps)) H_A + regularization
Cov = pseudo_inverse(F)
```

bootstrap option も用意する。

### 20.6 出力

```python
@dataclass(frozen=True)
class SourceEstimate:
    estimate_id: str
    basis_activity_bq: np.ndarray
    covariance_diag: np.ndarray
    point_hypotheses: tuple
    predicted_measurements: np.ndarray
    objective_value: float
    converged: bool
    diagnostics: dict
```

## 21. prediction–measurement residual

### 21.1 predicted post-action

Planner は BeliefState の clone に action の nominal effect を適用する。

```python
predicted_belief_after = action_model.preview(action, belief_before)
predicted_measurement = forward_model.predict(
    verification_poses,
    predicted_belief_after,
)
```

### 21.2 observed post-action

ActionExecutor は TruthState に stochastic actual effect を適用し、実際の USD pose と source activity を更新する。verification measurement を実施する。

### 21.3 normalized residual

```text
r = y_observed - y_predicted
z = r / sqrt(max(y_predicted + variance_model, 1))
```

energy bin、pose、time を保持する。

### 21.4 residual hypotheses

```python
class ResidualHypothesis(Protocol):
    hypothesis_id: str
    def fit(self, context) -> HypothesisFit: ...

class DeconResidualHypothesis: ...
class ShieldPoseErrorHypothesis: ...
class HiddenSourceHypothesis: ...
class GlobalGainBackgroundHypothesis: ...
class SourceLocalizationErrorHypothesis: ...
```

#### DeconResidualHypothesis

処理領域内 activity の残存係数を回帰する。

#### ShieldPoseErrorHypothesis

nominal pose 周辺の有限候補を生成し、各候補で predicted measurement を再計算して likelihood 最大の pose correction を求める。

#### HiddenSourceHypothesis

既知 source contribution を引いた residual に対し、未使用 candidate basis 上で sparse inversion を行う。

#### GlobalGainBackgroundHypothesis

```text
y_observed ≈ gain * y_predicted + background_offset
```

を fit する。

### 21.5 hypothesis selection

各仮説について negative log-likelihood と parameter 数から BIC を計算する。最良仮説と confidence を返し、BeliefState と action effect parameter を更新する。

## 22. planner

### 22.1 action candidate generator

```python
class ActionCandidateGenerator:
    def generate_measurement_actions(...): ...
    def generate_decon_actions(...): ...
    def generate_shield_actions(...): ...
    def generate_move_remove_actions(...): ...
    def generate_repair_actions(...): ...
```

### 22.2 feasibility

各 candidate に対し:

- mobile path の有無
- manipulator reachability
- collision
- grasp frame
- shield placement stability
- disposal zone capacity
- resource availability
- robot availability

を評価する。

MVP では deterministic geometric feasibility、physics mode では controller dry-run を使う。

### 22.3 objective

```text
score(a) =
  w_dose * expected_task_path_dose_after(a)
+ w_peak * expected_peak_dose_after(a)
+ w_unc  * residual_source_uncertainty_after(a)
+ w_time * action_time(a)
+ w_res  * resource_cost(a)
+ w_risk * robot_execution_risk(a)
- w_info * expected_information_gain(a)
```

小さいほど良いとする。

### 22.4 resource state

```python
@dataclass
class ResourceState:
    remaining_measurement_time_s: float
    remaining_work_time_s: float
    remaining_robot_runtime_s: dict[str, float]
    remaining_shield_units: dict[str, int]
    remaining_decon_media: float
    remaining_countermeasure_count: int
```

### 22.5 baseline planners

論文比較用に必ず実装する。

- `OpenLoopPlanner`
- `GreedyDoseReductionPlanner`
- `NearestSourcePlanner`
- `RandomPlanner`
- `OraclePlanner` — TruthState を使うが実験評価専用
- `ClosedLoopResidualPlanner` — 提案手法

## 23. closed-loop orchestrator

```python
class WorkflowState(str, Enum):
    INITIALIZE = "initialize"
    MEASURE = "measure"
    ESTIMATE = "estimate"
    PLAN = "plan"
    PREDICT = "predict"
    EXECUTE = "execute"
    VERIFY = "verify"
    DIAGNOSE = "diagnose"
    UPDATE = "update"
    COMPLETE = "complete"
    FAILED = "failed"
```

```python
class ClosedLoopCoordinator:
    async def run_episode(self, config) -> EpisodeResult: ...
    async def step(self) -> WorkflowState: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def stop(self) -> None: ...
```

遷移:

```text
INITIALIZE
 -> MEASURE
 -> ESTIMATE
 -> PLAN
 -> PREDICT
 -> EXECUTE
 -> VERIFY
 -> DIAGNOSE
 -> UPDATE
 -> PLAN or COMPLETE
```

終了条件:

- task path dose が閾値以下
- peak dose が閾値以下
- resources 枯渇
- 最大 step 数
- 有効 action なし
- safety violation

各遷移で immutable snapshot を保存する。

## 24. UI

`radcounter.isaac` は current UI template を基に実装する。

### 24.1 Frames

1. **Scenario**
   - config path
   - Load/Reset/Clear
   - seed
   - deterministic/physics mode
2. **Radiation Scene**
   - stage scan
   - rebuild Embree
   - revision display
   - material/source counts
3. **Measurement**
   - detector selection
   - duration
   - start/cancel
   - latest count/dose
4. **Estimation**
   - basis selection
   - regularization parameters
   - run/update
5. **Countermeasure**
   - action type
   - target prim/region
   - preview effect
   - execute
6. **Closed Loop**
   - one step
   - auto run
   - pause/stop
   - current workflow state
7. **Visualization**
   - truth toggle
   - estimate toggle
   - dose map
   - residual map
   - ray debug
8. **Experiment**
   - baseline
   - run ID
   - save report

UI callback 内で重い計算を同期実行しない。async task を生成し、進捗と cancel token を管理する。

## 25. ROS 2 interface

core functionality は ROS 2 なしで動作する。ROS 2 は adapter とする。

### 25.1 messages

`RadiationMeasurement.msg`

```text
std_msgs/Header header
string measurement_id
string detector_id
geometry_msgs/Pose detector_pose
float64 duration_s
float64[] energy_bin_edges_kev
uint32[] counts_per_bin
float64 dose_rate_sv_h
float64[] covariance_flat
uint64 scene_revision
```

`SourceEstimate.msg`

```text
std_msgs/Header header
string estimate_id
geometry_msgs/Point[] positions
float64[] activity_bq
float64[] position_covariance_flat
float64[] activity_std_bq
```

`CountermeasureStatus.msg`

```text
std_msgs/Header header
string action_id
string action_type
string state
float32 progress
string message
```

### 25.2 actions/services

```text
MeasureRadiation.action
ExecuteCountermeasure.action
GetDoseMap.srv
EvaluateCountermeasure.srv
ResetEpisode.srv
```

### 25.3 robot topics

標準 `/tf`, `/joint_states`, `/cmd_vel`, FollowJointTrajectory、MoveIt 2 を使用する。

## 26. scenario configuration

```yaml
schema_version: 1
scenario_id: closed_loop_demo_001
seed: 42

simulation:
  action_mode: deterministic
  physics_dt_s: 0.008333333
  rendering_dt_s: 0.033333333
  max_episode_steps: 8

world:
  usd_path: assets/environments/demo_room.usd
  radiation_transport_mesh_policy: tagged_or_render

radiation:
  backend: embree
  scatter_model_truth: truth_only_bias
  scatter_model_planner: none
  min_distance_m: 0.05
  ray_epsilon_m: 1.0e-4
  max_hits_per_ray: 128
  source_quadrature: adaptive
  chunk_rays: 100000

materials:
  directory: configs/materials

sources:
  - id: floor_contamination
    type: surface
    prim_path: /World/Room/Floor
    isotope_id: example_isotope
    activity_map_uri: assets/environments/maps/floor_activity.npz
    hidden_from_estimator: false
  - id: hidden_object_source
    type: point
    prim_path: /World/Props/Box03
    activity_bq: 1.0e6
    isotope_id: example_isotope
    hidden_from_estimator: true

robots:
  measurement:
    config: configs/robots/measurement_robot.yaml
    initial_pose: [0.5, 0.5, 0.0, 0.0]
  countermeasure:
    config: configs/robots/countermeasure_robot.yaml
    initial_pose: [1.0, 0.5, 0.0, 0.0]

detectors:
  - config: configs/detectors/rotating_shield_counter.yaml
    robot_id: measurement
    mount_prim: /World/MeasurementRobot/SensorMount

truth_action_uncertainty:
  decon_efficiency_mean: 0.75
  decon_efficiency_std: 0.15
  shield_translation_std_m: 0.03
  shield_rotation_std_deg: 2.0
  grasp_failure_probability: 0.02

estimator:
  type: surface_poisson_tv
  lambda_l1: 1.0e-3
  lambda_tv: 1.0e-2
  max_iterations: 500
  uncertainty: fisher

planner:
  type: closed_loop_residual
  weights:
    task_path_dose: 1.0
    peak_dose: 0.3
    uncertainty: 0.2
    time: 0.05
    resource: 0.1
    execution_risk: 0.1
    information_gain: 0.2

resources:
  measurement_time_s: 600
  work_time_s: 1800
  robot_runtime_s:
    measurement: 1800
    countermeasure: 1800
  shields:
    lead_panel_small: 2
  decon_media_units: 100
  max_countermeasure_count: 6

verification:
  poses:
    - [1.0, 1.0, 0.8]
    - [2.0, 1.0, 0.8]
  measurement_duration_s: 10.0

outputs:
  run_root: outputs
  save_truth: true
  save_transfer_matrices: false
  save_dose_maps: true
  save_video: false
```

`validate_scenario.py` で schema、asset path、unit、prim path、material table の整合性を Isaac Sim 起動前に検査する。

## 27. logging と実験再現性

run directory:

```text
outputs/<scenario>/<timestamp>_<run_id>/
├── manifest.json
├── resolved_config.yaml
├── events.jsonl
├── measurements.parquet
├── estimates.parquet
├── actions.parquet
├── resources.parquet
├── metrics.json
├── maps/
├── snapshots/
└── report.html
```

`manifest.json`:

- git commit SHA
- dirty status
- Isaac Sim version
- Embree version
- OS、CPU、GPU
- Python package versions
- random seeds
- config SHA256
- asset SHA256

重要な event:

- scene loaded
- radiation scene committed
- measurement started/completed
- estimate completed
- action previewed/started/completed/failed
- verification completed
- residual hypothesis selected
- belief updated
- resource consumed
- episode completed

## 28. test suite

### 28.1 unit tests — radiation

1. 自由空間 point source が `1/r^2` に従う。
2. 単一 slab が `exp(-mu*l)` に従う。
3. 二材料 slab の exponent が加算される。
4. thin sheet の斜入射厚さが正しい。
5. closed cube の entry/exit 通過長が正しい。
6. nested solids が正しい。
7. surface source rectangle が高密度数値積分の基準値に収束する。
8. zero activity、disabled source、zero efficiency を扱える。
9. Poisson seed で結果が再現する。
10. cache invalidation が revision 規則どおり。

### 28.2 unit tests — actions

1. 50% nominal decon で対象 triangle activity が半減する。
2. 非対象 triangle は変化しない。
3. repeated decon が累積する。
4. removed activity が waste source に移る。
5. shield placement で geometry revision が増える。
6. shield move で transmission が変化する。
7. contaminated object move で source sample pose が追従する。
8. disposal zone 前には source が deactivate されない。

### 28.3 unit tests — estimation

1. noiseless single source を回収する。
2. Poisson noisy single source の平均誤差が許容範囲。
3. two-source case。
4. surface sparse patch。
5. regularization zero と nonzero。
6. gradient finite-difference check。
7. Fisher covariance shape/positive semidefinite。

### 28.4 unit tests — residual

1. decon residual hypothesis を正しく選ぶ。
2. shield translation error を近似回収する。
3. hidden source を新規 candidate として検出する。
4. global gain error を source error と誤分類しにくい。

### 28.5 integration tests — headless Isaac

`test_closed_loop_smoke.py`:

1. stage load
2. robot spawn
3. source/material scan
4. Embree build
5. measurement robot move
6. initial measurement
7. estimate
8. shield action
9. verification measurement
10. residual update
11. second action plan
12. output files validation

`test_all_countermeasures.py`:

- decon
- shield
- move contaminated object
- remove object
- move non-contaminated obstacle

### 28.6 regression tests

fixed seed の小シナリオについて、主要 metric と map hash を保存する。version update で差が出た場合は理由をレビューする。

## 29. validation criteria

### 29.1 physics/math

- free-space analytic relative error < 1e-6
- slab attenuation relative error < 1e-4
- surface quadrature convergence を report
- material path debug で期待長と一致

### 29.2 functional

- 全 action が deterministic mode で完走
- physics mode で最低一つの shield pick-and-place と obstacle move が完走
- decon footprint が tool path に沿って activity map を更新
- action 後の detector measurement が scene state を反映
- estimator が TruthState を参照していないことを test double で保証

### 29.3 performance target

基準機を manifest に明記し、次を初期目標とする。

- 100,000 segment rays の transmission query: 0.2 s 以下
- 2D 128×128 map、source samples 2,000、cache 未使用: 2 s 以下
- activity-only update 後の map: 0.1 s 以下
- shield pose update + verification 32 poses: 0.5 s 以下

未達でも correctness を優先し、benchmark result を保存する。性能値を論文に使う場合は reference hardware と scene complexity を併記する。

## 30. 実装順序

### Milestone 0: scaffold

- extension templates
- core package
- CI
- config validation
- logging skeleton

完了条件: Isaac Sim で UI が表示され、headless startup test が通る。

### Milestone 1: analytic radiation core

- data models
- point source
- material interpolation
- detector model
- analytic backend
- unit tests

完了条件: free-space/slab tests pass。

### Milestone 2: USD registry + mesh extraction

- custom attributes
- stage scan
- mesh triangulation
- revision management

完了条件: demo USD から期待した source/attenuator descriptors が得られる。

### Milestone 3: Embree backend

- C++ extension
- pybind
- scene build/update
- multi-hit path length
- benchmarks

完了条件: analytic slab/cube tests と batch performance test pass。

### Milestone 4: source sampling + radiation sensor

- surface source
- detector integration
- Poisson measurement
- rotating shield sensor

完了条件: robot-mounted sensor が計測を返す。

### Milestone 5: map/cache/visualization

- transfer matrix
- cache
- dose map
- UI visualization

完了条件: decon activity update が ray rebuild なしに map を更新。

### Milestone 6: deterministic countermeasure actions

- decon
- shield
- move/remove
- resources
- action state machine

完了条件: 全 action の統合試験 pass。

### Milestone 7: robot physics execution

- navigation adapter
- manipulator adapter
- grasp/release
- shield placement
- obstacle/source object movement
- decon tool contact

完了条件: physics mode smoke test pass。

### Milestone 8: estimation

- grid/surface basis
- Poisson MLE/L1/TV
- uncertainty
- estimate visualization

完了条件: synthetic recovery tests pass。

### Milestone 9: residual diagnosis

- predicted post-action
- verification measurement
- four residual hypotheses
- belief update

完了条件: fault injection tests pass。

### Milestone 10: planner + closed loop

- candidates
- feasibility
- objective
- resources
- baselines
- coordinator

完了条件: closed-loop demo が open-loop baseline より指定 metric で改善するテストを生成。ただし固定の有意差を CI pass 条件にはしない。

### Milestone 11: ROS 2 and experiment automation

- messages/actions
- MoveIt/Nav2 adapter
- batch runner
- reports

完了条件: ROS 2 measurement と action round trip、headless sweep 実行。

## 31. Definition of Done

次の全項目を満たした時点で「シミュレーション実装完了」とする。

- GUI と headless の双方で起動できる。
- point/source surface/contaminated object source を扱える。
- air/solid/thin-sheet attenuation を扱える。
- decon、shield、move、remove が scene と放射線場を更新する。
- mobile robot と manipulator の action interface がある。
- deterministic mode で全 action が完走する。
- physics mode で shield と object manipulation の実例がある。
- detector count と dose map を action 前後で即時更新できる。
- source estimation と uncertainty が動く。
- predicted/observed/residual が保存・可視化される。
- residual に基づく belief update が動く。
- resource-constrained planner と baselines が動く。
- TruthState leakage test が通る。
- 全 unit/integration tests が通る。
- version/seed/config/hardware を含む manifest が保存される。
- API と scenario schema の文書がある。
- 一つの end-to-end closed-loop demo と再現スクリプトがある。

## 32. 実装上の禁止事項

- Estimator/Planner から TruthState にアクセスしない。
- 除染と遮蔽を同一の activity scale 操作として実装しない。
- 遮蔽設置後に predicted pose をそのまま radiation engine に渡さず、USD actual pose を読む。
- 毎 physics tick に全 dose map を再計算しない。
- 巨大 per-face activity array を USD custom attribute に直接埋め込まない。
- source activity、dose、count、length の単位を曖昧にしない。
- UI callback で重い同期計算をしない。
- 物理データ値を出典・version なしに hardcode しない。
- action 実装を単なる見た目の animation にしない。scene state と radiation state の一貫性を必ず更新する。
- action 後の効果を TruthState の差から Planner に直接伝えない。必ず再計測を経由する。

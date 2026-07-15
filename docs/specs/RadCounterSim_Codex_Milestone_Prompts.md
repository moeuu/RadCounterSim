# RadCounterSim — Codex 用段階実装プロンプト

## 使用方法

一度に全機能を実装させず、以下を上から一つずつ Codex に渡す。各 milestone では、まず既存コードと対象 Isaac Sim version の API を調査し、設計変更を `docs/decisions/ADR-XXXX.md` に残し、テスト成功後に次へ進む。

共通指示:

```text
- RadCounterSim_Codex_Implementation_Spec.md を最上位仕様とする。
- 関係のない既存コードを変更しない。
- public API には型注釈と docstring を付ける。
- 単位を変数名に含める。
- TruthState と BeliefState の依存方向を破らない。
- 実装前に変更予定ファイル一覧とテスト計画を提示する。
- 実装後に実行したコマンド、成功したテスト、未解決事項を報告する。
- mock だけで完了扱いにしない。ただし外部依存がない unit-testable core を先に作る。
```

---

## Prompt 0 — リポジトリと extension scaffold

```text
RadCounterSim_Codex_Implementation_Spec.md の Milestone 0 を実装してください。

要件:
1. Isaac Sim 6.0.1 source workspace の UI extension template と C++ extension templateを使用する。
2. radcounter.core、radcounter.radiation.native、radcounter.isaac の三層を作る。
3. UI extension は Examples Browser に登録し、Load/Reset/Clear が動く。
4. headless startup test と純 Python unit test の実行経路を用意する。
5. YAML scenario schema、Pydantic config model、validate_scenario.py を作る。
6. run manifest と JSONL event logger の skeleton を作る。
7. CI 用コマンドと local setup を README に書く。
8. Isaac Sim の extension dependency 名はローカル 6.0.1 の公式 template/example から確認し、推測で書かない。

受入条件:
- extension startup test pass
- pytest の空でない test suite pass
- sample scenario validation pass
- GUI で RadCounterSim example が表示される
```

## Prompt 1 — 基本モデルと解析 radiation backend

```text
Milestone 1 を実装してください。

実装対象:
- SourceType、EmissionLine、IsotopeSpec、PointSourceState、SurfaceSourceState
- MaterialSpec、DetectorSpec、RadiationMeasurement、RevisionState
- MaterialTable の energy interpolation
- Point source inverse-square forward model
- NoScatterModel
- AnalyticTransportBackend: no attenuation と single slab
- detector efficiency、background、dead time、Poisson sampling
- deterministic RNG hierarchy

テスト:
- 1/r^2
- exp(-mu*l)
- energy interpolation
- Poisson reproducibility
- zero/disabled source
- unit validation

受入条件:
- 全 math unit test pass
- source/length/activity/count/dose の単位が docs に明記
```

## Prompt 2 — USD radiation metadata と mesh extraction

```text
Milestone 2 を実装してください。

実装対象:
- rad:* custom attributes の helper
- UsdRadiationRegistry
- SceneDescriptor
- UsdGeom.Mesh extraction
- triangulation、metersPerUnit、world transform、instances、negative/non-uniform scale
- material ID per triangle
- transport mesh selection
- USD change notice と revision update
- per-face activity sidecar URI/checksum

作る demo USD:
- room
- concrete wall
- thin shield panel
- contaminated floor
- movable contaminated box
- detector mount

テスト:
- demo stage scan
- descriptor counts
- triangle/area/transform
- revision classification
- full rescan が不要な変更通知
```

## Prompt 3 — Embree C++ backend

```text
Milestone 3 を実装してください。

実装対象:
- EmbreeTransportScene C++ class
- pybind11 bindings
- triangle mesh registration
- static geometry と dynamic instances
- transform update、remove、commit
- finite segment rays
- repeated closest-hit による全交差収集
- solid entry/exit path length
- thin-sheet effective thickness
- material-wise path lengths
- energy-wise transmission
- GIL release、parallel batch query、thread safety
- explicit error handling

テスト geometry:
- slab
- cube
- nested cubes
- two materials
- thin sheet normal/oblique
- moving instance
- odd-hit invalid mesh

benchmark:
- 1k、10k、100k rays
- result を JSON に保存

受入条件:
- analytic tests pass
- memory leak がない
- commit と trace の race test pass
```

## Prompt 4 — Surface source、sensor、dose map、cache

```text
Milestone 4 と 5 の radiation 部分を実装してください。

実装対象:
- SourceSampleBatch
- point/surface quadrature: centroid、stratified、adaptive
- attached source transform update
- RadiationForwardModel
- OmnidirectionalCounter、RotatingShieldCounter、DoseRateMeter
- measurement state machine
- moving integration trajectory sampling
- transfer matrix H
- TransferMatrixCache と revision rules
- DoseMapEvaluator
- chunk processing

重要:
- decon activity change では ray trace cache を無効化しない。
- shield geometry change では geometry-dependent cache を無効化する。
- truth scatter/bias と planner model を分ける。

テスト:
- surface rectangle convergence
- sensor on moving prim
- cache hit/miss
- activity-only fast update
- rotating shield physical geometry mode
```

## Prompt 5 — UI と visualization

```text
Milestone 5 の UI/visualization を実装してください。

Frames:
- Scenario
- Radiation Scene
- Measurement
- Estimation placeholder
- Countermeasure placeholder
- Closed Loop placeholder
- Visualization
- Experiment

Visualization:
- 2D dose heatmap
- truth source debug overlay
- belief source overlay placeholder
- selected ray path/material lengths
- revision and timing display

要件:
- heavy computation は async task
- cancel token
- stale revision の result は破棄
- UI callback で blocking trace をしない
```

## Prompt 6 — Deterministic countermeasure actions

```text
Milestone 6 を実装してください。

実装対象:
- CountermeasureAction、ActionResult、ActionType
- action lifecycle/state machine
- ResourceState と consumption
- DeterministicRobotController
- DecontaminationExecutor
- ShieldPlacementExecutor
- MoveObjectExecutor
- RemoveObjectExecutor
- DisposalZone
- truth action uncertainty
- public_details と truth_details の分離

除染:
- triangle activity map
- footprint path
- exposure model
- spatial efficiency random field
- discard/transfer_to_waste

遮蔽:
- shield asset spawn/move
- actual pose error
- Embree update

移動・撤去:
- attached source follow
- disposal validation

統合テストで全 action 前後の measurement と map 変化を確認する。
```

## Prompt 7 — Physics robot execution

```text
Milestone 7 を実装してください。

実装対象:
- IsaacPhysicsRobotController
- measurement mobile robot navigation
- countermeasure mobile manipulator or base+arm composition
- end-effector planning
- grasp/release constraint
- shield pick-and-place
- obstacle pick/push
- contaminated object movement
- decon tool footprint ray/contact integration
- settle detection
- timeout、abort、recovery

要件:
- deterministic action と同じ API
- actual USD pose を radiation registry に反映
- physics execution failure を action result に記録
- manipulation animation だけでなく radiation state が更新される

最低 demo:
1. shield panel を把持し target pose に設置
2. contaminated box を移動
3. non-contaminated obstacle をどかす
4. decon tool で指定 patch を処理
```

## Prompt 8 — Source estimation

```text
Milestone 8 を実装してください。

実装対象:
- CandidateBasis: 3D grid、surface triangle graph
- measurement stacking
- Poisson negative log likelihood
- nonnegative MLE
- L1 proximal solver
- surface TV regularization
- coarse grid -> connected components -> continuous MLE refinement
- Fisher uncertainty
- optional bootstrap
- SourceEstimate serialization/visualization

禁止:
- TruthState 参照
- ground-truth source count を solver に与える

テスト:
- noiseless/noisy one source
- two sources
- hidden surface patch
- source height variation
- gradient finite difference
- uncertainty shape
```

## Prompt 9 — Predicted/observed residual と再推定

```text
Milestone 9 を実装してください。

実装対象:
- nominal action preview on BeliefState clone
- predicted verification measurement
- observed verification measurement
- raw/normalized residual
- DeconResidualHypothesis
- ShieldPoseErrorHypothesis
- HiddenSourceHypothesis
- GlobalGainBackgroundHypothesis
- SourceLocalizationErrorHypothesis
- likelihood/BIC selection
- belief update
- action-effect parameter update
- residual visualization and logs

fault injection tests:
- 30% decon residual
- 5 cm shield translation error
- hidden source
- detector gain bias
- mixed failure case
```

## Prompt 10 — Planner と closed-loop coordinator

```text
Milestone 10 を実装してください。

実装対象:
- action candidate generators
- measurement pose candidates
- decon region candidates
- shield placement candidates
- object move/remove candidates
- repair candidates
- navigation/manipulation/resource feasibility
- action objective
- expected information gain approximation
- OpenLoop、Greedy、Nearest、Random、Oracle、ClosedLoopResidual planners
- ClosedLoopCoordinator state machine
- pause/resume/stop
- termination conditions
- snapshot persistence

end-to-end demo:
MEASURE -> ESTIMATE -> PLAN -> PREDICT -> EXECUTE -> VERIFY -> DIAGNOSE -> UPDATE -> REPLAN

要件:
- Oracle 以外は TruthState を参照しない
- baseline と proposed の同一 seed batch runner
```

## Prompt 11 — ROS 2、MoveIt 2、Nav2 adapter

```text
Milestone 11 の ROS 2 部分を実装してください。

実装対象:
- radcounter_msgs
- RadiationMeasurement、SourceEstimate、CountermeasureStatus
- MeasureRadiation.action
- ExecuteCountermeasure.action
- GetDoseMap/EvaluateCountermeasure services
- Ros2RobotController
- standard tf/joint_states/cmd_vel/trajectory integration
- MoveIt 2 manipulation adapter
- Nav2 navigation adapter
- namespace/multi-robot support

要件:
- ROS 2 なしでも core と GUI が動く optional dependency
- Jazzy を primary target
- simulated time を使用
- timeout と QoS を設定ファイル化
```

## Prompt 12 — 実験自動化、性能、文書、リリース

```text
最終 milestone を実装してください。

実装対象:
- headless batch runner
- seed sweep
- baseline sweep
- output parquet/json/npz
- HTML report
- git/config/asset/hardware manifest
- benchmark suite
- regression baselines
- API docs
- scenario authoring guide
- troubleshooting
- one-command demo scripts

生成する実験:
1. analytic radiation validation
2. decon primitive
3. shield primitive
4. movable contaminated object
5. hidden source residual
6. closed-loop vs open-loop
7. resource-constrained multi-action

最終確認:
- Definition of Done 全項目を checklist 化
- 未実装項目を明示
- versioned release tag 用 changelog を作る
```

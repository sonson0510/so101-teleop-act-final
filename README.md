# SO-101 MuJoCo Teleoperation and ACT Imitation Learning

Physical AI応用編1の最終課題として、MuJoCo上のSO-101アームに対して、人がGUIで実演を収集し、ACT（Action Chunking with Transformers）で模倣学習する一連のパイプラインを実装しました。

## タスク

赤いブロックを把持し、緑色の目標地点へ運び、開放して退避します。

パイプラインは次の4段階です。

1. 固定カメラ画像を見ながらGUIでSO-101を操作
2. RGB画像、関節状態、関節指令を25 Hzで記録
3. LeRobot形式へ変換し、ACTを学習
4. 固定初期条件のMuJoCoで閉ループ推論・自動評価

## 結果

人手による成功実演21本のうち、18本を学習、3本を検証に分離しました。旧100エピソード実験で成功した設定を適用した10,000 stepモデルは、固定初期条件の600制御step評価で成功しました。

| 指標 | 結果 |
|---|---:|
| 最大持上げ | 0.06057 m |
| 最終目標XY誤差 | 0.01161 m |
| 退避量 | 0.11230 m |
| 最終物体速度 | 1.51e-7 m/s |
| グリッパー開放 | 成立 |
| 厳格成功判定 | success |

成功動画: [media/success_rollout_24s.mp4](media/success_rollout_24s.mp4)

詳細値: [results/success_metrics.json](results/success_metrics.json)

## 工夫した点

- 操作用カメラをほぼ真上からの視点にし、画面の上下左右とXY操作を一致させた
- 記録用カメラを固定し、学習画像の相関を維持した
- 記録開始・停止・成功・失敗の状態遷移に応じてGUIボタンを無効化した
- 関節可動域と1周期あたりの指令変化量を制限した
- 生データ、変換データ、モデル、評価出力を分離した
- 成功ラベルだけでなく、欠損・非有限値・オーバーラン・モデル指紋を検査した
- 学習損失だけでは成功扱いせず、閉ループ推論で把持・移動・開放・退避・静止を判定した

## 重要な知見

推論時の `n_action_steps` は動作成立に大きく影響しました。`5` では短い周期で再計画されて分布外へ逸脱しましたが、学習時のaction chunkと同じ `25` にすると把持・運搬が成立しました。

また、人手実演はscripted実演より長く、400制御step（16秒）では運搬途中でした。600制御step（24秒）で開放・退避・静止まで完了しました。制御周期は0.04秒（25 Hz）です。

## 構成

```text
scripts/                 GUI、変換、学習、評価、再生
configs/home.json        固定初期姿勢
scenes/                  pick-and-placeシーン
third_party/ICL-so101/   SO-101 MuJoCoモデル資産
tests/                   データ・制御契約テスト
results/                 成功評価指標
media/                   短い成功ロールアウト
```


## 環境

- Windows 11 + WSL2
- Python 3.12
- MuJoCo
- PySide6
- PyTorch + CUDA
- LeRobot / ACT

## GUIの起動例

```powershell
cd C:\path\to\repository
wsl.exe -e bash -lc "cd '/mnt/c/path/to/repository' && python scripts/so101_teleop.py"
```

詳細な依存関係は `requirements.txt` と `requirements-ml.txt` を参照してください。

## 学習例

```bash
python scripts/so101_teleop_train.py \
  /path/to/lerobot_train dataset_output \
  --repo-id local/so101_teleop \
  --steps 10000 --batch-size 32 \
  --chunk-size 25 --n-action-steps 25 \
  --learning-rate 1e-4 --backbone-learning-rate 1e-4 \
  --pretrained-backbone-weights none --save-every 2000
```

## 評価例

```bash
python scripts/so101_teleop_eval.py \
  /path/to/training_artifact \
  /path/to/eval_output \
  --steps 600 --n-action-steps 25 \
  --video-output rollout.mp4
```

## 制約

確認済みなのは、固定初期姿勢・固定物体位置・固定カメラにおけるMuJoCo上の1回の成功です。実機、異なる初期位置、異なるカメラ、複数seedでの成功率は未検証です。

## 第三者資産

SO-101のMuJoCoモデルとメッシュは `desmondzee/ICL-so101` に由来します。詳細は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。


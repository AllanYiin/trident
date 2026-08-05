# WP3：幾何同步與進階增強驗收

WP3 的幾何變換以 sample 為單位只抽樣一次，再依 `FieldSpec.kind` 同步處理影像、
mask、depth、bbox、keypoints、landmarks、polygon、densepose 與 optical flow。

## 完成項目

- `TransformContext` 的亂數由 seed、epoch、rank、sample id、transform 名稱及出現次序決定；
  worker id 不影響結果。
- `TransformRecord` 可 JSON 序列化；所有 `GeometryTransform` 可用 record 重播。
- `GeometryCompose.replay()` 可重播完整幾何鏈。
- 原有 `Resize`、`RandomCrop`、`RandomHorizontalFlip` 保留。
- 新增 `RandomAffine`、`RandomPerspective`，record 保存 3×3 matrix 與 inverse matrix。
- 新增 `SanitizeTargets`，支援 drop、clip、keep、error 策略，並依
  `FieldSpec.metadata.linked_to` 同步過濾 class 等 instance 欄位。
- 新增明確的多樣本 API：`MixUp`、`CutMix`、`Mosaic`、`CopyPaste`；來源池仍由有界
  `SamplePool` 管理，不把隱藏 cache 放進單樣本 transform。

## 驗收

```powershell
python -m pytest tests\test_data_pipeline_wp3_augmentation.py -q
```

測試涵蓋 affine record JSON round-trip/replay、image/box/keypoint 同步、worker-count
independence、linked target 過濾，以及四種 multi-sample augmentation 的輸出契約。

## 限制與明確選擇

- bbox 的 matrix transform 目前明確接受 `xyxy`；其他格式需先轉換，避免默默誤算。
- `Mosaic` 核心只負責等尺寸 HWC 影像拼接；偵測框的專案特定合併可放在其後的
  transform，或先統一成標準 instance schema。
- `CopyPaste` 使用 donor mask；instance id／類別的合併由 linked fields policy 處理。

# 使用 Google Colab 訓練 YOLOv5 口罩偵測模型

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_ID/YOUR_REPO/blob/main/yolov5_mask_training.ipynb)

> 組員：11325017\_梁駿勝, 11325006\_黃靖傑

> 程式碼在 `yolov5_mask_training.ipynb`，說明與實驗結果在這份 `README.md`。本文的數據都來自 notebook 保留的實際執行輸出。

## 快速開始：資料集怎麼下載

### 這份 notebook 可以直接執行

notebook 已經把「下載資料集、安裝 YOLOv5、訓練、偵測」全部寫成程式，**不需要手動下載或整理任何檔案**。只要完成下面兩件事，就可以從 Colab 選單「執行階段」→「全部執行」：

1. 把硬體切到 GPU：「執行階段」→「變更執行階段類型」→ **T4 GPU**。  
2. 執行到掛載 Google Drive 時，跳出授權視窗要按「允許」。

執行時，程式會自動完成以下事情：

| 自動完成的事情 | 對應步驟 |
| :---- | :---- |
| 用 `gdown` 依檔案 ID 下載資料集 zip，並解壓縮 | Step 4 |
| 讀取資料集裡的 `data.yaml`，取得類別名稱與順序，並產生訓練用的 `mask.yaml` | Step 4、5 |
| 從 GitHub 下載 YOLOv5 原始碼，並安裝需要的套件 | Step 3 |
| 訓練時自動下載預訓練權重 `yolov5s.pt` | Step 6 |
| 訓練結果直接存到 Google 雲端硬碟（資料夾不存在會自動建立） | Step 6 |
| 用訓練好的 `best.pt` 偵測驗證集圖片，並畫出結果 | Step 8 |

> notebook 裡的 `FILE_ID` 已經填入作者上傳的資料集，所以**不換資料集時，什麼都不用改**。這個下載依賴作者的 Google Drive 分享連結，如果連結被關閉或刪除，`gdown` 會失敗，這時請依照下方「方式二」換成自己的檔案。

### 資料集來源

本專案使用 Roboflow Universe 上的 **Face Mask Detection**（作者 `sparrowdhanashree`，version 1）：

- 資料集頁面：[https\://universe.roboflow.com/sparrowdhanashree/face-mask-detection-dvj6g/dataset/1](https://universe.roboflow.com/sparrowdhanashree/face-mask-detection-dvj6g/dataset/1)  
- 內容：共 2,157 張圖片，其中訓練集 1,824 張、驗證集 239 張、測試集 94 張。  
- 類別：`mask_weared_incorrect`、`with_mask`、`without_mask`。

### 方式一：直接使用 notebook 內建的連結（最快）

不用做任何事，直接執行即可。

### 方式二：自己下載資料集，上傳到 Google Drive

1. 登入 Roboflow（免費註冊），開啟上面的資料集頁面。![圖 1　開啟資料集頁面](images/dataset_1_open_page.png)

&nbsp;

2. 點選下方YOLOv5的選項

![][image1]

3. 點選頁面上的「Download Dataset」（下載資料集）。

![][image2]![圖 2　點選 Download Dataset](images/dataset_2_download.png)

4. 格式選 **YOLOv5 PyTorch**，並選擇「Download zip to computer」（下載 zip 到電腦），下載後會得到一個 zip 檔。

![圖 3　選擇 YOLOv5 格式](images/dataset_3_format.png)![][image3]

5. 把 zip 檔上傳到自己的 Google 雲端硬碟，對檔案按右鍵 →「共用」→ 一般存取權改成「**知道連結的任何人**」。

   ![圖 4　設定共用權限](images/dataset_4_drive_share.png)

6. 複製分享連結，取出檔案 ID。連結格式是 `https://drive.google.com/file/d/這一段就是ID/view`，把這段 ID 貼到 notebook 的 `FILE_ID`。

   ![圖 5　從連結取得檔案 ID](images/dataset_5_file_id.png)

```py
FILE_ID = "貼上你的檔案ID"
```

> Roboflow 的按鈕名稱與位置可能隨版本調整，以實際畫面為準。重點是匯出格式要選 **YOLOv5 PyTorch**，zip 裡要有 `data.yaml`、`train/`、`valid/`。

## 一、專題簡介

YOLOv5 是常用的物件偵測模型，能在一張圖片中同時找出多個物件的**位置**與**類別**。本專題用它做**口罩偵測**：輸入人臉照片，模型框出每張臉，並判斷屬於以下哪一類：

| 類別 | 意義 |
| :---- | :---- |
| `with_mask` | 正確戴口罩 |
| `without_mask` | 沒戴口罩 |
| `mask_weared_incorrect` | 口罩戴法不正確 |

訓練模型需要 GPU，所以在 Google Colab 的免費 T4 GPU 上進行，資料集與訓練結果都放在 Google 雲端硬碟。流程是：**下載資料集 → 呼叫 YOLOv5 訓練 → 查看結果 → 偵測圖片**。

## 二、實驗環境

以下資訊取自 notebook 的執行輸出。

| 項目 | 內容 |
| :---- | :---- |
| 平台 | Google Colab |
| GPU | Tesla T4（約 15 GB 顯存） |
| Python / PyTorch | Python 3.13 / PyTorch 2.11（CUDA 13.0） |
| 模型 | [ultralytics/yolov5](https://github.com/ultralytics/yolov5)，預訓練權重 `yolov5s.pt` |
| 模型規模 | 68 層、約 702 萬個參數、15.8 GFLOPs（融合後） |

## 三、資料集

- **格式**：YOLO 格式，從 Roboflow 匯出（YOLOv5 PyTorch），內容有 `data.yaml`、`train/`、`valid/`、`test/`。  
- **類別**：3 類，順序為 `mask_weared_incorrect`、`with_mask`、`without_mask`（取自資料集的 `data.yaml`）。  
- **訓練集**：1,824 張圖片（與訓練時每輪 114 個批次 × 每批 16 張相符）。  
- **驗證集**：239 張圖片、1,311 個標註框。  
- **類別分布不均**：驗證集中 `with_mask` 有 991 個，`without_mask` 277 個，`mask_weared_incorrect` 只有 43 個。  
- **來源**：Roboflow Universe 的 Face Mask Detection（`sparrowdhanashree`，version 1）。圖片檔名以 `maksssksksss` 開頭，應與 Kaggle 的 Face Mask Detection 資料集同源。

資料集的下載與上傳方式見最前面的[快速開始](#快速開始資料集怎麼下載)。

## 四、操作步驟

開啟方式：上傳到 GitHub 後點最上方的 **Open In Colab** 按鈕，或在 Colab 選「檔案」→「上傳筆記本」。先把硬體切到 GPU（「執行階段」→「變更執行階段類型」→ T4 GPU），再由上往下執行每個儲存格。下表的儲存格編號是 notebook 由上往下的順序。

### Step 1：確認 GPU（儲存格 1）

`!nvidia-smi` 列出 Colab 分配到的顯示卡。實際輸出為 `Tesla T4`，顯存 0 / 15360 MiB，表示 GPU 可用且尚未被占用。

### Step 2：掛載 Google Drive 並設定路徑（儲存格 2、3）

- `drive.mount` 讓 Colab 能讀寫雲端硬碟，執行時會跳出授權視窗。  
- `FILE_ID`：資料集 zip 的檔案 ID。  
- `SAVE_DIR`：訓練結果存放位置。結果直接存在雲端硬碟，Colab 關閉後也不會消失。

### Step 3：下載 YOLOv5 並安裝套件（儲存格 4、5）

`git clone --depth 1` 只抓最新版程式碼，再用 `%cd yolov5` 進入資料夾，並以 `pip install -r requirements.txt` 安裝需要的套件。

### Step 4：下載並解壓縮資料集（儲存格 6～9）

- 儲存格 6：匯入 `Path`、`yaml`、`zipfile`。  
- 儲存格 7：`gdown` 依 `FILE_ID` 把 zip 下載到 `/content/mask_dataset.zip`。實際輸出顯示檔案約 54.7 MB。  
- 儲存格 8：解壓縮到 `/content/dataset`。  
- 儲存格 9：找出資料集內的 `data.yaml`，並讀出類別名稱。

### Step 5：產生訓練設定檔（儲存格 10、11）

YOLOv5 靠設定檔知道資料集位置與類別。儲存格 10 寫出 `data/mask.yaml`，儲存格 11 印出內容：

```py
{'path': '/content/dataset', 'train': 'train/images', 'val': 'valid/images',
 'names': ['mask_weared_incorrect', 'with_mask', 'without_mask']}
```

### Step 6：訓練（儲存格 12）

```sh
python train.py --img 640 --batch-size 16 --epochs 50 --data data/mask.yaml \
    --weights yolov5s.pt --project <SAVE_DIR> --name mask --exist-ok
```

&nbsp;

| 參數 | 意義 |
| :---- | :---- |
| `--img` | 圖片縮放後的邊長（像素） |
| `--batch-size` | 每次更新參數用幾張圖；出現 `CUDA out of memory` 就改小 |
| `--epochs` | 完整看過資料集幾輪 |
| `--data` | Step 5 產生的設定檔 |
| `--weights` | 預訓練權重。從這個模型微調，比從零訓練快得多；其他可用的模型見下方「其他 YOLOv5 模型」 |
| `--project`、`--name` | 結果存放的位置與資料夾名稱 |
| `--exist-ok` | 同名資料夾已存在時直接沿用 |

#### 其他 YOLOv5 模型

YOLOv5 依「深度與寬度」分成 n、s、m、l、x 五種大小，越大越準，但也越慢、越吃顯存。要更換時，只要把 `--weights` 改成對應的檔名，權重會自動下載，其他參數不用動。

**標準模型（輸入尺寸 640）**，適合一般用途，本專案使用的是 `yolov5s`：

| 模型 | `--weights` | mAP 50-95 | mAP 50 | CPU 速度 (ms) | V100 速度 (ms) | 參數量 (M) | FLOPs (B) | 適合情境 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| YOLOv5n | `yolov5n.pt` | 28.0 | 45.7 | 45 | 6.3 | 1.9 | 4.5 | 手機、嵌入式裝置，追求速度 |
| **YOLOv5s** | `yolov5s.pt` | 37.4 | 56.8 | 98 | 6.4 | 7.2 | 16.5 | **入門首選，本專案使用** |
| YOLOv5m | `yolov5m.pt` | 45.4 | 64.1 | 224 | 8.2 | 21.2 | 49.0 | 精度與速度折衷 |
| YOLOv5l | `yolov5l.pt` | 49.0 | 67.3 | 430 | 10.1 | 46.5 | 109.1 | 追求精度 |
| YOLOv5x | `yolov5x.pt` | 50.7 | 68.9 | 766 | 12.1 | 86.7 | 205.7 | 最高精度，需要較大顯存 |

**P6 大尺寸模型（輸入尺寸 1280）**，名稱結尾多一個 `6`，適合圖片中的物件很小的情況：

| 模型 | `--weights` | mAP 50-95 | mAP 50 | CPU 速度 (ms) | V100 速度 (ms) | 參數量 (M) | FLOPs (B) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| YOLOv5n6 | `yolov5n6.pt` | 36.0 | 54.4 | 153 | 8.1 | 3.2 | 4.6 |
| YOLOv5s6 | `yolov5s6.pt` | 44.8 | 63.7 | 385 | 8.2 | 12.6 | 16.8 |
| YOLOv5m6 | `yolov5m6.pt` | 51.3 | 69.3 | 887 | 11.1 | 35.7 | 50.0 |
| YOLOv5l6 | `yolov5l6.pt` | 53.7 | 71.3 | 1784 | 15.8 | 76.8 | 111.4 |
| YOLOv5x6 | `yolov5x6.pt` | 55.0 | 72.7 | 3136 | 26.2 | 140.7 | 209.8 |

> 以上數據引用自 YOLOv5 官方 README，是在 COCO 資料集（80 類）上測得的，**不代表**在本專案口罩資料集上的表現。mAP 越高越準；速度與參數量越小越快、越省資源。

換模型時要注意：

- **標準模型**：訓練與偵測的 `--img` 維持 `640`。  
- **P6 模型**：`--img` 要改成 `1280`，而且訓練與偵測要用同一個值。圖片變大，顯存需求大增，在 Colab 的 T4 上通常要把 `--batch-size` 調小（例如 4 或 8），訓練時間也會變長。  
- **較大的模型**（`m`、`l`、`x`）：同樣可能需要調小 `--batch-size`，否則會出現 `CUDA out of memory`。  
- **比較不同模型**：用不同的 `--name`（例如 `mask_m`、`mask_n`），結果才不會互相覆蓋。  
- **取捨**：不是越大越好。資料集只有一千多張圖時，較大的模型不一定有明顯的提升，卻要花更多時間，建議先用 `yolov5s` 當基準，再嘗試 `yolov5m` 或 `yolov5n` 比較。

換成 `yolov5m` 的訓練指令範例：

```sh
python train.py --img 640 --batch-size 16 --epochs 50 --data data/mask.yaml \
    --weights yolov5m.pt --project <SAVE_DIR> --name mask_m --exist-ok
```

實際執行：50 輪共花約 **0.62 小時（約 37 分鐘）**，每輪約 40 秒。訓練會自動下載 `yolov5s.pt`，並把輸出層改成 3 類（`nc=3`）。

### Step 7：查看訓練結果（儲存格 13、14）

顯示 `results.png` 與 `confusion_matrix.png`，結果分析見[第六節](#六實驗結果)。最重要的檔案是 `weights/best.pt`，它是驗證集表現最好的權重，約 14.5 MB。

### Step 8：偵測（儲存格 15、16、17）

```sh
python detect.py --weights <SAVE_DIR>/mask/weights/best.pt \
    --source <資料集>/valid/images --img 640 --conf-thres 0.4 \
    --name mask_detect --exist-ok
```

&nbsp;

| 參數 | 意義 |
| :---- | :---- |
| `--weights` | 訓練好的模型 |
| `--source` | 要偵測的圖片或資料夾；想測自己的照片，改成你的資料夾 |
| `--conf-thres` | 信心度門檻，低於這個值的框會被丟掉。框太多就調高，漏抓就調低 |

偵測結果存在 `runs/detect/mask_detect/`。儲存格 16、17 用 matplotlib 把前 6 張排成 2×3 的圖。

<!-- 跑完後把偵測結果截圖存成 images/detect_result.png，取消下一行的註解 -->

&nbsp;

<!-- ![偵測結果](images/detect_result.png) -->

&nbsp;

## 六、實驗結果

### 整體與各類別表現

以 `best.pt` 在驗證集（239 張、1,311 個框）上評估：

| 類別 | 標註框數 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **全部** | 1,311 | 0.872 | 0.747 | 0.821 | 0.515 |
| `with_mask` | 991 | 0.908 | 0.911 | 0.936 | 0.628 |
| `without_mask` | 277 | 0.875 | 0.726 | 0.813 | 0.451 |
| `mask_weared_incorrect` | 43 | 0.834 | 0.605 | 0.716 | 0.466 |

&nbsp;

| 指標 | 意義 |
| :---- | :---- |
| Precision | 模型框出來的東西，有多少是對的 |
| Recall | 實際存在的物件，有多少被找出來 |
| mAP@0.5 | 框的重疊度（IoU）達 0.5 時的平均精確度 |
| mAP@0.5:0.95 | IoU 門檻從 0.5 到 0.95 的平均，標準更嚴格 |

### 訓練曲線

![訓練曲線](images/results.png)

- 訓練的三種損失（`box_loss`、`obj_loss`、`cls_loss`）從第 0 輪到第 50 輪持續下降，表示模型有在學習。  
- 驗證集的 `mAP@0.5` 在前 20 輪快速上升到約 0.75，之後緩慢提高並趨於平穩，約在第 40 輪達到 0.82 左右。  
- Precision 在前 20 輪波動很大，之後逐漸穩定。Recall 則持續緩慢上升。  
- 驗證集的 `obj_loss` 在第 10 輪之後略微回升，同時訓練損失仍在下降，可能是開始有些過擬合的跡象，再多訓練的收穫有限。

### 混淆矩陣

![混淆矩陣](images/confusion_matrix.png)

此圖按「實際類別」（橫軸）為 100% 來計算：

| 實際類別 | 判對 | 主要錯誤 |
| :---- | :---- | :---- |
| `with_mask` | 92% | 約 7% 沒被偵測到 |
| `without_mask` | 75% | 約 21% 沒被偵測到，約 3% 判成 `with_mask` |
| `mask_weared_incorrect` | 56% | 約 33% 被判成 `with_mask`，約 7% 沒被偵測到 |

### 結果分析

1. **`with_mask` 表現最好**（mAP@0.5 為 0.936）：它在資料集中占絕大多數（991 / 1,311），模型見過最多樣本。  
2. **`mask_weared_incorrect` 最弱**（Recall 僅 0.605）：驗證集只有 43 個樣本，訓練集中的樣本也可能不多；而且它的外觀與 `with_mask` 很接近，所以有三分之一被誤判成 `with_mask`。這是類別不平衡造成的典型問題。  
3. **`without_mask` 的主要問題是漏抓**：約五分之一沒被偵測出來，可能是人臉太小或被遮擋。  
4. **偵測速度**：在 T4 上，每張圖的推論約 8.3 毫秒，加上前處理 0.5 毫秒與 NMS 1.4 毫秒，共約 10 毫秒，足以做即時偵測。

### 注意事項

偵測用的 `valid/images` 同時是挑選 `best.pt` 的依據，所以結果略微偏樂觀。資料集另有 `test/` 資料夾，要得到更公正的評估，可以把 `--source` 改成 `test/images`，或用 `val.py --task test` 評估。

## 七、執行時遇到的狀況

| 現象 | 說明 |
| :---- | :---- |
| 訓練開始時停住約 30 秒，並出現 W\&B 選項 | YOLOv5 詢問是否使用 Weights & Biases 記錄。30 秒沒回應會自動停用，不影響訓練；在訓練前執行 `%env WANDB_MODE=disabled` 可以跳過 |
| 出現 `Could not find cuda drivers on your machine, GPU will not be used` | 這是 TensorFlow 的訊息，與 YOLOv5 無關。同一份輸出的下一段是 `CUDA:0 (Tesla T4...)`，表示訓練確實用了 GPU |
| 出現 `Drive already mounted` | 代表雲端硬碟已經掛載過，不是錯誤 |
| 重複執行 `%cd yolov5` 後路徑變成 `/content/yolov5/yolov5` | `%cd` 是相對路徑，重複執行會一直往下一層。若已經在 `yolov5` 資料夾內，不要再執行 |
| `gdown` 下載失敗或 Access denied | 檔案的共用權限不是「知道連結的任何人」，或 `FILE_ID` 貼錯 |
| `CUDA out of memory` | 把 `--batch-size` 改成 8 或 4 |
| Colab 斷線 | 訓練結果存在雲端硬碟，不會遺失，可直接用 `best.pt` 偵測 |

## 八、結論與改進方向

本專題用 Colab 的 T4 GPU，以 `yolov5s.pt` 微調 50 輪（約 37 分鐘），在驗證集上得到 **mAP@0.5 \= 0.821**、Precision \= 0.872、Recall \= 0.747。整個流程都是呼叫 YOLOv5 現成的 `train.py` 與 `detect.py`，不需要自己寫模型。

可以再改進的方向：

- **補強少數類別的資料**：`mask_weared_incorrect` 樣本太少，增加這類圖片最有幫助。  
- **換較大的模型**：例如 `yolov5m.pt`（比較表見[其他 YOLOv5 模型](#其他-yolov5-模型)），通常能提高精度，但訓練較慢、顯存需求較高。  
- **用測試集評估**：得到更公正的成績。

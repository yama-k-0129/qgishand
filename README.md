# qgishand
# **HAND（Height Above Nearest Drainage）を算出するプログラム**

## **入力データ(dataというフォルダを作成してその中に格納)**
- **流向データ（tif）**
- **累積流量を抽出したデータ（tif）**
- **標高データ（tif）**

## **出力データ(outputというフォルダを作成してその中に格納)**
- **handデータ（tif）**
---

## **算出する手順**

### **Step 1: 依存するライブラリをダウングレードする**

まずは、NumPyなどの依存するライブラリをダウングレードします。具体的にはNumPyのバージョンを1.24以下に設定します。

#### **コード**

```bash
conda install gdal numpy numba rasterio matplotlib
conda install 'numpy<=1.24'
```

---

### **Step 2: ダウングレードした環境でプログラムを実行する**

ダウングレードが完了したら、`.ipynb`か`.py`ファイルを実行します。どちらの形式でも好きな方を使用してください。

---

### Dijkstraのアルゴリズムの詳細

このアルゴリズムにおけるDijkstraのアルゴリズムは、優先度キュー（`pq`）を使用して最短距離を効率的に求めます。具体的な手順は以下の通りです。

1. **優先度キューの取り出し(セルの計算する優先順番みたいなもの)**:
    - 優先度キュー（`pq`）から最も「フロー距離」が短いセル（`ci`, `cj`）とその距離（`current_distance`）を取り出す。

2. **近傍セルの更新**:
    - 取り出したセル（`ci`, `cj`）に隣接するセル（`ni`, `nj`）に対して以下を行う。
        1. 新しい距離（`new_distance`）を計算。この際、標高（`next_elevation`と`current_elevation`）も考慮する。
        2. 計算した新しい距離が現在の「フロー距離」（`flow_distance[ni, nj]`）よりも短い場合は、
            - `flow_distance[ni, nj]`を`new_distance`で更新。
            - 更新したセル（`ni`, `nj`）と新しい距離（`new_distance`）を優先度キューに追加。


この手順を優先度キューが空になるまで繰り返します。

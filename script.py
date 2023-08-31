from osgeo import gdal
import numpy as np
from numba import jit, float64, int64
import heapq
import rasterio
import matplotlib.pyplot as plt

@jit
def calculate_flow_distance(flow_direction, dem, river_100):
    rows, cols = flow_direction.shape
    flow_distance = np.full((rows, cols), np.inf, dtype=np.float64)

    pq = []

    for i in range(rows):
        for j in range(cols):
            if river_100[i, j] == 1:
                flow_distance[i, j] = 0
                heapq.heappush(pq, (0, i, j))

    # Dijkstra's algorithm
    directions = [
        (-2, -1), (-2, 0), (-2, 1), (-2, 2),
        (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
        (0, -2),           (0, 0),           (0, 2),
        (1, -2),  (1, -1),  (1, 0),  (1, 1),  (1, 2),
        (2, -1),  (2, 0),   (2, 1),  (2, 2)
    ]

    
    while len(pq) > 0:
        current_distance, ci, cj = heapq.heappop(pq)

        for dx, dy in directions:
            ni, nj = ci + dx, cj + dy

            if 0 <= ni < rows and 0 <= nj < cols:
                next_elevation = dem[ni, nj]
                current_elevation = dem[ci, cj]
                
                slope_distance = np.sqrt(dx**2 + dy**2 + (next_elevation - current_elevation)**2)
                new_distance = current_distance + slope_distance

                if new_distance < flow_distance[ni, nj]:
                    flow_distance[ni, nj] = new_distance
                    heapq.heappush(pq, (new_distance, ni, nj))

    return flow_distance

# Step 1: Read raster data
flow_ds = gdal.Open("data/waterdirection.tif")#流向データの場所（パス）を指定
dem_ds = gdal.Open("data/umetatedem.tif")#標高データの場所(パス)を指定
river_ds = gdal.Open("data/suikeisize.tif")#水系サイズを抽出したファイルの場所(パス)を指定

flow_direction = np.array(flow_ds.GetRasterBand(1).ReadAsArray(), dtype=np.float64)
dem = np.array(dem_ds.GetRasterBand(1).ReadAsArray(), dtype=np.float64)
river_100 = np.array(river_ds.GetRasterBand(1).ReadAsArray(), dtype=np.int64)

# Step 2: Calculate flow distance
flow_distance = calculate_flow_distance(flow_direction, dem, river_100)

# Step 3: Save result
output_ds = gdal.GetDriverByName('GTiff').Create("output/flow_distance.tif", flow_direction.shape[1], flow_direction.shape[0], 1, gdal.GDT_Float32)
output_ds.GetRasterBand(1).WriteArray(flow_distance)
# 座標系と地理座標変換を入力ファイルからコピー
output_ds.SetProjection(flow_ds.GetProjection())
output_ds.SetGeoTransform(flow_ds.GetGeoTransform())
output_ds = None  # Close file

# TIFFファイルを読み込む
with rasterio.open("output/flow_distance.tif") as src:
    # バンド1のデータを読み込む
    image = src.read(1)

# データを図示する
im = plt.imshow(image, cmap="gray", vmax=100)  # vmaxでカラーバーの上限を100に設定
cbar = plt.colorbar(im, label="Pixel value")
plt.title("Your TIFF File")
plt.xlabel("Column #")
plt.ylabel("Row #")
plt.show()
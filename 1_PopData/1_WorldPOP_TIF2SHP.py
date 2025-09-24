import os
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

# 输入和输出文件夹
in_folder = r"D:\SCUT\PAPER\EVCS_Part1\EVCS_data1\2015-2030年我国100米分辨率人口总数栅格数据\研究区域的城市\province_tif"
out_folder = r"D:\SCUT\PAPER\EVCS_Part1\EVCS_data1\2015-2030年我国100米分辨率人口总数栅格数据\研究区域的城市\province_shp"

# 确保输出文件夹存在
os.makedirs(out_folder, exist_ok=True)

# 获取所有 tif 文件，并按文件大小排序（升序）
tif_files = [f for f in os.listdir(in_folder) if f.lower().endswith(".tif")]
tif_files = sorted(tif_files, key=lambda f: os.path.getsize(os.path.join(in_folder, f)))

total_files = len(tif_files)
print(f"总共找到 {total_files} 个文件需要处理（已按大小排序）。\n")

# 遍历每一个 tif 文件
for idx, tif_file in enumerate(tif_files, start=1):
    in_path = os.path.join(in_folder, tif_file)
    out_name = os.path.splitext(tif_file)[0] + ".shp"
    out_path = os.path.join(out_folder, out_name)

    print(f"({idx}/{total_files}) 正在处理: {tif_file}  （大小: {os.path.getsize(in_path)/1024/1024:.2f} MB）")

    # 打开栅格文件
    with rasterio.open(in_path) as src:
        band1 = src.read(1)
        transform = src.transform
        rows, cols = band1.shape
        data = []

        # tqdm 显示单文件进度条
        for row in tqdm(range(rows), desc=f"{tif_file} 转换进度", leave=False):
            for col in range(cols):
                value = band1[row, col]
                if value != src.nodata:  # 跳过 NoData
                    x, y = transform * (col + 0.5, row + 0.5)
                    data.append({"geometry": Point(x, y), "value": float(value)})

    # 转换成 GeoDataFrame
    gdf = gpd.GeoDataFrame(data, crs=src.crs)

    # 保存为 Shapefile
    gdf.to_file(out_path, driver="ESRI Shapefile", encoding="utf-8")

    print(f"✅ 完成 {tif_file} → {out_name}\n")

print("🎉 全部文件处理完成！")

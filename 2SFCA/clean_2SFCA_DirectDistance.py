import os
import time
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree

def ensure_projected(gdf: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    """投影到目标 CRS。"""
    if gdf.crs is None:
        raise ValueError("输入图层没有 CRS。")
    if gdf.crs == target_crs:
        return gdf
    return gdf.to_crs(target_crs)

def g2sfca_euclidean(pop_gdf, ev_gdf,
                     pop_field='vaule',
                     supply_field='加权数量',
                     radius=1000.0,
                     sigma=None,
                     eps=1e-9,
                     verbose=True):
    if sigma is None:
        sigma = radius / 3.0

    pop_coords = np.vstack([geom.coords[0] for geom in pop_gdf.geometry.to_list()])
    ev_coords = np.vstack([geom.coords[0] for geom in ev_gdf.geometry.to_list()])

    pop_tree = cKDTree(pop_coords)
    ev_tree = cKDTree(ev_coords)

    pop_weights = pop_gdf[pop_field].astype(float).to_numpy()
    ev_supply = ev_gdf[supply_field].astype(float).to_numpy()

    n_pop, n_ev = len(pop_coords), len(ev_coords)
    if verbose:
        print(f"人口点: {n_pop}, EVCS点: {n_ev}")

    # Step1
    R = np.zeros(n_ev, dtype=float)
    ev_to_pop = ev_tree.query_ball_tree(pop_tree, r=radius)
    for j in range(n_ev):
        pop_idxs = ev_to_pop[j]
        if not pop_idxs:
            continue
        dists = np.linalg.norm(pop_coords[pop_idxs] - ev_coords[j], axis=1)
        weights = np.exp(-(dists**2) / (2.0 * sigma**2))
        denom = np.sum(pop_weights[pop_idxs] * weights)
        if denom >= eps:
            R[j] = ev_supply[j] / denom
        if verbose and j % 1000 == 0 and j > 0:
            print(f"Step1: 已处理 {j}/{n_ev} 个 EVCS")

    # Step2
    access_scores = np.zeros(n_pop, dtype=float)
    pop_to_ev = pop_tree.query_ball_tree(ev_tree, r=radius)
    for i in range(n_pop):
        ev_idxs = pop_to_ev[i]
        if not ev_idxs:
            continue
        dists = np.linalg.norm(ev_coords[ev_idxs] - pop_coords[i], axis=1)
        weights = np.exp(-(dists**2) / (2.0 * sigma**2))
        access_scores[i] = np.sum(R[ev_idxs] * weights)
        if verbose and i % 10000 == 0 and i > 0:
            print(f"Step2: 已处理 {i}/{n_pop} 个人口点")

    pop_out = pop_gdf.copy()
    pop_out['access_g2sfca'] = access_scores

    if verbose:
        print("可达性分布：")
        print("  均值:", np.mean(access_scores))
        print("  最大:", np.max(access_scores))
        print("  零值比例:", np.mean(access_scores==0))

    return pop_out, R

if __name__ == "__main__":
    # ==== 配置 ====
    pop_shp = r"D:\SCUT\PAPER\EVCS_Part1\EVCS_data1\2015-2030年我国100米分辨率人口总数栅格数据\研究区域的城市\demo_shp\[Shenzhen]chn_pop_2022_CN_100m_R2025A_v1.shp"
    ev_shp  = r"D:\SCUT\PAPER\EVCS_Part2\G2SFCA\EVCS_Data\Stars_EVCS_Points.shp"
    pop_field = 'value'
    supply_field = 'TotalC'
    radius = 1000.0
    sigma = None
    target_crs = "EPSG:4544"  # 全国统一投影，先用 GK 105E
    out_shp = None
    # ==============

    t0 = time.time()
    pop_gdf = gpd.read_file(pop_shp)
    ev_gdf  = gpd.read_file(ev_shp)

    pop_gdf = ensure_projected(pop_gdf, target_crs)
    ev_gdf  = ensure_projected(ev_gdf, target_crs)

    pop_with_access, R_vals = g2sfca_euclidean(
        pop_gdf, ev_gdf,
        pop_field=pop_field,
        supply_field=supply_field,
        radius=radius,
        sigma=sigma,
        verbose=True
    )

    if out_shp is None:
        base, ext = os.path.splitext(pop_shp)
        out_shp = base + "_g2sfca.shp"
    pop_with_access.to_file(out_shp)

    print(f"✅ 完成！结果写入: {out_shp}")
    print("总耗时: %.2f 秒" % (time.time()-t0))

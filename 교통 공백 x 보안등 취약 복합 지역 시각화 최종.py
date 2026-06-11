"""
서울시 교통 공백 × 보안등 취약 복합 위험 지역 시각화
=====================================================
N버스 정류장 1km 초과(교통 공백) + 보안등 없는 격자가
동시에 해당하는 복합 취약 격자를 빨간색으로 표시합니다.

필요 라이브러리:
    pip install geopandas shapely matplotlib pandas numpy pyproj scipy openpyxl

입력 파일 (스크립트와 같은 폴더):
    - admstr_zone_lgldong_bndry_24.shp  (+ .dbf/.shx/.prj)
    - 서울시버스노선별정류소정보_20260506_.xlsx
    - 서울시_보안등_최종3.csv

출력 (output/ 폴더 자동 생성):
    - seoul_combined_vuln.png       시각화 이미지
    - seoul_combined_vuln.csv       격자 전체 데이터
    - seoul_combined_vuln_only.csv  복합 취약 격자만
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from shapely.geometry import Point, box
from scipy.spatial import cKDTree
from pathlib import Path
import platform
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# 한글 폰트 자동 설정
# ──────────────────────────────────────────────────────────────
def set_korean_font():
    system = platform.system()
    candidates = ['AppleGothic', 'Apple SD Gothic Neo'] if system == 'Darwin' else \
                 ['Malgun Gothic'] if system == 'Windows' else ['NanumGothic']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            return
    for f in fm.fontManager.ttflist:
        if any(k in f.name for k in ['Gothic', 'Nanum', 'Malgun', 'Apple']):
            plt.rcParams['font.family'] = f.name
            plt.rcParams['axes.unicode_minus'] = False
            return

set_korean_font()

# ──────────────────────────────────────────────────────────────
# 설정값
# ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
CRS_KOREA = "EPSG:5179"
GRID_SIZE = 100
VULN_DIST = 1000
SEOUL_LAT = (37.4, 37.72)
SEOUL_LON = (126.75, 127.25)


# ──────────────────────────────────────────────────────────────
# STEP 1. 서울시 행정경계 로드
# ──────────────────────────────────────────────────────────────
def load_seoul_boundary():
    shp_path = BASE_DIR / 'admstr_zone_lgldong_bndry_24.shp'
    if not shp_path.exists():
        raise FileNotFoundError(f"행정경계 파일 없음: {shp_path}")
    gdf = gpd.read_file(str(shp_path), encoding='cp949')
    seoul = gdf[gdf['EMD_CD'].astype(str).str.startswith('11')].to_crs(CRS_KOREA)
    boundary = seoul.union_all()
    print(f"[1] 서울시 행정경계: {boundary.area/1_000_000:.2f}km²")
    return seoul, boundary


# ──────────────────────────────────────────────────────────────
# STEP 2. 100m×100m 격자 생성
# ──────────────────────────────────────────────────────────────
def create_grid(boundary):
    minx, miny, maxx, maxy = boundary.bounds
    xs = np.arange(int(minx), int(maxx) + GRID_SIZE, GRID_SIZE)
    ys = np.arange(int(miny), int(maxy) + GRID_SIZE, GRID_SIZE)
    polygons = [box(x, y, x + GRID_SIZE, y + GRID_SIZE) for y in ys for x in xs]
    grid = gpd.GeoDataFrame({'grid_id': range(len(polygons))},
                             geometry=polygons, crs=CRS_KOREA)
    grid = grid[grid.geometry.intersects(boundary)].copy().reset_index(drop=True)
    grid['grid_id']    = range(len(grid))
    grid['centroid_x'] = grid.geometry.centroid.x
    grid['centroid_y'] = grid.geometry.centroid.y
    print(f"[2] 격자 생성: {len(grid):,}개 ({len(grid)*0.01:.1f}km²)")
    return grid


# ──────────────────────────────────────────────────────────────
# STEP 3. N버스 교통 공백 판정
# ──────────────────────────────────────────────────────────────
def map_nbus(grid):
    xlsx_path = BASE_DIR / '서울시버스노선별정류소정보_20260506_.xlsx'
    df = pd.read_excel(str(xlsx_path))
    n_stops = df[df['노선명'].astype(str).str.match(r'^N\d+')].copy()
    n_stops['X좌표'] = pd.to_numeric(n_stops['X좌표'], errors='coerce')
    n_stops['Y좌표'] = pd.to_numeric(n_stops['Y좌표'], errors='coerce')
    n_stops = n_stops[
        n_stops['X좌표'].between(*SEOUL_LON) &
        n_stops['Y좌표'].between(*SEOUL_LAT)
    ].drop_duplicates('ARS_ID').reset_index(drop=True)

    n_gdf = gpd.GeoDataFrame(
        n_stops,
        geometry=[Point(x, y) for x, y in zip(n_stops['X좌표'], n_stops['Y좌표'])],
        crs="EPSG:4326"
    ).to_crs(CRS_KOREA)

    stop_xy = np.array([[g.x, g.y] for g in n_gdf.geometry])
    grid_xy = np.column_stack([grid['centroid_x'], grid['centroid_y']])
    dist, idx = cKDTree(stop_xy).query(grid_xy, k=1)
    grid['nearest_dist_m']    = dist.round(1)
    grid['is_transport_vuln'] = grid['nearest_dist_m'] > VULN_DIST

    n_t = grid['is_transport_vuln'].sum()
    print(f"[3] 교통 공백 (1km 초과): {n_t:,}개 ({n_t/len(grid)*100:.1f}%)")
    return grid


# ──────────────────────────────────────────────────────────────
# STEP 4. 보안등 격자 매핑
# ──────────────────────────────────────────────────────────────
def map_lights(grid):
    csv_path = BASE_DIR / '서울시_보안등_최종3.csv'
    df = pd.read_csv(str(csv_path), encoding='utf-8')
    df['위도']    = pd.to_numeric(df['위도'],    errors='coerce')
    df['경도']    = pd.to_numeric(df['경도'],    errors='coerce')
    df['설치개수'] = pd.to_numeric(df['설치개수'], errors='coerce').fillna(1).astype(int)
    df = df.dropna(subset=['위도', '경도'])
    df = df[df['위도'].between(*SEOUL_LAT) & df['경도'].between(*SEOUL_LON)]

    light_gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(x, y) for x, y in zip(df['경도'], df['위도'])],
        crs="EPSG:4326"
    ).to_crs(CRS_KOREA)

    joined = gpd.sjoin(light_gdf, grid[['grid_id', 'geometry']],
                       how='left', predicate='within')
    agg = joined.groupby('grid_id')['설치개수'].sum().reset_index(name='light_count')
    grid = grid.merge(agg, on='grid_id', how='left')
    grid['light_count']   = grid['light_count'].fillna(0).astype(int)
    grid['is_light_vuln'] = grid['light_count'] == 0

    n_l = grid['is_light_vuln'].sum()
    print(f"[4] 보안등 없는 격자: {n_l:,}개 ({n_l/len(grid)*100:.1f}%)")
    return grid


# ──────────────────────────────────────────────────────────────
# STEP 5. 복합 취약 격자 판정
# ──────────────────────────────────────────────────────────────
def flag_combined(grid):
    grid['is_combined_vuln'] = grid['is_transport_vuln'] & grid['is_light_vuln']
    n_c   = grid['is_combined_vuln'].sum()
    n_tot = len(grid)
    print(f"[5] 복합 취약 (교통+보안등): {n_c:,}개 ({n_c/n_tot*100:.1f}%)")
    return grid


# ──────────────────────────────────────────────────────────────
# STEP 6. 시각화 (N버스 정류장 포인트 없음)
# ──────────────────────────────────────────────────────────────
def visualize(grid, seoul):
    n_tot    = len(grid)
    n_t      = grid['is_transport_vuln'].sum()
    n_l      = grid['is_light_vuln'].sum()
    n_c      = grid['is_combined_vuln'].sum()
    n_t_only = (grid['is_transport_vuln'] & ~grid['is_light_vuln']).sum()
    n_l_only = (~grid['is_transport_vuln'] & grid['is_light_vuln']).sum()
    n_good   = (~grid['is_transport_vuln'] & ~grid['is_light_vuln']).sum()

    fig, ax = plt.subplots(1, 1, figsize=(14, 14), facecolor='#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    ax.tick_params(colors='#555')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

    # 양호 (어두운 배경)
    grid[~grid['is_transport_vuln'] & ~grid['is_light_vuln']].plot(
        ax=ax, color='#1a1a2e', edgecolor='none', alpha=0.7)

    # 교통 공백만 (파란색)
    grid[grid['is_transport_vuln'] & ~grid['is_light_vuln']].plot(
        ax=ax, color='#3B82F6', edgecolor='none', alpha=0.7)

    # 보안등 없음만 (주황색)
    grid[~grid['is_transport_vuln'] & grid['is_light_vuln']].plot(
        ax=ax, color='#F59E0B', edgecolor='none', alpha=0.5)

    # 복합 취약 (빨간색) ← 가장 위에 표시
    grid[grid['is_combined_vuln']].plot(
        ax=ax, color='#EF4444', edgecolor='none', alpha=0.9)

    # 서울시 경계선
    seoul.boundary.plot(ax=ax, color='#aaaaaa', linewidth=0.4, alpha=0.5)

    ax.set_title(
        '서울시 교통 공백 × 보안등 취약 복합 위험 지역\n'
        '(빨간색: 교통 공백 + 보안등 없음 동시 해당)',
        fontsize=15, fontweight='bold', color='white', pad=16)
    ax.set_xlabel('X좌표 (EPSG:5179)', fontsize=9, color='#888')
    ax.set_ylabel('Y좌표 (EPSG:5179)', fontsize=9, color='#888')
    ax.tick_params(labelsize=7, colors='#666')

    ax.legend(handles=[
        mpatches.Patch(color='#EF4444',
                       label=f'복합 취약 (교통+보안등): {n_c:,}개 ({n_c/n_tot*100:.1f}%)'),
        mpatches.Patch(color='#3B82F6',
                       label=f'교통 공백만: {n_t_only:,}개 ({n_t_only/n_tot*100:.1f}%)'),
        mpatches.Patch(color='#F59E0B',
                       label=f'보안등 없음만: {n_l_only:,}개 ({n_l_only/n_tot*100:.1f}%)'),
        mpatches.Patch(color='#1a1a2e',
                       label=f'양호: {n_good:,}개 ({n_good/n_tot*100:.1f}%)',
                       edgecolor='#444', linewidth=0.5),
    ], loc='upper right', fontsize=8,
       facecolor='#0d0d1a', edgecolor='#555', labelcolor='white')

    ax.text(0.02, 0.02,
            f"전체 격자: {n_tot:,}개  |  격자 크기: 100m×100m\n"
            f"교통 공백: {n_t:,}개 ({n_t/n_tot*100:.1f}%)"
            f"  |  보안등 취약: {n_l:,}개 ({n_l/n_tot*100:.1f}%)\n"
            f"복합 위험: {n_c:,}개 ({n_c/n_tot*100:.1f}%)",
            transform=ax.transAxes, fontsize=8, color='#aaa',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#0d0d1a',
                      alpha=0.85, edgecolor='#444'))

    plt.tight_layout()
    out_path = BASE_DIR / 'output' / 'seoul_combined_vuln.png'
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.show()
    print(f"[6] 시각화 저장: {out_path}")


# ──────────────────────────────────────────────────────────────
# STEP 7. 결과 저장
# ──────────────────────────────────────────────────────────────
def save_results(grid):
    out_dir = BASE_DIR / 'output'
    out_dir.mkdir(exist_ok=True)

    save_cols = ['grid_id', 'centroid_x', 'centroid_y', 'nearest_dist_m',
                 'is_transport_vuln', 'light_count', 'is_light_vuln',
                 'is_combined_vuln']

    grid[save_cols].to_csv(
        out_dir / 'seoul_combined_vuln.csv', index=False, encoding='utf-8-sig')

    grid[grid['is_combined_vuln']][save_cols].to_csv(
        out_dir / 'seoul_combined_vuln_only.csv', index=False, encoding='utf-8-sig')

    n_c = grid['is_combined_vuln'].sum()
    print(f"[7] 저장 완료 → {out_dir}/")
    print(f"    ├ seoul_combined_vuln.csv       (격자 전체 {len(grid):,}개)")
    print(f"    └ seoul_combined_vuln_only.csv  (복합 취약만 {n_c:,}개)")


# ──────────────────────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  서울시 교통 공백 × 보안등 취약 복합 위험 지역 분석")
    print("=" * 60)

    seoul, boundary = load_seoul_boundary()
    grid            = create_grid(boundary)
    grid = map_nbus(grid)
    grid            = map_lights(grid)
    grid            = flag_combined(grid)
    visualize(grid, seoul)                    # n_gdf 인자 제거
    save_results(grid)

    print("\n✓ 완료")

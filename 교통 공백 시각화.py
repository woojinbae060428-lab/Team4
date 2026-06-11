"""
서울시 여성 안전 교통공백 분석 - 최종 통합 코드
=================================================
지금까지 진행한 모든 분석을 하나의 파일로 통합합니다.

[ 포함된 분석 ]
  Step 1. 서울시 100m×100m 격자 생성 (행정경계 기반)
  Step 2. N버스 교통 공백 격자 판정 (정류장 1km 초과)
  Step 3. 보안등 격자 매핑 (격자별 보안등 수 집계)
  Step 4. 시각화 (교통 공백 지역 파란색 표시)
  Step 5. 결과 저장 (CSV + GeoJSON)

[ 필요 라이브러리 ]
  pip install geopandas shapely matplotlib pandas numpy pyproj scipy openpyxl

[ 입력 파일 (스크립트와 같은 폴더) ]
  - admstr_zone_lgldong_bndry_24.shp  (+ .dbf / .shx / .prj)
  - 서울시버스노선별정류소정보_20260506_.xlsx
  - 서울시_보안등_최종3.csv

[ 출력 (output/ 폴더 자동 생성) ]
  - seoul_grid_final.csv        격자 전체 통합 데이터
  - seoul_grid_final.geojson    격자 전체 (지도 시각화용)
  - seoul_nbus_vuln_blue.png    교통 공백 시각화 이미지

[ 최종 데이터프레임 컬럼 ]
  grid_id          격자 고유 ID
  centroid_x/y     격자 중심점 좌표 (EPSG:5179)
  nearest_dist_m   가장 가까운 N버스 정류장까지 거리 (m)
  nearest_stop     가장 가까운 N버스 정류장명
  is_transport_vuln  교통 취약 여부 (1km 초과 = True)
  light_count      격자 내 보안등 수
  is_light_vuln    보안등 취약 여부 (0개 = True)
  is_combined_vuln 교통 + 보안등 동시 취약 여부

[ 좌표계 ]
  EPSG:5179 (Korea 2000 / Central Belt 2010, 단위: 미터)
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
            print(f"    한글 폰트: {font}")
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
GRID_SIZE = 100     # 격자 크기 (m)
VULN_DIST = 1000    # 교통 취약 기준 거리 (m): N버스 정류장 1km 초과

# 서울시 WGS84 유효 범위 (이상치 필터링용)
SEOUL_LAT = (37.4, 37.72)
SEOUL_LON = (126.75, 127.25)


# ──────────────────────────────────────────────────────────────
# STEP 1. 서울시 행정경계 로드
# ──────────────────────────────────────────────────────────────
def load_seoul_boundary():
    shp_path = BASE_DIR / 'admstr_zone_lgldong_bndry_24.shp'
    if not shp_path.exists():
        raise FileNotFoundError(
            f"행정경계 파일 없음: {shp_path}\n"
            "admstr_zone_lgldong_bndry_24.shp (.dbf/.shx/.prj 포함)을 같은 폴더에 두세요."
        )
    gdf = gpd.read_file(str(shp_path), encoding='cp949')
    seoul = gdf[gdf['EMD_CD'].astype(str).str.startswith('11')].to_crs(CRS_KOREA)
    boundary = seoul.union_all()
    print(f"[1] 서울시 행정경계 로드")
    print(f"    실제 면적: {boundary.area/1_000_000:.2f}km²")
    return seoul, boundary


# ──────────────────────────────────────────────────────────────
# STEP 2. 100m×100m 격자 생성
# ──────────────────────────────────────────────────────────────
def create_grid(boundary):
    minx, miny, maxx, maxy = boundary.bounds
    xs = np.arange(int(minx), int(maxx) + GRID_SIZE, GRID_SIZE)
    ys = np.arange(int(miny), int(maxy) + GRID_SIZE, GRID_SIZE)
    polygons = [box(x, y, x + GRID_SIZE, y + GRID_SIZE)
                for y in ys for x in xs]

    grid = gpd.GeoDataFrame(
        {'grid_id': range(len(polygons))},
        geometry=polygons, crs=CRS_KOREA
    )
    grid = grid[grid.geometry.intersects(boundary)].copy().reset_index(drop=True)
    grid['grid_id']    = range(len(grid))
    grid['centroid_x'] = grid.geometry.centroid.x
    grid['centroid_y'] = grid.geometry.centroid.y

    print(f"\n[2] 격자 생성")
    print(f"    격자 수: {len(grid):,}개")
    print(f"    격자 크기: {GRID_SIZE}m × {GRID_SIZE}m")
    print(f"    격자 면적 합계: {len(grid) * 0.01:.1f}km²")
    return grid


# ──────────────────────────────────────────────────────────────
# STEP 3. N버스 정류장 로드 및 교통 공백 판정
# ──────────────────────────────────────────────────────────────
def map_nbus(grid):
    xlsx_path = BASE_DIR / '서울시버스노선별정류소정보_20260506_.xlsx'
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"N버스 정류장 파일 없음: {xlsx_path}\n"
            "서울시버스노선별정류소정보_20260506_.xlsx를 같은 폴더에 두세요."
        )

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

    # KD-Tree 최근접 거리 계산
    stop_xy = np.array([[g.x, g.y] for g in n_gdf.geometry])
    grid_xy = np.column_stack([grid['centroid_x'], grid['centroid_y']])
    dist, idx = cKDTree(stop_xy).query(grid_xy, k=1)

    grid['nearest_dist_m']     = dist.round(1)
    grid['nearest_stop']       = n_gdf.iloc[idx]['정류소명'].values
    grid['is_transport_vuln']  = grid['nearest_dist_m'] > VULN_DIST

    n_vuln = grid['is_transport_vuln'].sum()
    n_tot  = len(grid)
    print(f"\n[3] N버스 교통 공백 판정")
    print(f"    N버스 고유 정류장: {len(n_gdf):,}개")
    print(f"    교통 공백 (>{VULN_DIST}m): {n_vuln:,}개 ({n_vuln/n_tot*100:.1f}%)")
    print(f"    교통 양호 (≤{VULN_DIST}m): {n_tot-n_vuln:,}개 ({(n_tot-n_vuln)/n_tot*100:.1f}%)")
    print(f"    평균 거리: {grid['nearest_dist_m'].mean():.0f}m")
    print(f"    최대 거리: {grid['nearest_dist_m'].max():.0f}m")
    return grid, n_gdf


# ──────────────────────────────────────────────────────────────
# STEP 4. 보안등 격자 매핑
# ──────────────────────────────────────────────────────────────
def map_lights(grid):
    csv_path = BASE_DIR / '서울시_보안등_최종3.csv'
    if not csv_path.exists():
        print(f"\n[4] 보안등 파일 없음: {csv_path}")
        print(f"    → 보안등 컬럼 없이 진행합니다.")
        grid['light_count']   = 0
        grid['is_light_vuln'] = True
        return grid

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

    # 공간 조인: 격자별 보안등 집계
    joined = gpd.sjoin(
        light_gdf, grid[['grid_id', 'geometry']],
        how='left', predicate='within'
    )
    agg = joined.groupby('grid_id')['설치개수'].sum().reset_index(name='light_count')
    grid = grid.merge(agg, on='grid_id', how='left')
    grid['light_count']   = grid['light_count'].fillna(0).astype(int)
    grid['is_light_vuln'] = grid['light_count'] == 0

    n_with  = (grid['light_count'] > 0).sum()
    n_tot   = len(grid)
    print(f"\n[4] 보안등 격자 매핑")
    print(f"    보안등 총 개소: {len(light_gdf):,}개소")
    print(f"    보안등 있는 격자: {n_with:,}개 ({n_with/n_tot*100:.1f}%)")
    print(f"    보안등 없는 격자: {n_tot-n_with:,}개 ({(n_tot-n_with)/n_tot*100:.1f}%)")
    return grid


# ──────────────────────────────────────────────────────────────
# STEP 5. 복합 취약 격자 판정
# ──────────────────────────────────────────────────────────────
def flag_combined(grid):
    grid['is_combined_vuln'] = (
        grid['is_transport_vuln'] & grid['is_light_vuln']
    )
    n = grid['is_combined_vuln'].sum()
    n_tot = len(grid)
    print(f"\n[5] 복합 취약 격자 판정 (교통 + 보안등 동시 취약)")
    print(f"    복합 취약: {n:,}개 ({n/n_tot*100:.1f}%)")
    return grid


# ──────────────────────────────────────────────────────────────
# STEP 6. 시각화 (교통 공백 지역 파란색)
# ──────────────────────────────────────────────────────────────
def visualize(grid, seoul, n_gdf):
    n_vuln = grid['is_transport_vuln'].sum()
    n_tot  = len(grid)

    fig, ax = plt.subplots(1, 1, figsize=(14, 14), facecolor='#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    ax.tick_params(colors='#555')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

    # 교통 양호 (어두운 배경)
    grid[~grid['is_transport_vuln']].plot(
        ax=ax, color='#1e1e2e', edgecolor='none', alpha=0.8)

    # 교통 공백 (파란색)
    grid[grid['is_transport_vuln']].plot(
        ax=ax, color='#3B82F6', edgecolor='none', alpha=0.85)

    # 서울시 경계선
    seoul.boundary.plot(ax=ax, color='#aaaaaa', linewidth=0.5, alpha=0.6)

    # N버스 정류장 (노란 점)
    ax.scatter(
        [g.x for g in n_gdf.geometry],
        [g.y for g in n_gdf.geometry],
        c='#FFD700', s=6, alpha=0.9, zorder=5
    )

    ax.set_title(
        '서울시 N버스 교통 공백 지역\n(파란색: N버스 정류장 1km 초과 지역)',
        fontsize=16, fontweight='bold', color='white', pad=16
    )
    ax.set_xlabel('X좌표 (EPSG:5179)', fontsize=9, color='#888')
    ax.set_ylabel('Y좌표 (EPSG:5179)', fontsize=9, color='#888')
    ax.tick_params(labelsize=7, colors='#666')

    ax.legend(handles=[
        mpatches.Patch(color='#3B82F6',
                       label=f'교통 공백 (1km 초과): {n_vuln:,}개 격자'),
        mpatches.Patch(color='#1e1e2e',
                       label=f'교통 양호 (1km 이내): {n_tot-n_vuln:,}개 격자',
                       edgecolor='#555', linewidth=0.5),
        mpatches.Patch(color='#FFD700',
                       label=f'N버스 정류장: {len(n_gdf):,}개'),
    ], loc='upper right', fontsize=9,
       facecolor='#0d0d1a', edgecolor='#555', labelcolor='white')

    ax.text(
        0.02, 0.02,
        f"전체 격자: {n_tot:,}개  |  격자 크기: 100m×100m\n"
        f"교통 공백률: {n_vuln/n_tot*100:.1f}%  |  N버스 노선 수: 18개\n"
        f"평균 거리: {grid['nearest_dist_m'].mean():.0f}m  |"
        f"  최대: {grid['nearest_dist_m'].max():.0f}m",
        transform=ax.transAxes, fontsize=8, color='#aaa',
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#0d0d1a',
                  alpha=0.85, edgecolor='#444')
    )

    plt.tight_layout()
    out_path = BASE_DIR / 'output' / 'seoul_nbus_vuln_blue.png'
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.show()
    print(f"\n[6] 시각화 저장: {out_path}")


# ──────────────────────────────────────────────────────────────
# STEP 7. 결과 저장
# ──────────────────────────────────────────────────────────────
def save_results(grid):
    out_dir = BASE_DIR / 'output'
    out_dir.mkdir(exist_ok=True)

    save_cols = [
        'grid_id', 'centroid_x', 'centroid_y',
        'nearest_dist_m', 'nearest_stop', 'is_transport_vuln',
        'light_count', 'is_light_vuln',
        'is_combined_vuln',
    ]
    save_cols = [c for c in save_cols if c in grid.columns]

    # CSV
    grid[save_cols].to_csv(
        out_dir / 'seoul_grid_final.csv',
        index=False, encoding='utf-8-sig'
    )

    # GeoJSON
    grid[save_cols + ['geometry']].to_file(
        str(out_dir / 'seoul_grid_final.geojson'), driver='GeoJSON'
    )

    print(f"\n[7] 저장 완료 → {out_dir}/")
    print(f"    ├ seoul_grid_final.csv      (격자 전체 {len(grid):,}개)")
    print(f"    ├ seoul_grid_final.geojson  (지도 시각화용)")
    print(f"    └ seoul_nbus_vuln_blue.png  (교통 공백 시각화)")
    print(f"\n    최종 컬럼 ({len(save_cols)}개):")
    for c in save_cols:
        print(f"      · {c}")


# ──────────────────────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  서울시 여성 안전 교통공백 분석 — 최종 통합 코드")
    print("=" * 60)

    seoul, boundary = load_seoul_boundary()   # 1. 행정경계
    grid            = create_grid(boundary)   # 2. 격자 생성
    grid, n_gdf     = map_nbus(grid)          # 3. N버스 교통 공백
    grid            = map_lights(grid)        # 4. 보안등 매핑
    grid            = flag_combined(grid)     # 5. 복합 취약 판정
    visualize(grid, seoul, n_gdf)             # 6. 시각화
    save_results(grid)                        # 7. 저장

    print("\n✓ 전체 분석 완료")

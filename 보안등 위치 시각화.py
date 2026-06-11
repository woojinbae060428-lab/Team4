"""
서울시 전체 100m×100m 격자 + 보안등 위치 시각화
==============================================
필요 라이브러리:
    pip install geopandas shapely matplotlib pandas pyproj

입력 파일 (스크립트와 같은 폴더):
    - admstr_zone_lgldong_bndry_24.shp  (서울시 행정경계, .dbf/.shx/.prj 포함)
    - 서울시_보안등_최종3.csv

출력 (output/ 폴더 자동 생성):
    - seoul_light_grid.png   시각화 이미지
    - seoul_light_grid.csv   격자 데이터
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from shapely.geometry import Point, box
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
            print(f"    한글 폰트 설정: {font}")
            return
    for f in fm.fontManager.ttflist:
        if any(k in f.name for k in ['Gothic', 'Nanum', 'Malgun', 'Apple']):
            plt.rcParams['font.family'] = f.name
            plt.rcParams['axes.unicode_minus'] = False
            print(f"    한글 폰트 설정: {f.name}")
            return
    print("    ⚠ 한글 폰트 없음 → pip3 install koreanize-matplotlib 실행 후 재시도")

set_korean_font()

# ──────────────────────────────────────────────────────────────
# 설정값
# ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
CRS_KOREA = "EPSG:5179"
GRID_SIZE = 100


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
    print(f"[1] 서울시 경계 로드 완료")
    print(f"    실제 면적: {boundary.area/1_000_000:.2f}km²")
    return seoul, boundary


# ──────────────────────────────────────────────────────────────
# STEP 2. 100m×100m 격자 생성 (서울시 경계 클리핑)
# ──────────────────────────────────────────────────────────────
def create_grid(boundary):
    minx, miny, maxx, maxy = boundary.bounds
    xs = np.arange(int(minx), int(maxx) + GRID_SIZE, GRID_SIZE)
    ys = np.arange(int(miny), int(maxy) + GRID_SIZE, GRID_SIZE)
    polygons = [box(x, y, x + GRID_SIZE, y + GRID_SIZE) for y in ys for x in xs]

    grid = gpd.GeoDataFrame({'grid_id': range(len(polygons))},
                             geometry=polygons, crs=CRS_KOREA)
    grid = grid[grid.geometry.intersects(boundary)].copy().reset_index(drop=True)
    grid['grid_id'] = range(len(grid))
    grid['centroid_x'] = grid.geometry.centroid.x
    grid['centroid_y'] = grid.geometry.centroid.y

    print(f"\n[2] 격자 생성 완료")
    print(f"    격자 수: {len(grid):,}개")
    print(f"    격자 면적 합계: {len(grid) * 0.01:.1f}km²")
    return grid


# ──────────────────────────────────────────────────────────────
# STEP 3. 보안등 데이터 로드
# ──────────────────────────────────────────────────────────────
def load_lights():
    csv_path = BASE_DIR / '서울시_보안등_최종3.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"보안등 파일 없음: {csv_path}")

    df = pd.read_csv(str(csv_path), encoding='utf-8')
    df['위도']    = pd.to_numeric(df['위도'],    errors='coerce')
    df['경도']    = pd.to_numeric(df['경도'],    errors='coerce')
    df['설치개수'] = pd.to_numeric(df['설치개수'], errors='coerce').fillna(1).astype(int)
    df = df.dropna(subset=['위도', '경도'])
    df = df[df['위도'].between(37.4, 37.72) & df['경도'].between(126.75, 127.25)]

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(x, y) for x, y in zip(df['경도'], df['위도'])],
        crs="EPSG:4326"
    ).to_crs(CRS_KOREA)

    print(f"\n[3] 보안등 데이터 로드 완료")
    print(f"    총 개소: {len(gdf):,}개소")
    print(f"    총 설치 수: {gdf['설치개수'].sum():,}개")
    return gdf


# ──────────────────────────────────────────────────────────────
# STEP 4. 격자별 보안등 집계 (공간 조인)
# ──────────────────────────────────────────────────────────────
def map_lights_to_grid(grid, light_gdf):
    joined = gpd.sjoin(light_gdf, grid[['grid_id', 'geometry']],
                       how='left', predicate='within')
    agg = joined.groupby('grid_id')['설치개수'].sum().reset_index(name='보안등수')

    grid = grid.merge(agg, on='grid_id', how='left')
    grid['보안등수']   = grid['보안등수'].fillna(0).astype(int)
    grid['보안등있음'] = grid['보안등수'] > 0

    n_with  = grid['보안등있음'].sum()
    n_tot   = len(grid)
    print(f"\n[4] 격자별 보안등 집계 완료")
    print(f"    보안등 있는 격자: {n_with:,}개 ({n_with/n_tot*100:.1f}%)")
    print(f"    보안등 없는 격자: {n_tot-n_with:,}개 ({(n_tot-n_with)/n_tot*100:.1f}%)")
    print(f"    격자당 평균 보안등: {grid[grid['보안등수']>0]['보안등수'].mean():.1f}개")
    print(f"    격자당 최대 보안등: {grid['보안등수'].max()}개")
    return grid


# ──────────────────────────────────────────────────────────────
# STEP 5. 시각화
# ──────────────────────────────────────────────────────────────
def visualize(grid, seoul, light_gdf):
    fig, axes = plt.subplots(1, 2, figsize=(20, 11), facecolor='#0d0d1a')
    fig.suptitle('서울시 100m×100m 격자 보안등 현황',
                 fontsize=20, fontweight='bold', color='white', y=0.98)

    for ax in axes:
        ax.set_facecolor('#0d0d1a')
        ax.tick_params(colors='#666')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

    no_light  = grid[~grid['보안등있음']]
    has_light = grid[grid['보안등있음']]
    n_tot = len(grid)

    # ── 왼쪽: 보안등 유무 (파란색) ──
    ax1 = axes[0]
    no_light.plot(ax=ax1, color='#1a1a2e', edgecolor='none', alpha=0.5)
    has_light.plot(ax=ax1, color='#3B82F6', edgecolor='none', alpha=0.75)
    seoul.boundary.plot(ax=ax1, color='#ffffff', linewidth=0.4, alpha=0.6)

    ax1.set_title('격자별 보안등 유무', fontsize=14, color='white', pad=12)
    ax1.set_xlabel('X좌표 (EPSG:5179)', fontsize=9, color='#888')
    ax1.set_ylabel('Y좌표 (EPSG:5179)', fontsize=9, color='#888')
    ax1.tick_params(labelsize=7, colors='#666')

    patch_blue = mpatches.Patch(color='#3B82F6',
                                 label=f"보안등 있음 ({has_light.shape[0]:,}개 격자)")
    patch_dark = mpatches.Patch(color='#1a1a2e',
                                 label=f"보안등 없음 ({no_light.shape[0]:,}개 격자)",
                                 edgecolor='#444', linewidth=0.5)
    ax1.legend(handles=[patch_blue, patch_dark], loc='upper right', fontsize=8,
               facecolor='#0d0d1a', edgecolor='#555', labelcolor='white')
    ax1.text(0.02, 0.02,
             f"전체 격자: {n_tot:,}개\n격자 크기: 100m × 100m\n면적: {n_tot*0.01:.1f}km²",
             transform=ax1.transAxes, fontsize=8, color='#aaa',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d0d1a', alpha=0.8))

    # ── 오른쪽: 보안등 밀도 ──
    ax2 = axes[1]
    no_light.plot(ax=ax2, color='#0d0d1a', edgecolor='none', alpha=0.5)
    if len(has_light) > 0:
        has_light.plot(ax=ax2, column='보안등수', cmap='Blues',
                       vmin=1, vmax=has_light['보안등수'].quantile(0.95),
                       edgecolor='none', alpha=0.9, legend=False)
        norm = Normalize(vmin=1, vmax=has_light['보안등수'].quantile(0.95))
        sm = ScalarMappable(cmap='Blues', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, fraction=0.03, pad=0.02)
        cbar.set_label('격자당 보안등 수', fontsize=8, color='#aaa')
        cbar.ax.yaxis.set_tick_params(color='#aaa', labelcolor='#aaa')

    seoul.boundary.plot(ax=ax2, color='#ffffff', linewidth=0.4, alpha=0.6)
    ax2.set_title('격자별 보안등 밀도 (수량)', fontsize=14, color='white', pad=12)
    ax2.set_xlabel('X좌표 (EPSG:5179)', fontsize=9, color='#888')
    ax2.set_ylabel('', fontsize=9, color='#888')
    ax2.tick_params(labelsize=7, colors='#666')
    ax2.text(0.02, 0.02,
             f"보안등 총 개소: {len(light_gdf):,}\n"
             f"설치 수 합계: {light_gdf['설치개수'].sum():,}개\n"
             f"격자당 최대: {grid['보안등수'].max()}개",
             transform=ax2.transAxes, fontsize=8, color='#aaa',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d0d1a', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = BASE_DIR / 'output' / 'seoul_light_grid.png'
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"\n[5] 시각화 저장: {out_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# STEP 6. 결과 저장
# ──────────────────────────────────────────────────────────────
def save_results(grid):
    out_dir = BASE_DIR / 'output'
    out_dir.mkdir(exist_ok=True)

    save_cols = ['grid_id', 'centroid_x', 'centroid_y', '보안등수', '보안등있음']
    grid[save_cols].to_csv(out_dir / 'seoul_light_grid.csv',
                            index=False, encoding='utf-8-sig')
    print(f"[6] CSV 저장 완료 → output/seoul_light_grid.csv ({len(grid):,}개 격자)")


# ──────────────────────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  서울시 100m×100m 격자 + 보안등 시각화")
    print("=" * 55)

    seoul, boundary = load_seoul_boundary()
    grid            = create_grid(boundary)
    light_gdf       = load_lights()
    grid            = map_lights_to_grid(grid, light_gdf)
    visualize(grid, seoul, light_gdf)
    save_results(grid)

    print("\n✓ 완료")

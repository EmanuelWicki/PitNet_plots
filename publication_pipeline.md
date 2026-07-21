# Analysis Pipeline Code

`publication_pipeline.py` — single-file pipeline used to generate the manuscript figures (coregistration, gland-midpoint correction, and tumour-probability mapping).

```python
"""
PitNET tumour analysis: single-file pipeline for the manuscript figures.

Three stages, run end-to-end from raw segmented meshes to the published figures:

  1. Coregistration, step 1: Translate the pitstalk-gland junction to the
     origin (translation only; each tumour's native orientation and size are
     preserved).
  2. Coregistration, step 2: Translate each gland along X so that x=0 sits
     at the midpoint of its mediolateral extent (corrects for pitstalks that
     are not perfectly midline, which would otherwise bias laterality
     comparisons).
  3. Topographic probability mapping: Voxelise each aligned tumour mesh,
     project onto the coronal/axial/sagittal plane, accumulate a per-patient
     presence vote, and normalise by N to obtain a per-pixel tumour-probability
     map for each phenotype and body side, plus Acromegaly-Cushing difference
     maps.

Coordinate system (set by mesh orientation):
    +X = Right, +Y = Posterior, +Z = Superior; origin = pitstalk-gland junction.

Input
-----
RAW_DATA_DIR/<id>_pit.stl, <id>_tumor.stl, <id>_pitstalk.stl  (per-patient meshes)
METADATA_FILE                                                 (case_ID, lab_phenotype)

Output (all under WORK_DIR)
----------------------------
stage1_junction_aligned/   aligned meshes + pitstalk_aligned_positions.json
stage2_gland_centered/     gland_centered_aligned_positions.json + x_corrections_summary.json
figures/probability/       per-phenotype, per-side tumour-probability maps
figures/difference/        Acromegaly-vs-Cushing difference maps (left / right / all)

Mesh processing: Python 3.11, trimesh, scipy, numpy, matplotlib, pandas.
"""

import json
import warnings
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter, binary_dilation, binary_fill_holes

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration -- values used to generate the published figures
# =============================================================================
RAW_DATA_DIR = Path("data/combined_final/Complete-segmentation-set_2026_06_18")
METADATA_FILE = RAW_DATA_DIR / "pitNET-anatomy-data_2026-06-23_anonym.xlsx"

WORK_DIR = Path("publication_pipeline_output")
STAGE1_DIR = WORK_DIR / "stage1_junction_aligned"
STAGE2_DIR = WORK_DIR / "stage2_gland_centered"
VOXEL_CACHE_DIR = WORK_DIR / "voxel_cache"
FIGURE_DIR = WORK_DIR / "figures"

JUNCTION_PERCENTILE = 5     # bottom percentile of pitstalk Y-vertices defining the junction
VOXEL_PITCH_MM = 0.05       # isotropic voxelisation / heatmap grid pitch, mm
SIGMA_MM = 0.0              # Gaussian smoothing bandwidth for these figures (0 = none)

AXIS_LABELS = {
    0: ('Y (-Anterior/+Posterior) [mm]', 'Z (-Inferior/+Superior) [mm]'),
    1: ('X (-Left/+Right) [mm]', 'Z (-Inferior/+Superior) [mm]'),
    2: ('X (-Left/+Right) [mm]', 'Y (-Anterior/+Posterior) [mm]'),
}
AXIS_NAMES = {0: 'Sagittal', 1: 'Coronal', 2: 'Axial'}

# Plot extents in mm: {axis: [h_min, h_max, v_min, v_max]}
PLOT_EXTENTS = {
    0: [-20, 20, -30, 10],
    1: [-20, 20, -30, 10],
    2: [-20, 20, -20, 20],
}

# Reference gland ellipse per projection for the difference figure: (cx, cy, width_mm, height_mm)
GLAND_ELLIPSE = {
    2: (0, 0, 12.0, 12.0),
    1: (0, -4.5, 12.0, 9.0),
    0: (0, -4.5, 12.0, 9.0),
}


# =============================================================================
# STAGE 1 -- pitstalk-gland junction to origin (translation only)
# =============================================================================
def find_pitstalk_gland_junction(pitstalk_mesh, percentile=JUNCTION_PERCENTILE):
    """
    Anatomical anchor point: centroid of the bottom-`percentile`% of pitstalk
    vertices by Y -- the pitstalk-gland junction, the most posterior/inferior
    part of the stalk, where it meets the gland.
    """
    y = pitstalk_mesh.vertices[:, 1]
    mask = y <= np.percentile(y, percentile)
    return pitstalk_mesh.vertices[mask].mean(axis=0)


def align_patient_to_origin(components, percentile=JUNCTION_PERCENTILE):
    """Translate pit/tumour/pitstalk so the junction sits at (0,0,0). No rotation or scaling."""
    anchor = find_pitstalk_gland_junction(components['pitstalk'], percentile)
    translation = -anchor
    aligned = {}
    for name, mesh in components.items():
        aligned[name] = mesh.copy()
        aligned[name].vertices = aligned[name].vertices + translation
    return aligned, anchor


def run_stage1_junction_alignment(raw_data_dir, output_dir):
    """Stage 1 for every patient found in raw_data_dir. Returns a list of result dicts."""
    output_dir = Path(output_dir)
    aligned_dir = output_dir / "aligned_meshes"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "pitstalk_aligned_positions.json"
    if results_path.exists():
        print(f"[stage1] Using existing results: {results_path}")
        return json.loads(results_path.read_text())

    raw_data_dir = Path(raw_data_dir)
    patient_ids = sorted(f.stem.replace('_pit', '') for f in raw_data_dir.glob('*_pit.stl'))
    print(f"[stage1] Aligning {len(patient_ids)} patients to the pitstalk-gland junction...")

    results = []
    for patient_id in patient_ids:
        components = {}
        for name in ('pit', 'tumor', 'pitstalk'):
            path = raw_data_dir / f"{patient_id}_{name}.stl"
            components[name] = trimesh.load_mesh(str(path)) if path.exists() else None

        if any(components[name] is None for name in ('pit', 'tumor', 'pitstalk')):
            print(f"  [skip] {patient_id}: missing pit/tumor/pitstalk mesh")
            continue

        aligned, anchor = align_patient_to_origin(components)

        for name, mesh in aligned.items():
            mesh.export(str(aligned_dir / f"{patient_id}_{name}_aligned.stl"))

        results.append({
            'patient_id': patient_id,
            'tumor_centroid': aligned['tumor'].centroid.tolist(),
            'pit_centroid': aligned['pit'].centroid.tolist(),
            'anchor_point_original': anchor.tolist(),
        })

    results_path.write_text(json.dumps(results, indent=2))
    print(f"[stage1] Done: {len(results)} patients -> {results_path}")
    return results


# =============================================================================
# STAGE 2 -- gland mediolateral (X) midpoint to x=0
# =============================================================================
def apply_gland_midpoint_correction(stage1_results, aligned_meshes_dir):
    """Shift each patient's X so their aligned gland's X-span midpoint sits at X=0."""
    aligned_meshes_dir = Path(aligned_meshes_dir)
    refined, corrections = [], {}

    for result in stage1_results:
        patient_id = result['patient_id']
        pit_path = aligned_meshes_dir / f"{patient_id}_pit_aligned.stl"
        if not pit_path.exists():
            continue

        x = trimesh.load_mesh(str(pit_path)).vertices[:, 0]
        x_midpoint = (x.min() + x.max()) / 2
        x_correction = -x_midpoint

        refined_result = dict(result)
        centroid = np.array(result['tumor_centroid'])
        centroid[0] += x_correction
        refined_result['tumor_centroid'] = centroid.tolist()
        refined.append(refined_result)

        corrections[patient_id] = {
            'x_correction': float(x_correction),
            'x_midpoint': float(x_midpoint),
            'x_span': float(x.max() - x.min()),
        }

    return refined, corrections


def run_stage2_gland_centering(stage1_results, stage1_aligned_dir, metadata, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    refined_path = output_dir / "gland_centered_aligned_positions.json"
    corrections_path = output_dir / "x_corrections_summary.json"
    if refined_path.exists() and corrections_path.exists():
        print(f"[stage2] Using existing results: {refined_path}")
        return json.loads(refined_path.read_text()), json.loads(corrections_path.read_text())

    refined, corrections = apply_gland_midpoint_correction(stage1_results, stage1_aligned_dir)
    refined = merge_with_metadata(refined, metadata)

    refined_path.write_text(json.dumps(refined, indent=2))
    corrections_path.write_text(json.dumps(corrections, indent=2))
    print(f"[stage2] Done: {len(refined)} patients -> {refined_path}")
    return refined, corrections


# =============================================================================
# Metadata + phenotype/laterality filtering
# =============================================================================
def load_metadata(metadata_file):
    xl = pd.ExcelFile(metadata_file)
    sheet = 'Database' if 'Database' in xl.sheet_names else xl.sheet_names[0]
    df = pd.read_excel(xl, sheet_name=sheet)
    id_col = next(c for c in df.columns if str(c).lower().replace(' ', '').replace('_', '')
                  in ('caseid', 'id', 'patientid', 'case', 'patient'))
    df['case_ID'] = df[id_col].astype(str).str.zfill(3)
    return df


def merge_with_metadata(results, metadata):
    for r in results:
        row = metadata[metadata['case_ID'] == str(r['patient_id']).zfill(3)]
        if not row.empty:
            row = row.iloc[0]
            r['lab_phenotype'] = row.get('lab_phenotype', 'unknown')
            r['clinical_phenotype'] = row.get('clinical_phenotype', r['lab_phenotype'])
        else:
            r['lab_phenotype'] = r['clinical_phenotype'] = 'unknown'
    return results


def filter_by_type(results, tumor_type):
    """Keep patients whose lab/clinical phenotype matches `tumor_type`; drop mixed phenotypes."""
    out = []
    for r in results:
        lab = str(r.get('lab_phenotype', '')).lower()
        clin = str(r.get('clinical_phenotype', '')).lower()
        if 'mixed' in lab or 'mixed' in clin:
            continue
        if tumor_type in lab or tumor_type in clin:
            out.append(r)
    return out


def filter_by_laterality(results, side):
    """side: 'left' (tumour centroid X < 0) or 'right' (X >= 0)."""
    return [r for r in results if (r['tumor_centroid'][0] < 0) == (side == 'left')]


# =============================================================================
# Voxel cache
# =============================================================================
def _largest_component(mesh):
    """Keep only the largest connected component; secondary components are reconstruction artifacts."""
    parts = mesh.split(only_watertight=False)
    return max(parts, key=lambda m: len(m.faces)) if len(parts) > 1 else mesh


def _voxelize_open_mesh(mesh, voxel_size):
    """Surface-sample + dilate + fill-holes voxelisation for non-watertight meshes."""
    bounds = mesh.bounds
    pad = voxel_size * 3
    grid_min = bounds[0] - pad
    shape = tuple(np.ceil((bounds[1] - bounds[0] + 2 * pad) / voxel_size).astype(int) + 1)

    n_samples = max(20000, int(mesh.area / (voxel_size ** 2) * 10))
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
    idx = np.clip(np.floor((pts - grid_min) / voxel_size).astype(int), 0, np.array(shape) - 1)

    shell = np.zeros(shape, dtype=bool)
    shell[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    filled = binary_fill_holes(binary_dilation(shell, iterations=2))

    ijk = np.argwhere(filled)
    return (ijk * voxel_size + grid_min + voxel_size / 2).astype(np.float32)


def get_cached_voxels(tumor_path, voxel_size=VOXEL_PITCH_MM, cache_dir=VOXEL_CACHE_DIR):
    """Interior point cloud for a tumour mesh, cached to disk (invalidated when the STL changes)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{Path(tumor_path).stem}_vox{voxel_size:.2f}mm.npy"
    if cache_path.exists() and cache_path.stat().st_mtime >= Path(tumor_path).stat().st_mtime:
        return np.load(str(cache_path))

    mesh = _largest_component(trimesh.load_mesh(str(tumor_path)))
    if mesh.is_watertight:
        points = np.array(mesh.voxelized(pitch=voxel_size).fill().points, dtype=np.float32)
    else:
        points = _voxelize_open_mesh(mesh, voxel_size)

    np.save(str(cache_path), points)
    return points


def get_tumor_paths(results, aligned_meshes_dir, x_corrections):
    """(path, x_shift) for each patient with an aligned tumour mesh on disk."""
    aligned_meshes_dir = Path(aligned_meshes_dir)
    out = []
    for r in results:
        pid = str(r['patient_id']).zfill(3)
        path = aligned_meshes_dir / f"{pid}_tumor_aligned.stl"
        if path.exists():
            shift = x_corrections.get(pid, {}).get('x_correction', 0.0)
            out.append((path, shift))
    return out


# =============================================================================
# STAGE 3 -- probability mapping
# =============================================================================
def build_probability_map(tumor_paths_with_shift, axis, extent,
                           voxel_pitch=VOXEL_PITCH_MM, sigma_mm=SIGMA_MM):
    """
    Per-pixel tumour-probability map: each patient casts a binary (0/1) vote per
    grid cell (present/absent), votes are summed across patients and divided by
    N to give the fraction of patients with tumour at that location, then
    optionally Gaussian-smoothed (sigma_mm, physical).
    """
    dims = [d for d in (0, 1, 2) if d != axis]
    xmin, xmax, ymin, ymax = extent
    res_x = max(50, int(round((xmax - xmin) / voxel_pitch)))
    res_y = max(50, int(round((ymax - ymin) / voxel_pitch)))

    grid = np.zeros((res_y, res_x))
    n_valid = 0
    for tumor_path, x_shift in tumor_paths_with_shift:
        voxels = get_cached_voxels(tumor_path, voxel_pitch)
        if x_shift:
            voxels = voxels.copy()
            voxels[:, 0] += x_shift
        if len(voxels) == 0:
            continue

        xi = ((voxels[:, dims[0]] - xmin) / (xmax - xmin) * (res_x - 1)).astype(int)
        yi = ((voxels[:, dims[1]] - ymin) / (ymax - ymin) * (res_y - 1)).astype(int)
        valid = (xi >= 0) & (xi < res_x) & (yi >= 0) & (yi < res_y)
        if valid.any():
            patient_mask = np.zeros((res_y, res_x), dtype=bool)
            patient_mask[yi[valid], xi[valid]] = True
            grid += patient_mask
            n_valid += 1

    if n_valid:
        grid /= n_valid
    if sigma_mm > 0:
        grid = gaussian_filter(grid, sigma=sigma_mm / voxel_pitch)
    return grid


def plot_probability_map(results, aligned_meshes_dir, x_corrections,
                          title, output_path):
    """Per-phenotype, per-side tumour-probability map: 3-view (axial/coronal/sagittal) figure."""
    tumor_paths = get_tumor_paths(results, aligned_meshes_dir, x_corrections)
    if len(tumor_paths) < 2:
        print(f"  [skip] {title}: fewer than 2 tumours with mesh data")
        return

    centroids = np.array([r['tumor_centroid'] for r in results])
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, axis in zip(axes, (2, 1, 0)):
        dims = [d for d in (0, 1, 2) if d != axis]
        xlabel, ylabel = AXIS_LABELS[axis]
        extent = PLOT_EXTENTS[axis]

        heatmap = build_probability_map(tumor_paths, axis, extent)

        if heatmap.max() > 0:
            im = ax.imshow(heatmap, extent=extent, origin='lower', cmap='hot',
                            interpolation='bilinear', alpha=0.85)
            plt.colorbar(im, ax=ax, label='Fraction of patients with tumour', shrink=0.8)

        proj_c = centroids[:, dims]
        ax.scatter(proj_c[:, 0], proj_c[:, 1], c='white', s=40, edgecolors='black', linewidth=1,
                   alpha=0.85, zorder=10, label='Tumour centroids')
        mean_xy = proj_c.mean(axis=0)
        ax.scatter(*mean_xy, marker='*', c='lime', s=400, edgecolors='black', linewidth=1.5,
                   zorder=30, label=f'Mean ({mean_xy[0]:+.1f}, {mean_xy[1]:+.1f} mm)')

        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(AXIS_NAMES[axis], fontweight='bold')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect('equal')
        if axis == 2:
            ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(f'{title}\n(Heatmap = fraction of patients with tumour)', fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] {output_path.name}")


def plot_difference_map(results_acro, results_cush, aligned_meshes_dir, x_corrections,
                         title, output_path):
    """Acromegaly-vs-Cushing difference map: 2-row x 3-col figure (3D diverging surface + 2D projection)."""
    acro_paths = get_tumor_paths(results_acro, aligned_meshes_dir, x_corrections)
    cush_paths = get_tumor_paths(results_cush, aligned_meshes_dir, x_corrections)
    if not acro_paths or not cush_paths:
        print(f"  [skip] {title}: one group has no tumour meshes")
        return

    cmap_div = plt.cm.RdBu
    fig = plt.figure(figsize=(21, 12), facecolor='white')
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.6, 1], hspace=0.38, wspace=0.28)

    for i, axis in enumerate((2, 1, 0)):
        xlabel, ylabel = AXIS_LABELS[axis]
        extent = PLOT_EXTENTS[axis]
        xmin, xmax, ymin, ymax = extent

        h_acro = build_probability_map(acro_paths, axis, extent)
        h_cush = build_probability_map(cush_paths, axis, extent)
        diff = h_acro - h_cush

        pos_vals, neg_vals = diff[diff > 0], diff[diff < 0]
        vmax_pos = float(np.percentile(pos_vals, 97)) if pos_vals.size else 1e-3
        vmax_neg = float(np.percentile(np.abs(neg_vals), 85)) if neg_vals.size else 1e-3
        vmax_sym = max(vmax_pos, vmax_neg)
        support = (h_acro > 0.005) | (h_cush > 0.005)
        norm = TwoSlopeNorm(vmin=-vmax_sym, vcenter=0, vmax=vmax_sym)

        X, Y = np.meshgrid(np.linspace(xmin, xmax, diff.shape[1]), np.linspace(ymin, ymax, diff.shape[0]))

        # --- Row 0: 3D difference surface (positives drawn behind negatives) ---
        ax3d = fig.add_subplot(gs[0, i], projection='3d')
        ax3d.computed_zorder = False

        fc_pos = cmap_div(norm(diff))
        fc_pos[~support | (diff <= 0), 3] = 0.0
        fc_pos[support & (diff > 0), 3] = 0.6
        ax3d.plot_surface(X, Y, diff, facecolors=fc_pos, linewidth=0, rcount=80, ccount=80, shade=False)

        fc_neg = cmap_div(norm(diff))
        fc_neg[~support | (diff >= 0), 3] = 0.0
        fc_neg[support & (diff < 0), 3] = 0.6
        ax3d.plot_surface(X, Y, diff, facecolors=fc_neg, linewidth=0, rcount=80, ccount=80, shade=False)

        cx, cy, ew, eh = GLAND_ELLIPSE[axis]
        theta = np.linspace(0, 2 * np.pi, 120)
        ax3d.plot(cx + (ew / 2) * np.cos(theta), cy + (eh / 2) * np.sin(theta), np.zeros(120),
                  color='gray', lw=1.5)
        ax3d.set_xlabel(xlabel, fontsize=8, labelpad=5)
        ax3d.set_ylabel(ylabel, fontsize=8, labelpad=5)
        ax3d.set_zlabel('Acro - Cush', fontsize=7)
        ax3d.set_title(AXIS_NAMES[axis], fontweight='bold', pad=6)
        ax3d.view_init(elev=25, azim=-60)

        # --- Row 1: 2D projection of the same data ---
        ax2d = fig.add_subplot(gs[1, i])
        ax2d.set_facecolor('#f2f2f2')
        cmap_masked = copy(cmap_div)
        cmap_masked.set_bad(alpha=0)
        diff_ma = np.ma.array(diff, mask=~support)
        im = ax2d.imshow(diff_ma, extent=extent, origin='lower', cmap=cmap_masked, norm=norm,
                          aspect='equal', interpolation='bilinear')
        ax2d.add_patch(Ellipse((cx, cy), width=ew, height=eh, facecolor='none', edgecolor='gray', lw=1.5))
        ax2d.axhline(0, color='#aaaaaa', lw=0.5, alpha=0.7)
        ax2d.axvline(0, color='#aaaaaa', lw=0.5, alpha=0.7)

        cbar = plt.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04)
        cbar.set_label('Acro - Cush\n(fraction of patients)', fontsize=7)
        ax2d.set_xlabel(xlabel)
        ax2d.set_ylabel(ylabel)
        ax2d.set_title(AXIS_NAMES[axis], fontweight='bold')
        if axis == 2:
            ax2d.invert_yaxis()

    fig.suptitle(
        f'{title}\nBLUE = Acromegaly dominant | RED = Cushing dominant '
        f'(N acro={len(results_acro)}, N cush={len(results_cush)})',
        fontweight='bold', y=1.01)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] {output_path.name}")


# =============================================================================
# Orchestration
# =============================================================================
def main():
    metadata = load_metadata(METADATA_FILE)

    stage1_results = run_stage1_junction_alignment(RAW_DATA_DIR, STAGE1_DIR)
    stage1_aligned_dir = STAGE1_DIR / "aligned_meshes"

    results, x_corrections = run_stage2_gland_centering(
        stage1_results, stage1_aligned_dir, metadata, STAGE2_DIR)

    print("\n=== Per-phenotype, per-side tumour-probability maps ===")
    for tumor_type in ('acromegaly', 'cushing'):
        type_results = filter_by_type(results, tumor_type)
        for side in ('left', 'right'):
            side_results = filter_by_laterality(type_results, side)
            print(f"{tumor_type} / {side}: N={len(side_results)}")
            if len(side_results) >= 2:
                plot_probability_map(
                    side_results, stage1_aligned_dir, x_corrections,
                    f'{tumor_type.title()} - {side.title()} side (N={len(side_results)})',
                    FIGURE_DIR / "probability" / f'{tumor_type}_{side}_probability.png')

    print("\n=== Acromegaly vs Cushing difference maps ===")
    for side in ('left', 'right', 'all'):
        side_results = results if side == 'all' else filter_by_laterality(results, side)
        acro = filter_by_type(side_results, 'acromegaly')
        cush = filter_by_type(side_results, 'cushing')
        print(f"{side}: acro N={len(acro)}, cush N={len(cush)}")
        plot_difference_map(
            acro, cush, stage1_aligned_dir, x_corrections,
            f'Acromegaly vs Cushing - {side.title()}',
            FIGURE_DIR / "difference" / f'difference_{side}.png')

    print("\nDone.")


if __name__ == "__main__":
    main()
```

import numpy as np


def interpolate_on_safe_reference_segments(
    ref_m: np.ndarray, ref_v: np.ndarray, ref_c: np.ndarray, m_grid: np.ndarray
):
    """To compare Druedahl-Jorgensen upper envelope to one where intersection and
    borrowing constraint are exactly included, we need to interpolate only on line-
    segments without consume all only on the left side and neighboring an intersection
    point."""
    dm = ref_m[1:] - ref_m[:-1]
    # Intersect idxs
    inter_idxs = np.where(dm == 0)[0]
    upper_forbidden_idxs = np.append(inter_idxs, inter_idxs + 1)

    # Get upper idxs of interpolation
    idxs_upper_interp = np.searchsorted(ref_m, m_grid, side="left")
    idxs_lower_interp = idxs_upper_interp - 1

    # Mark all idxs_interp which are in upper_forbidden_idxs as unsafe
    unsafe = np.isin(idxs_upper_interp, upper_forbidden_idxs)

    # Also if lower idx is consume all mark as unsafe
    unsafe |= ((ref_m[idxs_lower_interp] - ref_c[idxs_lower_interp]) == 0) & (
        (ref_m[idxs_upper_interp] - ref_c[idxs_upper_interp]) != 0
    )

    # Now linear interpolate for all and mark after unsafe as nan
    # Start with simple linear interpolation for ref_v
    weight = (m_grid - ref_m[idxs_lower_interp]) / (
        ref_m[idxs_upper_interp] - ref_m[idxs_lower_interp]
    )
    v_interp = ref_v[idxs_lower_interp] + weight * (
        ref_v[idxs_upper_interp] - ref_v[idxs_lower_interp]
    )
    v_interp[unsafe] = np.nan
    c_interp = ref_c[idxs_lower_interp] + weight * (
        ref_c[idxs_upper_interp] - ref_c[idxs_lower_interp]
    )
    c_interp[unsafe] = np.nan
    return v_interp, c_interp

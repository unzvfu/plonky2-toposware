use std::ops::Range;

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use crate::cpu::columns::{NUM_COLS_TO_CHECK, NUM_RANGE_CHECK_COLS, START_RANGE_CHECK};
use crate::lookup::permuted_cols;

pub const RANGE_MAX: u64 = 1 << 16;
// Cpu columns involved in range check
pub(crate) const RANGE_CHECK_COLUMNS: Range<usize> =
    START_RANGE_CHECK..START_RANGE_CHECK + NUM_RANGE_CHECK_COLS;
pub(crate) const RANGE_CHECK_LIMBS: Range<usize> =
    START_RANGE_CHECK..START_RANGE_CHECK + 2 * NUM_COLS_TO_CHECK;
pub(crate) const RANGE_CHECK_COL: usize = START_RANGE_CHECK + NUM_RANGE_CHECK_COLS - 1;
pub(crate) const PERMUTED_COLS: Range<usize> = RANGE_CHECK_LIMBS.end..RANGE_CHECK_COLUMNS.end;
pub fn add_range_check_rows<T: Copy, const D: usize>(cpu_rows: &mut Vec<Vec<T>>)
where
    T: RichField + Extendable<D>,
{
    let n_rows = cpu_rows.len();
    for i in 0..RANGE_MAX as usize {
        cpu_rows[RANGE_CHECK_COL][i] = T::from_canonical_usize(i);
    }
    for i in RANGE_MAX as usize..n_rows {
        cpu_rows[RANGE_CHECK_COL][i] = T::from_canonical_usize(RANGE_MAX as usize - 1);
    }

    for i in 0..NUM_COLS_TO_CHECK {
        // For each column to check, there are two associated limbs to range check. Each limb column requires a permuted column and a permuted table column.
        let low_limb_col = &cpu_rows[START_RANGE_CHECK + 3 * i];
        let (low_limb_perm, low_limb_table_perm) =
            permuted_cols(&low_limb_col, &cpu_rows[RANGE_CHECK_COL]);
        cpu_rows[RANGE_CHECK_LIMBS.end + 4 * i].copy_from_slice(&low_limb_perm);
        cpu_rows[RANGE_CHECK_LIMBS.end + 4 * i + 1].copy_from_slice(&low_limb_table_perm);
        let high_limb_col = &cpu_rows[START_RANGE_CHECK + 4 * i + 1];
        let (high_limb_perm, high_limb_table_perm) =
            permuted_cols(&high_limb_col, &cpu_rows[RANGE_CHECK_COL]);

        cpu_rows[RANGE_CHECK_LIMBS.end + 2 * i + 2].copy_from_slice(&high_limb_perm);
        cpu_rows[RANGE_CHECK_LIMBS.end + 2 * i + 3].copy_from_slice(&high_limb_table_perm);
    }
}

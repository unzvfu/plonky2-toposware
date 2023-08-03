use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;

use super::columns::{CpuColumnsView, NUM_COLS_TO_CHECK};
use super::membus::NUM_GP_CHANNELS;
use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::witness::range_check::RANGE_MAX;

pub fn eval_packed<P: PackedField>(
    lv: &CpuColumnsView<P>,
    yield_constr: &mut ConstraintConsumer<P>,
) {
    // Check that the `prover_input` limbs are correctly split into two 16-bits limbs so they can be range checked.
    let filter = lv.op.prover_input;
    let input = lv.mem_channels[NUM_GP_CHANNELS - 1].value;
    for i in 0..NUM_COLS_TO_CHECK {
        let expected_limb = lv.range_check_cols[2 * i]
            + lv.range_check_cols[2 * i + 1] * P::Scalar::from_canonical_u32(RANGE_MAX as u32);
        yield_constr.constraint(filter * (input[i] - expected_limb));
    }
}

pub fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
    lv: &CpuColumnsView<ExtensionTarget<D>>,
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    // Check that the gas limbs are correctly set in the columns to be range checked
    let filter = lv.op.prover_input;
    let input = lv.mem_channels[NUM_GP_CHANNELS - 1].value;
    let range_max = builder.constant_extension(F::Extension::from_canonical_u32(RANGE_MAX as u32));
    for i in 0..NUM_COLS_TO_CHECK {
        let expected_limb = builder.mul_add_extension(
            lv.range_check_cols[2 * i + 1],
            range_max,
            lv.range_check_cols[2 * i],
        );
        let limb_diff = builder.sub_extension(input[i], expected_limb);
        let limb_constraint = builder.mul_extension(filter, limb_diff);
        yield_constr.constraint(builder, limb_constraint);
    }
}

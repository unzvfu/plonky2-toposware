use alloc::string::String;
use alloc::vec::Vec;
use alloc::{format, vec};
use core::marker::PhantomData;
use core::ops::Range;

use crate::field::extension::algebra::ExtensionAlgebra;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::hash::poseidon::Poseidon;
use crate::iop::ext_target::{ExtensionAlgebraTarget, ExtensionTarget};
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// Poseidon MDS Gate
#[derive(Debug, Default)]
pub struct PoseidonMdsGate<
    F: RichField + Extendable<D> + Poseidon<W>,
    const W: usize,
    const D: usize,
>(PhantomData<F>);

impl<F: RichField + Extendable<D> + Poseidon<W>, const W: usize, const D: usize>
    PoseidonMdsGate<F, W, D>
{
    pub fn new() -> Self {
        Self(PhantomData)
    }

    pub fn wires_input(i: usize) -> Range<usize> {
        assert!(i < W);
        i * D..(i + 1) * D
    }

    pub fn wires_output(i: usize) -> Range<usize> {
        assert!(i < W);
        (W + i) * D..(W + i + 1) * D
    }

    // Following are methods analogous to ones in `Poseidon`, but for extension algebras.

    /// Same as `mds_row_shf` for an extension algebra of `F`.
    fn mds_row_shf_algebra(
        r: usize,
        v: &[ExtensionAlgebra<F::Extension, D>; W],
    ) -> ExtensionAlgebra<F::Extension, D> {
        debug_assert!(r < W);
        let mut res = ExtensionAlgebra::ZERO;

        for i in 0..W {
            let coeff = F::Extension::from_canonical_u64(<F as Poseidon<W>>::MDS_MATRIX_CIRC[i]);
            res += v[(i + r) % W].scalar_mul(coeff);
        }
        {
            let coeff = F::Extension::from_canonical_u64(<F as Poseidon<W>>::MDS_MATRIX_DIAG[r]);
            res += v[r].scalar_mul(coeff);
        }

        res
    }

    /// Same as `mds_row_shf_recursive` for an extension algebra of `F`.
    fn mds_row_shf_algebra_circuit(
        builder: &mut CircuitBuilder<F, D>,
        r: usize,
        v: &[ExtensionAlgebraTarget<D>; W],
    ) -> ExtensionAlgebraTarget<D> {
        debug_assert!(r < W);
        let mut res = builder.zero_ext_algebra();

        for i in 0..W {
            let coeff = builder.constant_extension(F::Extension::from_canonical_u64(
                <F as Poseidon<W>>::MDS_MATRIX_CIRC[i],
            ));
            res = builder.scalar_mul_add_ext_algebra(coeff, v[(i + r) % W], res);
        }
        {
            let coeff = builder.constant_extension(F::Extension::from_canonical_u64(
                <F as Poseidon<W>>::MDS_MATRIX_DIAG[r],
            ));
            res = builder.scalar_mul_add_ext_algebra(coeff, v[r], res);
        }

        res
    }

    /// Same as `mds_layer` for an extension algebra of `F`.
    fn mds_layer_algebra(
        state: &[ExtensionAlgebra<F::Extension, D>; W],
    ) -> [ExtensionAlgebra<F::Extension, D>; W] {
        let mut result = [ExtensionAlgebra::ZERO; W];

        for r in 0..W {
            result[r] = Self::mds_row_shf_algebra(r, state);
        }

        result
    }

    /// Same as `mds_layer_recursive` for an extension algebra of `F`.
    fn mds_layer_algebra_circuit(
        builder: &mut CircuitBuilder<F, D>,
        state: &[ExtensionAlgebraTarget<D>; W],
    ) -> [ExtensionAlgebraTarget<D>; W] {
        let mut result = [builder.zero_ext_algebra(); W];

        for r in 0..W {
            result[r] = Self::mds_row_shf_algebra_circuit(builder, r, state);
        }

        result
    }
}

impl<F: RichField + Extendable<D> + Poseidon<W>, const W: usize, const D: usize> Gate<F, D>
    for PoseidonMdsGate<F, W, D>
{
    fn id(&self) -> String {
        format!("{self:?}<WIDTH={W}>")
    }

    fn serialize(&self, _dst: &mut Vec<u8>) -> IoResult<()> {
        Ok(())
    }

    fn deserialize(_src: &mut Buffer) -> IoResult<Self> {
        Ok(PoseidonMdsGate::new())
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let inputs: [_; W] = (0..W)
            .map(|i| vars.get_local_ext_algebra(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let computed_outputs = Self::mds_layer_algebra(&inputs);

        (0..W)
            .map(|i| vars.get_local_ext_algebra(Self::wires_output(i)))
            .zip(computed_outputs)
            .flat_map(|(out, computed_out)| (out - computed_out).to_basefield_array())
            .collect()
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        let inputs: [_; W] = (0..W)
            .map(|i| vars.get_local_ext(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let computed_outputs = <F as Poseidon<W>>::mds_layer_field(&inputs);

        yield_constr.many(
            (0..W)
                .map(|i| vars.get_local_ext(Self::wires_output(i)))
                .zip(computed_outputs)
                .flat_map(|(out, computed_out)| (out - computed_out).to_basefield_array()),
        )
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let inputs: [_; W] = (0..W)
            .map(|i| vars.get_local_ext_algebra(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let computed_outputs = Self::mds_layer_algebra_circuit(builder, &inputs);

        (0..W)
            .map(|i| vars.get_local_ext_algebra(Self::wires_output(i)))
            .zip(computed_outputs)
            .flat_map(|(out, computed_out)| {
                builder
                    .sub_ext_algebra(out, computed_out)
                    .to_ext_target_array()
            })
            .collect()
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F>> {
        let gen = PoseidonMdsGenerator::<W, D> { row };
        vec![WitnessGeneratorRef::new(gen.adapter())]
    }

    fn num_wires(&self) -> usize {
        2 * D * W
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn num_constraints(&self) -> usize {
        W * D
    }
}

#[derive(Clone, Debug, Default)]
pub struct PoseidonMdsGenerator<const W: usize, const D: usize> {
    row: usize,
}

impl<F: RichField + Extendable<D> + Poseidon<W>, const W: usize, const D: usize> SimpleGenerator<F>
    for PoseidonMdsGenerator<W, D>
{
    fn id(&self) -> String {
        format!("PoseidonMdsGenerator<WIDTH={W}>")
    }

    fn dependencies(&self) -> Vec<Target> {
        (0..W)
            .flat_map(|i| {
                Target::wires_from_range(self.row, PoseidonMdsGate::<F, W, D>::wires_input(i))
            })
            .collect()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let get_local_get_target = |wire_range| ExtensionTarget::from_range(self.row, wire_range);
        let get_local_ext =
            |wire_range| witness.get_extension_target(get_local_get_target(wire_range));

        let inputs: [_; W] = (0..W)
            .map(|i| get_local_ext(PoseidonMdsGate::<F, W, D>::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let outputs = <F as Poseidon<W>>::mds_layer_field(&inputs);

        for (i, &out) in outputs.iter().enumerate() {
            out_buffer.set_extension_target(
                get_local_get_target(PoseidonMdsGate::<F, W, D>::wires_output(i)),
                out,
            );
        }
    }

    fn serialize(&self, dst: &mut Vec<u8>) -> IoResult<()> {
        dst.write_usize(self.row)
    }

    fn deserialize(src: &mut Buffer) -> IoResult<Self> {
        let row = src.read_usize()?;
        Ok(Self { row })
    }
}

#[cfg(test)]
mod tests {
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::gates::poseidon_mds::PoseidonMdsGate;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        const D: usize = 2;
        const W: usize = 12;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = PoseidonMdsGate::<F, W, D>::new();
        test_low_degree(gate)
    }

    #[test]
    fn eval_fns() -> anyhow::Result<()> {
        const D: usize = 2;
        const W: usize = 12;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = PoseidonMdsGate::<F, W, D>::new();
        test_eval_fns::<F, C, _, D>(gate)
    }
}

//! Implementation of the Poseidon2 hash function, as described in
//! <https://eprint.iacr.org/2023/323.pdf>

use alloc::vec;
use std::fmt::Debug;

use unroll::unroll_for_loops;

use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::{Field, PrimeField64};
use crate::gates::poseidon2::Poseidon2Gate;
use crate::hash::hash_types::{HashOut, RichField};
use crate::hash::hashing::{compress, hash_n_to_hash_no_pad, PlonkyPermutation};
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::{AlgebraicHasher, Hasher};

pub const SPONGE_RATE: usize = 8;
pub const SPONGE_CAPACITY: usize = 4;
pub const SPONGE_WIDTH: usize = SPONGE_RATE + SPONGE_CAPACITY;
pub const COMPRESSION_W: usize = 8;

// NB: Changing any of these values will require regenerating all of
// the precomputed constant arrays in this file.
pub const HALF_N_FULL_ROUNDS: usize = 4;
pub(crate) const N_FULL_ROUNDS_TOTAL: usize = 2 * HALF_N_FULL_ROUNDS;
pub const N_PARTIAL_ROUNDS: usize = 22;
pub const N_ROUNDS: usize = N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS;
const MAX_W: usize = 16;

/// Note that these work for the Goldilocks field, but not necessarily others. See
/// `generate_constants` about how these were generated. We include enough for a width of 12;
/// smaller widths just use a subset.
#[rustfmt::skip]
pub const ALL_ROUND_CONSTANTS: [u64; MAX_W * N_ROUNDS]  = [
    // WARNING: The AVX2 Goldilocks specialization relies on all round constants being in
    // 0..0xfffeeac900011537. If these constants are randomly regenerated, there is a ~.6% chance
    // that this condition will no longer hold.
    //
    // WARNING: If these are changed in any way, then all the
    // implementations of Poseidon2 must be regenerated. See comments
    // in `poseidon2_goldilocks.rs`.
    0x15ebea3fc73397c3, 0xd73cd9fbfe8e275c, 0x8c096bfce77f6c26, 0x4e128f68b53d8fea,
    0x29b779a36b2763f6, 0xfe2adc6fb65acd08, 0x8d2520e725ad0955, 0x1c2392b214624d2a,
    0x37482118206dcc6e, 0x2f829bed19be019a, 0x2fe298cb6f8159b0, 0x2bbad982deccdbbf,
    0xbad568b8cc60a81e, 0xb86a814265baad10, 0xbec2005513b3acb3, 0x6bf89b59a07c2a94,
    0xa25deeb835e230f5, 0x3c5bad8512b8b12a, 0x7230f73c3cb7a4f2, 0xa70c87f095c74d0f,
    0x6b7606b830bb2e80, 0x6cd467cfc4f24274, 0xfeed794df42a9b0a, 0x8cf7cf6163b7dbd3,
    0x9a6e9dda597175a0, 0xaa52295a684faf7b, 0x017b811cc3589d8d, 0x55bfb699b6181648,
    0xc2ccaf71501c2421, 0x1707950327596402, 0xdd2fcdcd42a8229f, 0x8b9d7d5b27778a21,
    0xac9a05525f9cf512, 0x2ba125c58627b5e8, 0xc74e91250a8147a5, 0xa3e64b640d5bb384,
    0xf53047d18d1f9292, 0xbaaeddacae3a6374, 0xf2d0914a808b3db1, 0x18af1a3742bfa3b0,
    0x9a621ef50c55bdb8, 0xc615f4d1cc5466f3, 0xb7fbac19a35cf793, 0xd2b1a15ba517e46d,
    0x4a290c4d7fd26f6f, 0x4f0cf1bb1770c4c4, 0x548345386cd377f5, 0x33978d2789fddd42,
    0xab78c59deb77e211, 0xc485b2a933d2be7f, 0xbde3792c00c03c53, 0xab4cefe8f893d247,
    0xc5c0e752eab7f85f, 0xdbf5a76f893bafea, 0xa91f6003e3d984de, 0x099539077f311e87,
    0x097ec52232f9559e, 0x53641bdf8991e48c, 0x2afe9711d5ed9d7c, 0xa7b13d3661b5d117,
    0x5a0e243fe7af6556, 0x1076fae8932d5f00, 0x9b53a83d434934e3, 0xed3fd595a3c0344a,
    0x28eff4b01103d100, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x60400ca3e2685a45, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x1c8636beb3389b84, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0xac1332b60e13eff0, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x2adafcc364e20f87, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x79ffc2b14054ea0b, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x3f98e4c0908f0a05, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0xcdb230bc4e8a06c4, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x1bcaf7705b152a74, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0xd9bca249a82a7470, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x91e24af19bf82551, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0xa62b43ba5cb78858, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0xb4898117472e797f, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0xb3228bca606cdaa0, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x844461051bca39c9, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0xf3411581f6617d68, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0xf7fd50646782b533, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x6ca664253c18fb48, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x2d2fcdec0886a08f, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x29da00dd799b575e, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x47d966cc3b6e1e93, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0xde884e9a17ced59e, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0xdacf46dc1c31a045, 0x5d2e3c121eb387f2, 0x51f8b0658b124499, 0x1e7dbd1daa72167d,
    0x8275015a25c55b88, 0xe8521c24ac7a70b3, 0x6521d121c40b3f67, 0xac12de797de135b0,
    0xafa28ead79f6ed6a, 0x685174a7a8d26f0b, 0xeff92a08d35d9874, 0x3058734b76dd123a,
    0xfa55dcfba429f79c, 0x559294d4324c7728, 0x7a770f53012dc178, 0xedd8f7c408f3883b,
    0x39b533cf8d795fa5, 0x160ef9de243a8c0a, 0x431d52da6215fe3f, 0x54c51a2a2ef6d528,
    0x9b13892b46ff9d16, 0x263c46fcee210289, 0xb738c96d25aabdc4, 0x5c33a5203996d38f,
    0x2626496e7c98d8dd, 0xc669e0a52785903a, 0xaecde726c8ae1f47, 0x039343ef3a81e999,
    0x2615ceaf044a54f9, 0x7e41e834662b66e1, 0x4ca5fd4895335783, 0x64b334d02916f2b0,
    0x87268837389a6981, 0x034b75bcb20a6274, 0x58e658296cc2cd6e, 0xe2d0f759acc31df4,
    0x81a652e435093e20, 0x0b72b6e0172eaf47, 0x4aec43cec577d66d, 0xde78365b028a84e6,
    0x444e19569adc0ee4, 0x942b2451fa40d1da, 0xe24506623ea5bd6c, 0x082854bf2ef7c743,
    0x69dbbc566f59d62e, 0x248c38d02a7b5cb2, 0x4f4e8f8c09d15edb, 0xd96682f188d310cf,
    0x6f9a25d56818b54c, 0xb6cefed606546cd9, 0x5bc07523da38a67b, 0x7df5a3c35b8111cf,
    0xaaa2cc5d4db34bb0, 0x9e673ff22a4653f8, 0xbd8b278d60739c62, 0xe10d20f6925b8815,
    0xf6c87b91dd4da2bf, 0xfed623e2f71b6f1a, 0xa0f02fa52a94d0d3, 0xbb5794711b39fa16,
    0xd3b94fba9d005c7f, 0x15a26e89fad946c9, 0xf3cb87db8a67cf49, 0x400d2bf56aa2a577,
];

// Applying cheap 4x4 MDS matrix to each 4-element part of the state
// The matrix in this case is:
// M_4 =
// [5   7   1   3]
// [4   6   1   1]
// [1   3   5   7]
// [1   1   4   6]
// The computation is shown in more detail in https://tosc.iacr.org/index.php/ToSC/article/view/888/839, Figure 13 (M_{4,4}^{8,4} with alpha = 2)
#[inline(always)]
fn matrix_mul_block(x: &mut [u64]) {
    let mut t_2 = x[1];
    let mut t_3 = x[3];
    let t_0 = x[0] + t_2;
    let t_1 = x[2] + t_3;
    t_2 = (t_2 << 1) + t_1;
    t_3 = (t_3 << 1) + t_0;
    let t_4 = (t_1 << 2) + t_3;
    let t_5 = (t_0 << 2) + t_2;
    let t_6 = t_3 + t_5;
    let t_7 = t_2 + t_4;
    x[0] = t_6;
    x[1] = t_5;
    x[2] = t_7;
    x[3] = t_4;
}

#[inline(always)]
#[unroll_for_loops]
fn combine_m4_prods(x: &mut [u64], s: [u64; 4]) {
    for i in 0..4 {
        x[i] += s[i];
    }
}

pub trait Poseidon2<const W: usize>: PrimeField64 {
    const T4: usize = W / 4;

    /// We only need INTERNAL_MATRIX_DIAG_M_1 here, specifying the diagonal - 1 of the internal matrix
    const INTERNAL_MATRIX_DIAG_M_1: [u64; W];

    // Apply external matrix to a state vector with at most 32 bits elements.
    // This is employed to compute product with external matrix employing only native u64 integer
    // arithmetic for efficiency
    #[inline(always)]
    #[unroll_for_loops]
    fn external_matrix_with_u64_arithmetic(x: &mut [u64; W]) {
        for i in 0..Self::T4 {
            matrix_mul_block(&mut x[i * 4..(i + 1) * 4]);
        }

        // TODO Robin: change this

        // Applying second cheap matrix
        // This completes the multiplication by
        // M_E, which, in the case W=12, would be
        // [2*M_4    M_4    M_4]
        // [  M_4  2*M_4    M_4]
        // [  M_4    M_4  2*M_4]
        // using the results with M_4 obtained above.

        // compute vector to be later used to combine M_4 multiplication results with current state x;
        // this operation is performed without loops for efficiency
        debug_assert_eq!(Self::T4, 3);
        let mut s: [u64; 4] = x[0..4].try_into().unwrap();
        for i in 1..Self::T4 {
            s[0] += x[4 * i];
            s[1] += x[4 * i + 1];
            s[2] += x[4 * i + 2];
            s[3] += x[4 * i + 3];
        }

        for i in 0..Self::T4 {
            combine_m4_prods(&mut x[i * 4..(i + 1) * 4], s);
        }
    }

    /// Compute the product between the state vector and the matrix employed in full rounds of
    /// the permutation
    #[inline(always)]
    #[unroll_for_loops]
    fn external_matrix(state: &mut [Self; W]) {
        let mut state_l = [0u64; W];
        let mut state_h = [0u64; W];
        for i in 0..W {
            let state_u64 = state[i].to_noncanonical_u64();
            state_h[i] = state_u64 >> 32;
            state_l[i] = (state_u64 as u32) as u64;
        }
        Self::external_matrix_with_u64_arithmetic(&mut state_l);
        Self::external_matrix_with_u64_arithmetic(&mut state_h);

        for i in 0..W {
            let (state_u64, carry) = state_l[i].overflowing_add(state_h[i] << 32);
            state[i] =
                Self::from_noncanonical_u96((state_u64, (state_h[i] >> 32) as u32 + carry as u32));
        }
    }

    /// Same as `external_matrix` for field extensions of `Self`.
    fn external_matrix_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; W],
    ) {
        // Applying cheap 4x4 MDS matrix to each 4-element part of the state
        for i in 0..Self::T4 {
            let start_index = i * 4;
            let mut t_0 = state[start_index];
            t_0 += state[start_index + 1];
            let mut t_1 = state[start_index + 2];
            t_1 += state[start_index + 3];
            let mut t_2 = state[start_index + 1];
            t_2 = t_2 + t_2;
            t_2 += t_1;
            let mut t_3 = state[start_index + 3];
            t_3 = t_3 + t_3;
            t_3 += t_0;
            let mut t_4 = t_1;
            t_4 = F::from_canonical_u64(4) * t_4;
            t_4 += t_3;
            let mut t_5 = t_0;
            t_5 = F::from_canonical_u64(4) * t_5;
            t_5 += t_2;
            let mut t_6 = t_3;
            t_6 += t_5;
            let mut t_7 = t_2;
            t_7 += t_4;
            state[start_index] = t_6;
            state[start_index + 1] = t_5;
            state[start_index + 2] = t_7;
            state[start_index + 3] = t_4;
        }

        // Applying second cheap matrix
        let mut stored = [F::ZERO; 4];
        for l in 0..4 {
            stored[l] = state[l];
            for j in 1..Self::T4 {
                stored[l] += state[4 * j + l];
            }
        }
        for i in 0..W {
            state[i] += stored[i % 4];
        }
    }

    /// Recursive version of `external_matrix`.
    fn external_matrix_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &mut [ExtensionTarget<D>; W],
    ) where
        Self: RichField + Extendable<D>,
    {
        // In contrast to the Poseidon circuit, we *may not need* PoseidonMdsGate, because the number of constraints will fit regardless
        let four = Self::from_canonical_u64(0x4);

        // Applying cheap 4x4 MDS matrix to each 4-element part of the state
        for i in 0..<Self as Poseidon2<W>>::T4 {
            let start_index = i * 4;
            let mut t_0 = state[start_index];
            t_0 = builder.add_extension(t_0, state[start_index + 1]);
            let mut t_1 = state[start_index + 2];
            t_1 = builder.add_extension(t_1, state[start_index + 3]);
            let mut t_2 = state[start_index + 1];
            t_2 = builder.add_extension(t_2, t_2); // Double
            t_2 = builder.add_extension(t_2, t_1);
            let mut t_3 = state[start_index + 3];
            t_3 = builder.add_extension(t_3, t_3); // Double
            t_3 = builder.add_extension(t_3, t_0);
            let mut t_4 = t_1;
            t_4 = builder.mul_const_extension(four, t_4); // times 4
            t_4 = builder.add_extension(t_4, t_3);
            let mut t_5 = t_0;
            t_5 = builder.mul_const_extension(four, t_5); // times 4
            t_5 = builder.add_extension(t_5, t_2);
            let mut t_6 = t_3;
            t_6 = builder.add_extension(t_6, t_5);
            let mut t_7 = t_2;
            t_7 = builder.add_extension(t_7, t_4);
            state[start_index] = t_6;
            state[start_index + 1] = t_5;
            state[start_index + 2] = t_7;
            state[start_index + 3] = t_4;
        }

        // Applying second cheap matrix
        let mut stored = [builder.zero_extension(); 4];
        for l in 0..4 {
            stored[l] = state[l];
            for j in 1..<Self as Poseidon2<W>>::T4 {
                stored[l] = builder.add_extension(stored[l], state[4 * j + l]);
            }
        }
        for i in 0..W {
            state[i] = builder.add_extension(state[i], stored[i % 4]);
        }
    }

    /// Compute the product between the state vector and the matrix employed in partial rounds of
    /// the permutation
    #[inline(always)]
    #[unroll_for_loops]
    fn internal_matrix(state: &mut [Self; W]) {
        // This computes the mutliplication with the matrix
        // M_I =
        // [r_1     1   1   ...     1]
        // [  1   r_2   1   ...     1]
        // ...
        // [  1     1   1   ...   r_t]
        // for pseudo-random values r_1, r_2, ..., r_t. Note that for efficiency in Self::INTERNAL_MATRIX_DIAG_M_1 only r_1 - 1, r_2 - 1, ..., r_t - 1 are stored
        // Compute input sum
        let f_sum = Self::from_noncanonical_u128(
            state
                .iter()
                .fold(0u128, |sum, el| sum + el.to_noncanonical_u64() as u128),
        );
        // Add sum + diag entry * element to each element
        for i in 0..W {
            state[i] *= Self::from_canonical_u64(Self::INTERNAL_MATRIX_DIAG_M_1[i]);
            state[i] += f_sum;
        }
    }

    /// Same as `internal_matrix` for field extensions of `Self`.
    fn internal_matrix_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; W],
    ) {
        // Compute input sum
        let sum = state.iter().fold(F::ZERO, |sum, el| sum + *el);
        // Add sum + diag entry * element to each element
        for i in 0..state.len() {
            state[i] *= F::from_canonical_u64(Self::INTERNAL_MATRIX_DIAG_M_1[i]);
            state[i] += sum;
        }
    }

    /// Recursive version of `internal_matrix`.
    fn internal_matrix_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &mut [ExtensionTarget<D>; W],
    ) where
        Self: RichField + Extendable<D>,
    {
        // Compute input sum
        let mut sum = state[0];
        for i in 1..state.len() {
            sum = builder.add_extension(sum, state[i]);
        }
        // Add sum + diag entry * element to each element
        for i in 0..state.len() {
            // Computes `C * x + y`
            state[i] = builder.mul_const_add_extension(
                Self::from_canonical_u64(<Self as Poseidon2<W>>::INTERNAL_MATRIX_DIAG_M_1[i]),
                state[i],
                sum,
            );
        }
    }

    /// Add round constant to `state` for round `round_ctr`
    #[inline(always)]
    #[unroll_for_loops]
    fn constant_layer(state: &mut [Self; W], round_ctr: usize) {
        for i in 0..12 {
            if i < W {
                let round_constant = ALL_ROUND_CONSTANTS[i + W * round_ctr];
                unsafe {
                    state[i] = state[i].add_canonical_u64(round_constant);
                }
            }
        }
    }

    /// Same as `constant_layer` for field extensions of `Self`.
    fn constant_layer_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; W],
        round_ctr: usize,
    ) {
        for i in 0..W {
            state[i] += F::from_canonical_u64(ALL_ROUND_CONSTANTS[i + W * round_ctr]);
        }
    }

    /// Recursive version of `constant_layer`.
    fn constant_layer_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &mut [ExtensionTarget<D>; W],
        round_ctr: usize,
    ) where
        Self: RichField + Extendable<D>,
    {
        for i in 0..W {
            let c = ALL_ROUND_CONSTANTS[i + W * round_ctr];
            let c = Self::Extension::from_canonical_u64(c);
            let c = builder.constant_extension(c);
            state[i] = builder.add_extension(state[i], c);
        }
    }

    /// Apply the sbox to a single state element
    #[inline(always)]
    fn sbox_monomial<F: FieldExtension<D, BaseField = Self>, const D: usize>(x: F) -> F {
        // x |--> x^7
        let x2 = x.square();
        let x4 = x2.square();
        let x3 = x * x2;
        x3 * x4
    }

    /// Recursive version of `sbox_monomial`.
    fn sbox_monomial_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        x: ExtensionTarget<D>,
    ) -> ExtensionTarget<D>
    where
        Self: RichField + Extendable<D>,
    {
        // x |--> x^7
        builder.exp_u64_extension(x, 7)
    }

    /// Apply the sbox-layer to the whole state of the permutation
    #[inline(always)]
    #[unroll_for_loops]
    fn sbox_layer(state: &mut [Self; W]) {
        for i in 0..12 {
            if i < W {
                state[i] = Self::sbox_monomial(state[i]);
            }
        }
    }

    /// Same as `sbox_layer` for field extensions of `Self`.
    fn sbox_layer_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; W],
    ) {
        for i in 0..W {
            state[i] = Self::sbox_monomial(state[i]);
        }
    }

    /// Recursive version of `sbox_layer`.
    fn sbox_layer_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &mut [ExtensionTarget<D>; W],
    ) where
        Self: RichField + Extendable<D>,
    {
        for i in 0..W {
            state[i] = <Self as Poseidon2<W>>::sbox_monomial_circuit(builder, state[i]);
        }
    }

    /// Apply half of the overall full rounds of the permutation. It can be employed to perform
    /// both the first and the second half of the full rounds, depending on the value of `round_ctr`
    #[inline]
    fn full_rounds(state: &mut [Self; W], round_ctr: &mut usize) {
        for _ in 0..HALF_N_FULL_ROUNDS {
            Self::constant_layer(state, *round_ctr);
            Self::sbox_layer(state);
            Self::external_matrix(state);
            *round_ctr += 1;
        }
    }

    /// Apply the partial rounds of the permutation
    #[inline]
    fn partial_rounds(state: &mut [Self; W], round_ctr: &mut usize) {
        let mut constant_counter = HALF_N_FULL_ROUNDS * W;
        for _ in 0..N_PARTIAL_ROUNDS {
            unsafe {
                state[0] = state[0].add_canonical_u64(ALL_ROUND_CONSTANTS[constant_counter]);
                constant_counter += W;
            }
            state[0] = Self::sbox_monomial(state[0]);
            Self::internal_matrix(state);
        }
        *round_ctr += N_PARTIAL_ROUNDS;
    }
    /// Apply the entire Poseidon2 permutation to `input`
    ///
    /// ```rust
    /// use plonky2::field::goldilocks_field::GoldilocksField as F;
    /// use plonky2::field::types::Sample;
    /// use poseidon2_plonky2::poseidon2_hash::Poseidon2;
    ///
    /// let output = F::poseidon2(F::rand_array());
    /// ```
    #[inline]
    fn poseidon2(input: [Self; W]) -> [Self; W] {
        let mut state = input;
        let mut round_ctr = 0;

        // First external matrix
        Self::external_matrix(&mut state);

        Self::full_rounds(&mut state, &mut round_ctr);
        Self::partial_rounds(&mut state, &mut round_ctr);
        Self::full_rounds(&mut state, &mut round_ctr);
        debug_assert_eq!(round_ctr, N_ROUNDS);

        state
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Poseidon2Permutation<T, const R: usize, const W: usize> {
    state: [T; W],
}

impl<T: Copy + Default, const R: usize, const W: usize> Default for Poseidon2Permutation<T, R, W> {
    fn default() -> Self {
        Self {
            state: [T::default(); W],
        }
    }
}

impl<T: Eq, const R: usize, const W: usize> Eq for Poseidon2Permutation<T, R, W> {}

impl<T, const R: usize, const W: usize> AsRef<[T]> for Poseidon2Permutation<T, R, W> {
    fn as_ref(&self) -> &[T] {
        &self.state
    }
}

trait Permuter<const W: usize>: Sized {
    fn permute(input: [Self; W]) -> [Self; W];
}

impl<const W: usize, F: Poseidon2<W>> Permuter<W> for F {
    fn permute(input: [Self; W]) -> [Self; W] {
        <F as Poseidon2<W>>::poseidon2(input)
    }
}

impl<const W: usize> Permuter<W> for Target {
    fn permute(_input: [Self; W]) -> [Self; W] {
        panic!("Call `permute_swapped()` instead of `permute()`");
    }
}

impl<
        const R: usize,
        const W: usize,
        T: Copy + Debug + Default + Eq + Permuter<W> + Send + Sync,
    > PlonkyPermutation<T> for Poseidon2Permutation<T, R, W>
{
    const RATE: usize = R;
    const WIDTH: usize = W;

    fn new<I: IntoIterator<Item = T>>(elts: I) -> Self {
        let mut perm = Self {
            state: [T::default(); W],
        };
        perm.set_from_iter(elts, 0);
        perm
    }

    fn set_elt(&mut self, elt: T, idx: usize) {
        self.state[idx] = elt;
    }

    fn set_from_slice(&mut self, elts: &[T], start_idx: usize) {
        let begin = start_idx;
        let end = start_idx + elts.len();
        self.state[begin..end].copy_from_slice(elts);
    }

    fn set_from_iter<I: IntoIterator<Item = T>>(&mut self, elts: I, start_idx: usize) {
        for (s, e) in self.state[start_idx..].iter_mut().zip(elts) {
            *s = e;
        }
    }

    fn permute(&mut self) {
        self.state = T::permute(self.state);
    }

    fn squeeze(&self) -> &[T] {
        &self.state[..Self::RATE]
    }
}

/// Poseidon2 hash function.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Poseidon2Hash;
impl<F: RichField> Hasher<F> for Poseidon2Hash {
    const HASH_SIZE: usize = 4 * 8;
    type Hash = HashOut<F>;
    type Permutation = Poseidon2Permutation<F, SPONGE_RATE, SPONGE_WIDTH>;

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        hash_n_to_hash_no_pad::<F, Self::Permutation>(input)
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        compress::<F, Self::Permutation>(left, right)
    }
}

impl<F: RichField> AlgebraicHasher<F> for Poseidon2Hash {
    type AlgebraicPermutation = Poseidon2Permutation<Target, SPONGE_RATE, SPONGE_WIDTH>;

    fn permute_swapped<const D: usize>(
        inputs: Self::AlgebraicPermutation,
        swap: BoolTarget,
        builder: &mut CircuitBuilder<F, D>,
    ) -> Self::AlgebraicPermutation
    where
        F: RichField + Extendable<D>,
    {
        let gate_type = Poseidon2Gate::<F, SPONGE_WIDTH, D>::new();
        let gate = builder.add_gate(gate_type, vec![]);

        let swap_wire = Poseidon2Gate::<F, SPONGE_WIDTH, D>::WIRE_SWAP;
        let swap_wire = Target::wire(gate, swap_wire);
        builder.connect(swap.target, swap_wire);

        // Route input wires.
        let inputs = inputs.as_ref();
        for i in 0..SPONGE_WIDTH {
            let in_wire = Poseidon2Gate::<F, SPONGE_WIDTH, D>::wire_input(i);
            let in_wire = Target::wire(gate, in_wire);
            builder.connect(inputs[i], in_wire);
        }

        // Collect output wires.
        Self::AlgebraicPermutation::new(
            (0..SPONGE_WIDTH)
                .map(|i| Target::wire(gate, Poseidon2Gate::<F, SPONGE_WIDTH, D>::wire_output(i))),
        )
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use crate::field::types::Field;
    use crate::hash::poseidon2::Poseidon2;

    pub(crate) fn check_test_vectors<const W: usize, F: Field>(
        test_vectors: Vec<([u64; W], [u64; W])>,
    ) where
        F: Poseidon2<W>,
    {
        for (input_, expected_output_) in test_vectors.into_iter() {
            let mut input = [F::ZERO; W];
            for i in 0..W {
                input[i] = F::from_canonical_u64(input_[i]);
            }
            let output = F::poseidon2(input);
            for i in 0..W {
                let ex_output = F::from_canonical_u64(expected_output_[i]);
                assert_eq!(output[i], ex_output);
            }
        }
    }
}

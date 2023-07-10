//! Implementations for Poseidon2 over the Goldilocks field.

use super::poseidon2::Poseidon2;
use crate::field::goldilocks_field::GoldilocksField;

#[rustfmt::skip]
impl Poseidon2<12> for GoldilocksField {
    // We only need INTERNAL_MATRIX_DIAG_M_1 here, specifying the diagonal - 1 of the internal matrix

    const INTERNAL_MATRIX_DIAG_M_1: [u64; 12]  = [
        0xcf6f77ac16722af9, 0x3fd4c0d74672aebc, 0x9b72bf1c1c3d08a8, 0xe4940f84b71e4ac2,
        0x61b27b077118bc72, 0x2efd8379b8e661e2, 0x858edcf353df0341, 0x2d9c20affb5c4516,
        0x5120143f0695defb, 0x62fc898ae34a5c5b, 0xa3d9560c99123ed2, 0x98fd739d8e7fc933,
    ];

    #[cfg(all(target_arch="aarch64", target_feature="neon"))]
    #[inline(always)]
    fn sbox_layer(state: &mut [Self; 12]) {
         unsafe {
             crate::hash::arch::aarch64::poseidon_goldilocks_neon::sbox_layer(state);
         }
    }
}
#[cfg(test)]
mod tests {
    use crate::field::goldilocks_field::GoldilocksField as F;
    use crate::hash::poseidon2::test_helpers::check_test_vectors;

    #[test]
    fn test_vectors() {
        // Test inputs are:
        // 1. range 0..WIDTH
        // 2. random elements of GoldilocksField.
        // expected output calculated with (modified) hadeshash reference implementation.

        #[rustfmt::skip]
        let test_vectors12: Vec<([u64; 12], [u64; 12])> = vec![
            ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ],
             [0xed3dbcc4ff1e8d33, 0xfb85eac6ac91a150, 0xd41e1e237ed3e2ef, 0x5e289bf0a4c11897,
                 0x4398b20f93e3ba6b, 0x5659a48ffaf2901d, 0xe44d81e89a88f8ae, 0x08efdb285f8c3dbc,
                 0x294ab7503297850e, 0xa11c61f4870b9904, 0xa6855c112cc08968, 0x17c6d53d2fb3e8c1, ]),
        ];

        check_test_vectors::<12, F>(test_vectors12);
    }
}

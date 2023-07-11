use std::collections::HashMap;

use criterion::{criterion_group, criterion_main, Criterion};
use eth_trie_utils::nibbles::Nibbles;
use eth_trie_utils::partial_trie::{HashedPartialTrie, PartialTrie};
use ethereum_types::{Address, U256};
use hex_literal::hex;
use keccak_hash::keccak;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::plonk::config::PoseidonGoldilocksConfig;
use plonky2::util::timing::TimingTree;
use plonky2_evm::all_stark::AllStark;
use plonky2_evm::config::StarkConfig;
use plonky2_evm::generation::mpt::AccountRlp;
use plonky2_evm::generation::{GenerationInputs, TrieInputs};
use plonky2_evm::proof::BlockMetadata;
use plonky2_evm::prover::prove;
use plonky2_evm::Node;

pub(crate) fn bench_prover(c: &mut Criterion) {
    type F = GoldilocksField;
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;

    let all_stark = AllStark::<F, D>::default();
    let config = StarkConfig::standard_fast_config();

    let beneficiary = hex!("deadbeefdeadbeefdeadbeefdeadbeefdeadbeef");
    let sender = hex!("2c7536e3605d9c16a7a3d7b1898e529396a65c23");

    let sender_state_key = keccak(sender);

    let sender_nibbles = Nibbles::from_bytes_be(sender_state_key.as_bytes()).unwrap();

    let sender_account_before = AccountRlp {
        nonce: 5.into(),
        balance: eth_to_wei(100_000.into()),
        storage_root: HashedPartialTrie::from(Node::Empty).hash(),
        code_hash: keccak([]),
    };

    let state_trie_before = Node::Leaf {
        nibbles: sender_nibbles,
        value: rlp::encode(&sender_account_before).to_vec(),
    }
    .into();
    let tries_before = TrieInputs {
        state_trie: state_trie_before,
        transactions_trie: HashedPartialTrie::from(Node::Empty),
        receipts_trie: HashedPartialTrie::from(Node::Empty),
        storage_tries: vec![],
    };

    // Generated using a little py-evm script.
    let txn = hex!("f861050a8255f094a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0648242421ba02c89eb757d9deeb1f5b3859a9d4d679951ef610ac47ad4608dc142beb1b7e313a05af7e9fbab825455d36c36c7f4cfcafbeafa9a77bdff936b52afb36d4fe4bcdd");

    let block_metadata = BlockMetadata {
        block_beneficiary: Address::from(beneficiary),
        block_timestamp: 0x03e8.into(),
        block_number: 1.into(),
        block_difficulty: 0x020000.into(),
        block_gaslimit: 0xff112233445566u64.into(),
        block_chain_id: 1.into(),
        block_base_fee: 0xa.into(),
    };

    let mut contract_code = HashMap::new();
    contract_code.insert(keccak(vec![]), vec![]);

    let inputs = GenerationInputs {
        signed_txns: vec![txn.to_vec()],
        tries: tries_before,
        contract_code,
        block_metadata,
        addresses: vec![],
    };

    c.bench_function("prover", |b| {
        b.iter(|| {
            prove::<F, C, D>(
                &all_stark,
                &config,
                inputs.clone(),
                &mut TimingTree::new("dummy", log::Level::Info),
            )
        })
    });
}
criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_prover
}
criterion_main!(benches);

fn eth_to_wei(eth: U256) -> U256 {
    // 1 ether = 10^18 wei.
    eth * U256::from(10).pow(18.into())
}

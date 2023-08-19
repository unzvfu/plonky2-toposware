use itertools::izip;
use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;

use super::kernel::stack;
use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::cpu::columns::ops::OpsColumnsView;
use crate::cpu::columns::CpuColumnsView;
use crate::cpu::membus::NUM_GP_CHANNELS;
use crate::memory::segments::Segment;
use crate::memory::VALUE_LIMBS;

#[derive(Clone, Copy)]
struct StackBehavior {
    num_pops: usize,
    pushes: bool,
    new_top_stack_channel: Option<usize>,
    disable_other_channels: bool,
}

const BASIC_BINARY_OP: Option<StackBehavior> = Some(StackBehavior {
    num_pops: 2,
    pushes: true,
    new_top_stack_channel: Some(NUM_GP_CHANNELS - 1),
    disable_other_channels: true,
});
const BASIC_TERNARY_OP: Option<StackBehavior> = Some(StackBehavior {
    num_pops: 3,
    pushes: true,
    new_top_stack_channel: Some(NUM_GP_CHANNELS - 1),
    disable_other_channels: true,
});

// AUDITORS: If the value below is `None`, then the operation must be manually checked to ensure
// that every general-purpose memory channel is either disabled or has its read flag and address
// propertly constrained. The same applies  when `disable_other_channels` is set to `false`,
// except the first `num_pops` and the last `pushes as usize` channels have their read flag and
// address constrained automatically in this file.
// If `new_top_stack_channel` contains a value, then this file will automatically constrain it
// (typically if an instruction pops and pushes, and you know where the new top of the stack
// will be). If it is set to `none`, the new top of the stack must be constrained manually by the
// operation. Note that instructions which only pop also have it set to `None`, even if we constrain
// the next top in this file: their logic is special and depends on stack_len.
const STACK_BEHAVIORS: OpsColumnsView<Option<StackBehavior>> = OpsColumnsView {
    add: BASIC_BINARY_OP,
    mul: BASIC_BINARY_OP,
    sub: BASIC_BINARY_OP,
    div: BASIC_BINARY_OP,
    mod_: BASIC_BINARY_OP,
    addmod: BASIC_TERNARY_OP,
    mulmod: BASIC_TERNARY_OP,
    addfp254: BASIC_BINARY_OP,
    mulfp254: BASIC_BINARY_OP,
    subfp254: BASIC_BINARY_OP,
    submod: BASIC_TERNARY_OP,
    lt: BASIC_BINARY_OP,
    gt: BASIC_BINARY_OP,
    eq: Some(StackBehavior {
        num_pops: 2,
        pushes: true,
        new_top_stack_channel: Some(1),
        disable_other_channels: true,
    }),
    iszero: Some(StackBehavior {
        num_pops: 1,
        pushes: true,
        new_top_stack_channel: Some(1),
        disable_other_channels: true,
    }),
    logic_op: BASIC_BINARY_OP,
    not: Some(StackBehavior {
        num_pops: 1,
        pushes: true,
        new_top_stack_channel: Some(NUM_GP_CHANNELS - 1),
        disable_other_channels: true,
    }),
    byte: BASIC_BINARY_OP,
    shl: Some(StackBehavior {
        num_pops: 2,
        pushes: true,
        new_top_stack_channel: Some(NUM_GP_CHANNELS - 1),
        disable_other_channels: false,
    }),
    shr: Some(StackBehavior {
        num_pops: 2,
        pushes: true,
        new_top_stack_channel: Some(NUM_GP_CHANNELS - 1),
        disable_other_channels: false,
    }),
    keccak_general: Some(StackBehavior {
        num_pops: 4,
        pushes: true,
        new_top_stack_channel: Some(NUM_GP_CHANNELS - 1),
        disable_other_channels: true,
    }),
    prover_input: None, // TODO
    pop: Some(StackBehavior {
        num_pops: 1,
        pushes: false,
        new_top_stack_channel: None,
        disable_other_channels: true,
    }),
    jump: Some(StackBehavior {
        num_pops: 1,
        pushes: false,
        new_top_stack_channel: None,
        disable_other_channels: false,
    }),
    jumpi: Some(StackBehavior {
        num_pops: 2,
        pushes: false,
        new_top_stack_channel: None,
        disable_other_channels: false,
    }),
    pc: Some(StackBehavior {
        num_pops: 0,
        pushes: true,
        new_top_stack_channel: None,
        disable_other_channels: true,
    }),
    jumpdest: Some(StackBehavior {
        num_pops: 0,
        pushes: false,
        new_top_stack_channel: None,
        disable_other_channels: true,
    }),
    push0: Some(StackBehavior {
        num_pops: 0,
        pushes: true,
        new_top_stack_channel: None,
        disable_other_channels: true,
    }),
    push: None, // TODO
    dup: None,
    swap: None,
    get_context: Some(StackBehavior {
        num_pops: 0,
        pushes: true,
        new_top_stack_channel: None,
        disable_other_channels: true,
    }),
    set_context: None, // SET_CONTEXT is special since it involves the old and the new stack.
    exit_kernel: Some(StackBehavior {
        num_pops: 1,
        pushes: false,
        new_top_stack_channel: None,
        disable_other_channels: true,
    }),
    mload_general: Some(StackBehavior {
        num_pops: 3,
        pushes: true,
        new_top_stack_channel: Some(2),
        disable_other_channels: false,
    }),
    mstore_general: Some(StackBehavior {
        num_pops: 4,
        pushes: false,
        new_top_stack_channel: None,
        disable_other_channels: false,
    }),
    syscall: Some(StackBehavior {
        num_pops: 0,
        pushes: true,
        new_top_stack_channel: None,
        disable_other_channels: false,
    }),
    exception: Some(StackBehavior {
        num_pops: 0,
        pushes: true,
        new_top_stack_channel: None,
        disable_other_channels: false,
    }),
};

fn eval_packed_one<P: PackedField>(
    lv: &CpuColumnsView<P>,
    nv: &CpuColumnsView<P>,
    filter: P,
    stack_behavior: StackBehavior,
    yield_constr: &mut ConstraintConsumer<P>,
) {
    // If you have pops.
    if stack_behavior.num_pops > 0 {
        for i in 1..stack_behavior.num_pops {
            let channel = lv.mem_channels[i - 1];

            yield_constr.constraint(filter * (channel.used - P::ONES));
            yield_constr.constraint(filter * (channel.is_read - P::ONES));

            yield_constr.constraint(filter * (channel.addr_context - lv.context));
            yield_constr.constraint(
                filter
                    * (channel.addr_segment - P::Scalar::from_canonical_u64(Segment::Stack as u64)),
            );
            // Remember that the first read (`i == 1`) is for the second stack element at `stack[stack_len - 1]`.
            let addr_virtual = lv.stack_len - P::Scalar::from_canonical_usize(i + 1);
            yield_constr.constraint(filter * (channel.addr_virtual - addr_virtual));
        }

        // If you also push, you don't need to read the new top of the stack.
        // If you don't:
        // - if the stack isn't empty after the pops, you read the new top from an extra pop.
        // - if not, the extra read is disabled.
        if !stack_behavior.pushes {
            // If stack_len != N...
            let new_filter =
                (lv.stack_len - P::Scalar::from_canonical_usize(stack_behavior.num_pops)) * filter;
            // Read an extra element.
            let channel = lv.mem_channels[stack_behavior.num_pops - 1];
            yield_constr.constraint(new_filter * (channel.used - P::ONES));
            yield_constr.constraint(new_filter * (channel.is_read - P::ONES));
            yield_constr.constraint(new_filter * (channel.addr_context - lv.context));
            yield_constr.constraint(
                new_filter
                    * (channel.addr_segment - P::Scalar::from_canonical_u64(Segment::Stack as u64)),
            );
            let addr_virtual =
                lv.stack_len - P::Scalar::from_canonical_usize(stack_behavior.num_pops + 1);
            yield_constr.constraint(new_filter * (channel.addr_virtual - addr_virtual));
            // This element is the new top of the stack.
            // Doesn't apply to the last row!
            for (limb_ch, limb_top) in channel.value.iter().zip(nv.stack_top.iter()) {
                yield_constr.constraint_transition(new_filter * (*limb_ch - *limb_top));
            }

            // TODO: disable channel if stack_len == N.
        }
    }
    // If the op only pushes, you only need to constrain the top of the stack if the stack isn't empty.
    else if stack_behavior.pushes {
        // If len > 0...
        let new_filter = lv.stack_len * filter;
        // You write the previous top of the stack in memory, in the last channel.
        let channel = lv.mem_channels[NUM_GP_CHANNELS - 1];
        yield_constr.constraint(new_filter * (channel.used - P::ONES));
        yield_constr.constraint(new_filter * channel.is_read);
        yield_constr.constraint(new_filter * (channel.addr_context - lv.context));
        yield_constr.constraint(
            new_filter
                * (channel.addr_segment - P::Scalar::from_canonical_u64(Segment::Stack as u64)),
        );
        let addr_virtual = lv.stack_len - P::ONES;
        yield_constr.constraint(new_filter * (channel.addr_virtual - addr_virtual));
    }

    // Maybe constrain next stack_top.
    if let Some(next_top_ch) = stack_behavior.new_top_stack_channel {
        for (limb_ch, limb_top) in lv.mem_channels[next_top_ch]
            .value
            .iter()
            .zip(nv.stack_top.iter())
        {
            yield_constr.constraint(filter * (*limb_ch - *limb_top));
        }
    }

    // Unused channels
    if stack_behavior.disable_other_channels {
        for i in stack_behavior.num_pops..NUM_GP_CHANNELS - (stack_behavior.pushes as usize) {
            let channel = lv.mem_channels[i];
            yield_constr.constraint(filter * channel.used);
        }
    }
}

pub fn eval_packed<P: PackedField>(
    lv: &CpuColumnsView<P>,
    nv: &CpuColumnsView<P>,
    yield_constr: &mut ConstraintConsumer<P>,
) {
    for (op, stack_behavior) in izip!(lv.op.into_iter(), STACK_BEHAVIORS.into_iter()) {
        if let Some(stack_behavior) = stack_behavior {
            eval_packed_one(lv, nv, op, stack_behavior, yield_constr);
        }
    }
}

fn eval_ext_circuit_one<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
    lv: &CpuColumnsView<ExtensionTarget<D>>,
    filter: ExtensionTarget<D>,
    stack_behavior: StackBehavior,
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    // let num_operands = stack_behavior.num_pops + (stack_behavior.pushes as usize);
    // assert!(num_operands <= NUM_GP_CHANNELS);

    // If you have pops.
    if stack_behavior.num_pops > 0 {
        for i in 1..stack_behavior.num_pops {
            let channel = lv.mem_channels[i - 1];

            {
                let constr = builder.mul_sub_extension(filter, channel.used, filter);
                yield_constr.constraint(builder, constr);
            }
            {
                let constr = builder.mul_sub_extension(filter, channel.is_read, filter);
                yield_constr.constraint(builder, constr);
            }
            {
                let diff = builder.sub_extension(channel.addr_context, lv.context);
                let constr = builder.mul_extension(filter, diff);
                yield_constr.constraint(builder, constr);
            }
            {
                let constr = builder.arithmetic_extension(
                    F::ONE,
                    -F::from_canonical_u64(Segment::Stack as u64),
                    filter,
                    channel.addr_segment,
                    filter,
                );
                yield_constr.constraint(builder, constr);
            }
            {
                let diff = builder.sub_extension(channel.addr_virtual, lv.stack_len);
                let constr = builder.arithmetic_extension(
                    F::ONE,
                    F::from_canonical_usize(i + 1),
                    filter,
                    diff,
                    filter,
                );
                yield_constr.constraint(builder, constr);
            }
        }

        // If you also push, you don't need to read the new top of the stack. You can constrain it
        // directly in the op's constraints.
        // If you don't:
        // - if the stack isn't empty after the pops, you read the new top from an extra pop.
        // - if not, the extra read is disabled.
        if !stack_behavior.pushes {
            let target_num_pops =
                builder.constant_extension(F::from_canonical_usize(stack_behavior.num_pops).into());
            let len_diff = builder.sub_extension(lv.stack_len, target_num_pops);
            let new_filter = builder.mul_extension(filter, len_diff);
            let channel = lv.mem_channels[stack_behavior.num_pops - 1];

            {
                let constr = builder.mul_sub_extension(new_filter, channel.used, new_filter);
                yield_constr.constraint(builder, constr);
            }
            {
                let constr = builder.mul_sub_extension(new_filter, channel.is_read, new_filter);
                yield_constr.constraint(builder, constr);
            }
            {
                let diff = builder.sub_extension(channel.addr_context, lv.context);
                let constr = builder.mul_extension(new_filter, diff);
                yield_constr.constraint(builder, constr);
            }
            {
                let constr = builder.arithmetic_extension(
                    F::ONE,
                    -F::from_canonical_u64(Segment::Stack as u64),
                    new_filter,
                    channel.addr_segment,
                    new_filter,
                );
                yield_constr.constraint(builder, constr);
            }
            {
                let diff = builder.sub_extension(channel.addr_virtual, lv.stack_len);
                let constr = builder.arithmetic_extension(
                    F::ONE,
                    F::from_canonical_usize(stack_behavior.num_pops + 1),
                    new_filter,
                    diff,
                    new_filter,
                );
                yield_constr.constraint(builder, constr);
            }
        }
    }
    // If the op only pushes, you only need to constrain the top of the stack if the stack isn't empty.
    else if stack_behavior.pushes {
        // If len > 0...
        let new_filter = builder.mul_extension(lv.stack_len, filter);
        // You write the previous top of the stack in memory, in the last channel.
        let channel = lv.mem_channels[NUM_GP_CHANNELS - 1];
        {
            let constr = builder.mul_sub_extension(new_filter, channel.used, new_filter);
            yield_constr.constraint(builder, constr);
        }
        {
            let constr = builder.mul_extension(new_filter, channel.is_read);
            yield_constr.constraint(builder, constr);
        }

        {
            let diff = builder.sub_extension(channel.addr_context, lv.context);
            let constr = builder.mul_extension(new_filter, diff);
            yield_constr.constraint(builder, constr);
        }
        {
            let constr = builder.arithmetic_extension(
                F::ONE,
                -F::from_canonical_u64(Segment::Stack as u64),
                new_filter,
                channel.addr_segment,
                new_filter,
            );
            yield_constr.constraint(builder, constr);
        }
        {
            let diff = builder.sub_extension(channel.addr_virtual, lv.stack_len);
            let constr = builder.arithmetic_extension(F::ONE, F::ONE, new_filter, diff, new_filter);
            yield_constr.constraint(builder, constr);
        }
    }

    // Unused channels
    if stack_behavior.disable_other_channels {
        for i in stack_behavior.num_pops..NUM_GP_CHANNELS - (stack_behavior.pushes as usize) {
            let channel = lv.mem_channels[i];
            let constr = builder.mul_extension(filter, channel.used);
            yield_constr.constraint(builder, constr);
        }
    }
}

pub fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
    lv: &CpuColumnsView<ExtensionTarget<D>>,
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    for (op, stack_behavior) in izip!(lv.op.into_iter(), STACK_BEHAVIORS.into_iter()) {
        if let Some(stack_behavior) = stack_behavior {
            eval_ext_circuit_one(builder, lv, op, stack_behavior, yield_constr);
        }
    }
}

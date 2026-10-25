// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const LEN: usize = 4096;
const ACCESSES: usize = 256;
const CASES: &[(&str, &str)] = &[
    ("nonnull", "random"),
    ("nonnull", "same_patch"),
    ("nonnull", "same_fill"),
    ("nullable", "random"),
    ("nullable", "same_patch"),
    ("nullable", "same_fill"),
];

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = array_session();
    vortex_sparse::initialize(&session);
    session
});

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

fn fixture(mode: &str) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let indices: Buffer<u32> = (0..LEN / 8).map(|index| (index * 8) as u32).collect();
    let (values, fill) = if mode == "nullable" {
        (
            PrimitiveArray::from_option_iter((0..LEN / 8).map(|index| {
                (index % 3 != 0).then_some(index as i32)
            }))
            .into_array(),
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
        )
    } else {
        (
            PrimitiveArray::from_iter((0..LEN / 8).map(|index| index as i32)).into_array(),
            Scalar::from(-1i32),
        )
    };
    Sparse::try_new(indices.into_array(), values, LEN, fill, &mut ctx)
        .unwrap()
        .into_array()
}

fn read_index(pattern: &str, read: usize) -> usize {
    match pattern {
        "random" => (read * 7919) % LEN,
        "same_patch" => 1000,
        "same_fill" => 1001,
        _ => unreachable!(),
    }
}

#[divan::bench(args = CASES)]
fn one_off(bencher: Bencher, (mode, pattern): (&str, &str)) {
    let array = fixture(mode);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for read in 0..ACCESSES {
                divan::black_box(array.execute_scalar(read_index(pattern, read), ctx).unwrap());
            }
        });
}

#[divan::bench(args = CASES)]
fn repeated(bencher: Bencher, (mode, pattern): (&str, &str)) {
    let array = fixture(mode);
    bencher
        .with_inputs(|| (SESSION.create_execution_ctx(), array.repeated_probe()))
        .bench_refs(|(ctx, probe)| {
            for read in 0..ACCESSES {
                divan::black_box(probe.execute_scalar(read_index(pattern, read), ctx).unwrap());
            }
        });
}

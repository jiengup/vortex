// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_session::VortexSession;
use vortex_zstd::ZstdBuffers;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const LEN: usize = 4096;
const ACCESSES: usize = 64;
const CASES: &[(&str, &str)] = &[
    ("primitive", "random"),
    ("primitive", "same_row"),
    ("nullable", "random"),
    ("nullable", "same_row"),
    ("utf8", "random"),
    ("utf8", "same_row"),
];

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty().with::<KernelSession>();
    vortex_zstd::initialize(&session);
    session
});

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

fn fixture(mode: &str) -> ArrayRef {
    let input = match mode {
        "primitive" => PrimitiveArray::from_iter((0..LEN).map(|index| index as i32)).into_array(),
        "nullable" => PrimitiveArray::from_option_iter(
            (0..LEN).map(|index| (index % 7 != 0).then_some(index as i32)),
        )
        .into_array(),
        "utf8" => VarBinViewArray::from_iter_str(
            (0..LEN).map(|index| format!("row-{index:08}-xxxxxxxxxxxxxxxx")),
        )
        .into_array(),
        _ => unreachable!(),
    };
    ZstdBuffers::compress(&input, 0, &SESSION)
        .unwrap()
        .into_array()
}

fn read_index(pattern: &str, read: usize) -> usize {
    match pattern {
        "random" => (read * 7919) % LEN,
        "same_row" => 1025,
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

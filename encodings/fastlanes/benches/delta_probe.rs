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
use vortex_fastlanes::Delta;
use vortex_fastlanes::DeltaArraySlotsExt;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const LEN: usize = 65_536;
const ACCESSES: usize = 256;
const CASES: &[(&str, &str)] = &[
    ("primitive_u32", "random"),
    ("primitive_u32", "same_chunk"),
    ("primitive_u32", "two_chunks"),
    ("bitpacked_u32", "random"),
    ("bitpacked_u32", "same_chunk"),
    ("bitpacked_u32", "two_chunks"),
    ("nullable_i32", "random"),
    ("nullable_i32", "same_chunk"),
    ("nullable_i32", "two_chunks"),
];

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    session
});

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

fn fixture(mode: &str) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    if mode == "nullable_i32" {
        let source = PrimitiveArray::from_option_iter((0..LEN).map(|index| {
            (index % 7 != 0).then_some(index as i32 - 32_000)
        }));
        return Delta::try_from_primitive_array(&source, &mut ctx)
            .unwrap()
            .into_array();
    }

    let source = PrimitiveArray::from_iter((0..LEN).map(|index| index as u32 * 2));
    let encoded = Delta::try_from_primitive_array(&source, &mut ctx).unwrap();
    if mode == "bitpacked_u32" {
        let deltas = encoded
            .deltas()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        let packed = bitpack_encode(&deltas, 16, None, &mut ctx).unwrap();
        Delta::try_new(
            encoded.bases().clone(),
            packed.into_array(),
            encoded.validity().unwrap(),
            0,
            LEN,
        )
        .unwrap()
        .into_array()
    } else {
        encoded.into_array()
    }
}

fn read_index(pattern: &str, read: usize) -> usize {
    match pattern {
        "random" => (read * 7919) % LEN,
        "same_chunk" => 1024 + read * 3,
        "two_chunks" => 1024 + (read % 2) * 8192 + read / 2,
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

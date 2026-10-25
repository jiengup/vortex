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
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_fastlanes::Delta;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_onpair::DEFAULT_CONFIG;
use vortex_onpair::OnPair;
use vortex_onpair::OnPairArrayExt;
use vortex_onpair::OnPairArraySlotsExt;
use vortex_onpair::onpair_compress;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const LEN: usize = 4096;
const ACCESSES: usize = 256;
const CASES: &[(&str, &str)] = &[
    ("short_plain", "random"),
    ("short_plain", "sequential"),
    ("short_plain", "same_row"),
    ("long_plain", "random"),
    ("long_plain", "sequential"),
    ("long_plain", "same_row"),
    ("short_compressed", "random"),
    ("short_compressed", "sequential"),
    ("short_compressed", "same_row"),
    ("long_compressed", "random"),
    ("long_compressed", "sequential"),
    ("long_compressed", "same_row"),
];

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = array_session();
    vortex_onpair::initialize(&session);
    vortex_fastlanes::initialize(&session);
    session
});

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

fn fixture(mode: &str) -> ArrayRef {
    let long = mode.starts_with("long");
    let strings: Vec<String> = (0..LEN)
        .map(|index| {
            if long {
                format!("https://example.com/tenants/{index:08}/events/{index:08}/details?source=benchmark")
            } else {
                format!("key-{:02}", index % 23)
            }
        })
        .collect();
    let input = VarBinArray::from_iter(
        strings.iter().map(|string| Some(string.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let mut ctx = SESSION.create_execution_ctx();
    let array = onpair_compress(input.as_array(), DEFAULT_CONFIG, &mut ctx)
        .unwrap()
        .try_downcast::<OnPair>()
        .unwrap();
    if mode.ends_with("plain") {
        return array.into_array();
    }

    let view = array.as_view();
    let offsets = view
        .codes_offsets()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let lengths = view
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let codes = view
        .codes()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let offsets = Delta::try_from_primitive_array(&offsets, &mut ctx)
        .unwrap()
        .into_array();
    let lengths = Delta::try_from_primitive_array(&lengths, &mut ctx)
        .unwrap()
        .into_array();
    let codes = bitpack_encode(&codes, 12, None, &mut ctx)
        .unwrap()
        .into_array();
    OnPair::try_new_with_data(
        view.dtype().clone(),
        view.data().clone(),
        view.dict_offsets().clone(),
        codes,
        offsets,
        lengths,
        view.array_validity(),
    )
    .unwrap()
    .into_array()
}

fn read_index(pattern: &str, read: usize) -> usize {
    match pattern {
        "random" => (read * 7919) % LEN,
        "sequential" => 1000 + read,
        "same_row" => 1000,
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

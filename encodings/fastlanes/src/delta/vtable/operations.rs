// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ProbeState;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;

use super::Delta;
use crate::FL_CHUNK_SIZE;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::DeltaArraySlotsExt;
use crate::delta::array::delta_decompress::delta_decompress;
use crate::delta::array::lane_count;

impl OperationsVTable<Delta> for Delta {
    type ProbeState = DeltaProbeState;

    fn probe_scalar(
        state: &mut ProbeState<'_, Delta>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        if !state.is_valid(index, ctx)? {
            return Ok(Scalar::null(state.array().dtype().clone()));
        }

        let array = state.array();
        let physical_index = array.offset() + index;
        let chunk = physical_index / FL_CHUNK_SIZE;
        let index_in_chunk = physical_index % FL_CHUNK_SIZE;
        if let Some(retained) = state.retained()
            && let Some(cached) = &retained.last_chunk
            && cached.chunk == chunk
        {
            return Ok(scalar_from_chunk(
                &cached.values,
                index_in_chunk,
                array.dtype(),
            ));
        }

        let values = decode_chunk(array, chunk, ctx)?;
        let scalar = scalar_from_chunk(&values, index_in_chunk, array.dtype());
        if let Some(retained) = state.retained() {
            retained.last_chunk = Some(DecodedChunk { chunk, values });
        }
        Ok(scalar)
    }

    fn scalar_at(
        array: ArrayView<'_, Delta>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

/// Retains the most recently decoded chunk for a repeated Delta probe.
#[derive(Default)]
pub struct DeltaProbeState {
    last_chunk: Option<DecodedChunk>,
}

struct DecodedChunk {
    chunk: usize,
    values: PrimitiveArray,
}

fn decode_chunk(
    array: ArrayView<'_, Delta>,
    chunk: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let lanes = lane_count(array.dtype().as_ptype());
    let bases = array.bases().slice(chunk * lanes..(chunk + 1) * lanes)?;
    let deltas = array
        .deltas()
        .slice(chunk * FL_CHUNK_SIZE..(chunk + 1) * FL_CHUNK_SIZE)?;
    let selected = Delta::try_new(bases, deltas, Validity::NonNullable, 0, FL_CHUNK_SIZE)?;
    delta_decompress(&selected, ctx)
}

fn scalar_from_chunk(values: &PrimitiveArray, index: usize, dtype: &DType) -> Scalar {
    match_each_integer_ptype!(dtype.as_ptype(), |T| {
        Scalar::primitive(values.as_slice::<T>()[index], dtype.nullability())
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::compute::conformance::consistency::test_scalar_probe_consistency;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::BitPacked;
    use crate::Delta;
    use crate::DeltaArray;
    use crate::DeltaArraySlotsExt;
    use crate::FoR;
    use crate::bitpack_compress::bitpack_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn da(array: &PrimitiveArray) -> DeltaArray {
        Delta::try_from_primitive_array(array, &mut SESSION.create_execution_ctx())
            .vortex_expect("Delta array construction should succeed")
    }

    #[test]
    fn test_slice_non_jagged_array_first_chunk_of_two() {
        let delta = da(&(0u32..2048).collect());

        let actual = delta.slice(10..250).unwrap();
        let expected = PrimitiveArray::from_iter(10u32..250).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_non_jagged_array_second_chunk_of_two() {
        let delta = da(&(0u32..2048).collect());

        let actual = delta.slice(1024 + 10..1024 + 250).unwrap();
        let expected = PrimitiveArray::from_iter((1024 + 10u32)..(1024 + 250)).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_non_jagged_array_span_two_chunks_chunk_of_two() {
        let delta = da(&(0u32..2048).collect());

        let actual = delta.slice(1000..1048).unwrap();
        let expected = PrimitiveArray::from_iter(1000u32..1048).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_non_jagged_array_span_two_chunks_chunk_of_four() {
        let delta = da(&(0u32..4096).collect());

        let actual = delta.slice(2040..2050).unwrap();
        let expected = PrimitiveArray::from_iter(2040u32..2050).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_non_jagged_array_whole() {
        let delta = da(&(0u32..4096).collect());

        let actual = delta.slice(0..4096).unwrap();
        let expected = PrimitiveArray::from_iter(0u32..4096).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_non_jagged_array_empty() {
        let delta = da(&(0u32..4096).collect());

        let actual = delta.slice(0..0).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());

        let actual = delta.slice(4096..4096).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());

        let actual = delta.slice(1024..1024).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_jagged_array_second_chunk_of_two() {
        let delta = da(&(0u32..2000).collect());

        let actual = delta.slice(1024 + 10..1024 + 250).unwrap();
        let expected = PrimitiveArray::from_iter((1024 + 10u32)..(1024 + 250)).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_jagged_array_empty() {
        let delta = da(&(0u32..4000).collect());

        let actual = delta.slice(0..0).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());

        let actual = delta.slice(4000..4000).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());

        let actual = delta.slice(1024..1024).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_of_slice_of_non_jagged() {
        let delta = da(&(0u32..2048).collect());

        let sliced = delta.slice(10..1013).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![10u32, 11]).into_array();
        assert_arrays_eq!(sliced_again, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_of_slice_of_jagged() {
        let delta = da(&(0u32..2000).collect());

        let sliced = delta.slice(10..1013).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![10u32, 11]).into_array();
        assert_arrays_eq!(sliced_again, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_of_slice_second_chunk_of_non_jagged() {
        let delta = da(&(0u32..2048).collect());

        let sliced = delta.slice(1034..1050).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![1034u32, 1035]).into_array();
        assert_arrays_eq!(sliced_again, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_of_slice_second_chunk_of_jagged() {
        let delta = da(&(0u32..2000).collect());

        let sliced = delta.slice(1034..1050).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![1034u32, 1035]).into_array();
        assert_arrays_eq!(sliced_again, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_of_slice_spanning_two_chunks_of_non_jagged() {
        let delta = da(&(0u32..2048).collect());

        let sliced = delta.slice(1010..1050).unwrap();
        let sliced_again = sliced.slice(5..20).unwrap();

        let expected = PrimitiveArray::from_iter(1015u32..1030).into_array();
        assert_arrays_eq!(sliced_again, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_of_slice_spanning_two_chunks_of_jagged() {
        let delta = da(&(0u32..2000).collect());

        let sliced = delta.slice(1010..1050).unwrap();
        let sliced_again = sliced.slice(5..20).unwrap();

        let expected = PrimitiveArray::from_iter(1015u32..1030).into_array();
        assert_arrays_eq!(sliced_again, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_scalar_at_non_jagged_array() {
        let delta = da(&(0u32..2048).collect()).into_array();

        let expected = PrimitiveArray::from_iter(0u32..2048).into_array();
        assert_arrays_eq!(delta, expected, &mut SESSION.create_execution_ctx());
        for index in [0, 1, 10, 1023, 1024, 1025, 2047] {
            let scalar = delta
                .execute_scalar(index, &mut SESSION.create_execution_ctx())
                .unwrap();
            let expected = Scalar::from(index as u32);
            assert_eq!(scalar.dtype(), expected.dtype());
            assert_eq!(scalar, expected);
        }
    }

    #[test]
    #[should_panic]
    fn test_scalar_at_non_jagged_array_oob() {
        let delta = da(&(0u32..2048).collect()).into_array();
        delta
            .execute_scalar(2048, &mut SESSION.create_execution_ctx())
            .unwrap();
    }
    #[test]
    fn test_scalar_at_jagged_array() {
        let delta = da(&(0u32..2000).collect()).into_array();

        let expected = PrimitiveArray::from_iter(0u32..2000).into_array();
        assert_arrays_eq!(delta, expected, &mut SESSION.create_execution_ctx());
        for index in [0, 1, 10, 1023, 1024, 1025, 1999] {
            let scalar = delta
                .execute_scalar(index, &mut SESSION.create_execution_ctx())
                .unwrap();
            let expected = Scalar::from(index as u32);
            assert_eq!(scalar.dtype(), expected.dtype());
            assert_eq!(scalar, expected);
        }
    }

    #[test]
    #[should_panic]
    fn test_scalar_at_jagged_array_oob() {
        let delta = da(&(0u32..2000).collect()).into_array();
        delta
            .execute_scalar(2000, &mut SESSION.create_execution_ctx())
            .unwrap();
    }

    #[rstest]
    // Basic delta arrays
    #[case::delta_u32((0u32..100).collect())]
    #[case::delta_u64((0..100).map(|i| i as u64 * 10).collect())]
    // Large arrays (multiple chunks)
    #[case::delta_large_u32((0u32..2048).collect())]
    #[case::delta_large_u64((0u64..2048).collect())]
    // Single element
    #[case::delta_single(PrimitiveArray::new(buffer![42u32], Validity::NonNullable))]
    // Signed inputs (added with signed-delta support).
    #[case::delta_i32_crossing_zero((-100i32..100).collect())]
    #[case::delta_i64_negative((0i64..100).map(|i| -i * 10).collect())]
    #[case::delta_large_i32((-1024i32..1024).collect())]
    #[case::delta_single_negative(PrimitiveArray::new(buffer![-42i32], Validity::NonNullable))]
    fn test_delta_consistency(#[case] array: PrimitiveArray) {
        test_array_consistency(
            &da(&array).into_array(),
            &mut SESSION.create_execution_ctx(),
        );
    }

    #[rstest]
    #[case::delta_u8_basic(PrimitiveArray::new(buffer![1u8, 1, 1, 1, 1], Validity::NonNullable))]
    #[case::delta_u16_basic(PrimitiveArray::new(buffer![1u16, 1, 1, 1, 1], Validity::NonNullable))]
    #[case::delta_u32_basic(PrimitiveArray::new(buffer![1u32, 1, 1, 1, 1], Validity::NonNullable))]
    #[case::delta_u64_basic(PrimitiveArray::new(buffer![1u64, 1, 1, 1, 1], Validity::NonNullable))]
    #[case::delta_u32_large(PrimitiveArray::new(buffer![1u32; 100], Validity::NonNullable))]
    #[case::delta_i8_basic(PrimitiveArray::new(buffer![-1i8, -1, -1, -1, -1], Validity::NonNullable))]
    #[case::delta_i32_basic(PrimitiveArray::new(buffer![-1i32, -1, -1, -1, -1], Validity::NonNullable))]
    fn test_delta_binary_numeric(#[case] array: PrimitiveArray) {
        test_binary_numeric_array(
            &da(&array).into_array(),
            &mut SESSION.create_execution_ctx(),
        );
    }

    #[test]
    fn probe_signed_nullable_chunk_revisits_and_slices() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source = PrimitiveArray::from_option_iter((0..3073).map(|index| {
            (index % 11 != 0).then_some(if index % 2 == 0 {
                i32::MAX.wrapping_sub(index)
            } else {
                i32::MIN.wrapping_add(index)
            })
        }));
        let delta = da(&source).into_array();
        test_scalar_probe_consistency(&delta, &mut ctx);
        let sliced = delta.slice(100..3073)?.slice(900..2500)?;
        test_scalar_probe_consistency(&sliced, &mut ctx);

        let mut probe = sliced.repeated_probe();
        for index in [23, 24, 1, 1024, 511, 23, 1500, 24, 1] {
            assert_eq!(
                probe.execute_scalar(index, &mut ctx)?,
                source.execute_scalar(index + 1000, &mut ctx)?
            );
        }
        Ok(())
    }

    #[test]
    fn probe_bitpacked_deltas_across_chunks() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source = PrimitiveArray::from_iter(0u32..2500);
        let encoded = da(&source);
        let deltas = encoded
            .deltas()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let packed = bitpack_encode(&deltas, 16, None, &mut ctx)?.into_array();
        let delta = Delta::try_new(
            encoded.bases().clone(),
            packed,
            source.validity()?,
            0,
            source.len(),
        )?
        .into_array();
        test_scalar_probe_consistency(&delta, &mut ctx);
        let sliced = delta.slice(900..2300)?.slice(100..1300)?;
        test_scalar_probe_consistency(&sliced, &mut ctx);

        let mut probe = sliced.repeated_probe();
        for index in [23, 24, 48, 1024, 23, 1100, 24] {
            assert_eq!(
                probe.execute_scalar(index, &mut ctx)?,
                source.execute_scalar(index + 1000, &mut ctx)?
            );
        }
        Ok(())
    }

    #[test]
    fn probe_skips_invalid_unselected_chunk() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let offsets = FoR::try_new(
            PrimitiveArray::from_iter([0u64, 128, 0]).into_array(),
            0u64.into(),
        )?
        .into_array();
        let deltas = BitPacked::try_new_with_block_offsets(
            BufferHandle::new_host(ByteBuffer::zeroed(128)),
            PType::U32,
            Validity::NonNullable,
            None,
            offsets,
            2048,
            0,
        )?
        .into_array();
        let delta = Delta::try_new(
            PrimitiveArray::from_iter([0u32; 64]).into_array(),
            deltas,
            Validity::from_iter((0..2048).map(|index| index != 1024)),
            0,
            2048,
        )?
        .into_array();

        let mut probe = delta.repeated_probe();
        assert_eq!(probe.execute_scalar(0, &mut ctx)?, Scalar::from(Some(0u32)));
        assert_eq!(
            probe.execute_scalar(1024, &mut ctx)?,
            Scalar::null(delta.dtype().clone())
        );
        assert!(probe.execute_scalar(1025, &mut ctx).is_err());
        assert_eq!(probe.execute_scalar(0, &mut ctx)?, Scalar::from(Some(0u32)));
        Ok(())
    }
}

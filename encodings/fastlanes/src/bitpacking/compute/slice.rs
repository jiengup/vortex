// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::max;
use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::patches::Patches;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitWidthsView;
use crate::FL_CHUNK_SIZE;
use crate::bitpacking::array::BitPackedArrayExt;
use crate::bitpacking::array::block_range;

impl SliceReduce for BitPacked {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let BitWidthsView::Global(bit_width) = array.bit_widths() else {
            return Ok(None);
        };
        // We cannot access buffers (to slice the patches).
        if array.patches().is_some() {
            return Ok(None);
        }

        Ok(Some(slice_bitpacked(array, bit_width, range, None)?))
    }
}

impl SliceKernel for BitPacked {
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let patches = array
            .patches()
            .map(|p| p.slice(range.clone(), ctx))
            .transpose()?
            .flatten();

        match array.bit_widths() {
            BitWidthsView::Global(bit_width) => {
                Ok(Some(slice_bitpacked(array, bit_width, range, patches)?))
            }
            BitWidthsView::Blocked(offsets) => {
                if range.is_empty() {
                    return Ok(None);
                }
                let start = range.start + usize::from(array.offset());
                let end = range.end + usize::from(array.offset());
                let first_block = start / FL_CHUNK_SIZE;
                let end_block = end.div_ceil(FL_CHUNK_SIZE);
                let mut probe = offsets.repeated_probe();
                let base = u64::try_from(&probe.execute_scalar(0, ctx)?)?;
                let mut boundaries = Vec::with_capacity(end_block - first_block + 1);
                for index in first_block..=end_block {
                    boundaries.push(u64::try_from(&probe.execute_scalar(index, ctx)?)?);
                }
                // Validate only selected blocks before exposing primitive offsets to the constructor.
                let mut packed_start = 0;
                let mut packed_end = 0;
                for (index, pair) in boundaries.windows(2).enumerate() {
                    let bytes = block_range(
                        base,
                        pair[0],
                        pair[1],
                        array.dtype().as_ptype().bit_width() as u64,
                        array.packed().len(),
                    )?;
                    if index == 0 {
                        packed_start = bytes.start;
                    }
                    packed_end = bytes.end;
                }
                Ok(Some(
                    BitPacked::try_new_with_block_offsets(
                        array.packed().slice(packed_start..packed_end),
                        array.dtype().as_ptype(),
                        array.validity()?.slice(range.clone())?,
                        patches,
                        PrimitiveArray::from_iter(boundaries).into_array(),
                        range.len(),
                        (start % FL_CHUNK_SIZE) as u16,
                    )?
                    .into_array(),
                ))
            }
        }
    }
}

fn slice_bitpacked(
    array: ArrayView<'_, BitPacked>,
    bit_width: u8,
    range: Range<usize>,
    patches: Option<Patches>,
) -> VortexResult<ArrayRef> {
    let offset_start = range.start + array.offset() as usize;
    let offset_stop = range.end + array.offset() as usize;
    let offset = offset_start % 1024;
    let block_start = max(0, offset_start - offset);
    let block_stop = offset_stop.div_ceil(1024) * 1024;

    let encoded_start = (block_start / 8) * bit_width as usize;
    let encoded_stop = (block_stop / 8) * bit_width as usize;

    Ok(BitPacked::try_new(
        array.packed().slice(encoded_start..encoded_stop),
        array.dtype().as_ptype(),
        array.validity()?.slice(range.clone())?,
        patches,
        bit_width,
        range.len(),
        offset as u16,
    )?
    .into_array())
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::SliceArray;
    use vortex_array::arrays::slice::SliceKernel;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::BitPacked;
    use crate::bitpack_compress::bitpack_encode;
    use crate::bitpack_compress::bitpack_encode_blocked;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case(0..1024)]
    #[case(1024..2048)]
    #[case(500..2500)]
    #[case(2048..3000)]
    fn blocked_slice_preserves_patches_validity_and_offset(
        #[case] range: Range<usize>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source =
            PrimitiveArray::from_option_iter((0u32..3000).map(|i| (i % 13 != 0).then_some(i % 31)));
        let packed = bitpack_encode_blocked(&source, &[2, 3, 4], None, &mut ctx)?;
        let sliced = <BitPacked as SliceKernel>::slice(packed.as_view(), range.clone(), &mut ctx)?
            .vortex_expect("blocked slice is supported");
        assert!(sliced.is::<BitPacked>());
        assert_arrays_eq!(sliced.clone(), source.slice(range.clone())?, &mut ctx);
        let nested_range = 1..sliced.len() - 1;
        let nested = <BitPacked as SliceKernel>::slice(
            sliced.as_::<BitPacked>(),
            nested_range.clone(),
            &mut ctx,
        )?
        .vortex_expect("nested blocked slice is supported");
        assert_arrays_eq!(
            nested,
            source.slice(range.start + nested_range.start..range.start + nested_range.end)?,
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_reduce_parent_returns_bitpacked_slice() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter(0u32..2048);
        let bitpacked = bitpack_encode(&values, 11, None, &mut ctx)?;

        let slice_array = SliceArray::new(bitpacked.clone().into_array(), 500..1500);

        let bitpacked_ref = bitpacked.into_array();
        let reduced = bitpacked_ref
            .reduce_parent(&slice_array.into_array(), 0)?
            .expect("expected slice kernel to execute");

        assert!(reduced.is::<BitPacked>());
        let reduced_bp = reduced.as_::<BitPacked>();
        assert_eq!(reduced_bp.offset(), 500);
        assert_eq!(reduced.len(), 1000);
        assert_arrays_eq!(reduced, PrimitiveArray::from_iter(500u32..1500), &mut ctx);

        Ok(())
    }
}

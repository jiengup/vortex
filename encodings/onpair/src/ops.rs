// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ProbeState;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::varbin::varbin_scalar;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use crate::OnPair;
use crate::OnPairArraySlotsExt;
use crate::OnPairSlots;
use crate::array::dict_view;
use crate::decode::collect_widened;

// Scalar probes beat row materialization for a few compressed codes, but lose for longer runs.
const MAX_POINT_CODE_PROBES: usize = 4;

impl OperationsVTable<OnPair> for OnPair {
    type ProbeState = OnPairProbeState;

    fn probe_scalar(
        state: &mut ProbeState<'_, OnPair>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        if !state.is_valid(index, ctx)? {
            return Ok(Scalar::null(state.array().dtype().clone()));
        }

        let array = state.array();
        let cached = state.retained().and_then(|retained| retained.last_bounds);
        let (row_start, row_end) = if let Some((row, start, end)) = cached
            && row == index
        {
            (start, end)
        } else {
            let row_start = if let Some((row, _, end)) = cached
                && row + 1 == index
            {
                end
            } else {
                metadata_at(
                    state,
                    array.codes_offsets(),
                    OnPairSlots::CODES_OFFSETS,
                    index,
                    "codes_offsets",
                    ctx,
                )?
            };
            let row_end = metadata_at(
                state,
                array.codes_offsets(),
                OnPairSlots::CODES_OFFSETS,
                index + 1,
                "codes_offsets",
                ctx,
            )?;
            (row_start, row_end)
        };
        if let Some(retained) = state.retained() {
            retained.last_bounds = Some((index, row_start, row_end));
        }

        let codes_slice = array.codes().slice(row_start..row_end)?;
        let codes = if codes_slice.is::<Primitive>() || codes_slice.len() > MAX_POINT_CODE_PROBES {
            collect_widened::<u16>(&codes_slice, ctx)?
        } else {
            let mut probe = state
                .slot(OnPairSlots::CODES)?
                .vortex_expect("OnPair codes slot is present");
            let codes = (row_start..row_end)
                .map(|code_index| {
                    probe
                        .execute_scalar(code_index, ctx)?
                        .as_primitive()
                        .as_opt::<u16>()
                        .flatten()
                        .ok_or_else(|| vortex_err!("OnPair code {code_index} does not fit u16"))
                })
                .collect::<VortexResult<Vec<u16>>>()?;
            Buffer::from(codes)
        };
        let dict = dict_view(array, ctx)?;

        // The per-row decoded length is recorded in the `uncompressed_lengths`
        // child, so read it directly instead of asking the decoder to compute it.
        let len = metadata_at(
            state,
            array.uncompressed_lengths(),
            OnPairSlots::UNCOMPRESSED_LENGTHS,
            index,
            "uncompressed_lengths",
            ctx,
        )?;
        // The stored length controls allocation; each code emits 1 to MAX_TOKEN_SIZE bytes.
        vortex_ensure!(
            codes.len() <= len && len <= codes.len().saturating_mul(onpair::MAX_TOKEN_SIZE),
            "OnPair row {index} recorded length {len} is impossible for {} codes",
            codes.len()
        );
        let mut buf: Vec<u8> = Vec::with_capacity(len);
        let written =
            match onpair::try_decode_into(codes.as_slice(), dict, buf.spare_capacity_mut()) {
                Ok(written) => written,
                Err(_) => vortex_bail!("OnPair row {index} exceeds its recorded length"),
            };
        vortex_ensure_eq!(written, len, "OnPair row {index} decoded length mismatch");
        // SAFETY: `try_decode_into` initialised exactly `written` bytes.
        unsafe { buf.set_len(written) };
        Ok(varbin_scalar(ByteBuffer::from(buf), array.dtype()))
    }

    fn scalar_at(
        array: ArrayView<'_, OnPair>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

/// Remembers the most recently accessed row's code boundaries.
#[derive(Default)]
pub struct OnPairProbeState {
    last_bounds: Option<(usize, usize, usize)>,
}

fn metadata_at(
    state: &mut ProbeState<'_, OnPair>,
    child: &ArrayRef,
    slot: usize,
    index: usize,
    name: &str,
    ctx: &mut ExecutionCtx,
) -> VortexResult<usize> {
    if let Some(primitive) = child.as_opt::<Primitive>()
        && primitive.buffer_handle().is_on_host()
    {
        return match_each_integer_ptype!(primitive.ptype(), |T| {
            usize::try_from(primitive.as_slice::<T>()[index])
                .map_err(|_| vortex_err!("OnPair {name}[{index}] is null, negative, or too large"))
        });
    }

    state
        .slot(slot)?
        .vortex_expect("OnPair metadata slot is present")
        .execute_scalar(index, ctx)?
        .as_primitive()
        .as_opt::<usize>()
        .flatten()
        .ok_or_else(|| vortex_err!("OnPair {name}[{index}] is null, negative, or too large"))
}

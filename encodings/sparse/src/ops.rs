// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ProbeState;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::Sparse;
use crate::SparseExt as _;
use crate::SparseSlots;
use crate::PATCH_SLOTS;

impl OperationsVTable<Sparse> for Sparse {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, Sparse>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        let (_, mut children) = state.split();
        let patch_idx = array
            .patches()
            .search_index_with_probes(index, &mut children, PATCH_SLOTS, ctx)?
            .to_found();
        match patch_idx {
            Some(patch_idx) => children
                .slot(SparseSlots::PATCH_VALUES)?
                .vortex_expect("Sparse patch values slot is present")
                .execute_scalar(patch_idx, ctx),
            None => Ok(array.fill_scalar().clone()),
        }
    }

    fn scalar_at(
        array: ArrayView<'_, Sparse>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Sparse;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[test]
    fn slice_partially_invalid() {
        let mut ctx = SESSION.create_execution_ctx();
        let values = buffer![0u64].into_array();
        let indices = buffer![0u8].into_array();

        let sparse = Sparse::try_new(indices, values, 1000, 999u64.into(), &mut ctx).unwrap();
        let sliced = sparse.slice(0..1000).unwrap();
        let mut expected = vec![999u64; 1000];
        expected[0] = 0;

        assert_arrays_eq!(sliced, PrimitiveArray::from_iter(expected), &mut ctx);
    }

    #[test]
    fn scalar_probe_consistency() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let array = Sparse::try_new(
            buffer![2u32, 5, 8].into_array(),
            PrimitiveArray::from_option_iter([Some(20i32), None, Some(80)]).into_array(),
            10,
            Scalar::null(dtype.clone()),
            &mut ctx,
        )?
        .into_array();
        let expected = [
            Scalar::null(dtype.clone()),
            Scalar::null(dtype.clone()),
            Scalar::from(Some(20i32)),
            Scalar::null(dtype.clone()),
            Scalar::null(dtype.clone()),
            Scalar::null(dtype.clone()),
            Scalar::null(dtype.clone()),
            Scalar::null(dtype.clone()),
            Scalar::from(Some(80i32)),
            Scalar::null(dtype),
        ];
        let mut repeated = array.repeated_probe();
        for index in [2, 1, 5, 8, 2, 9, 5] {
            assert_eq!(array.execute_scalar(index, &mut ctx)?, expected[index]);
            assert_eq!(repeated.execute_scalar(index, &mut ctx)?, expected[index]);
        }
        let sliced = array.slice(1..9)?;
        let mut sliced_probe = sliced.repeated_probe();
        for index in [1, 4, 7, 0, 1] {
            assert_eq!(sliced_probe.execute_scalar(index, &mut ctx)?, expected[index + 1]);
        }
        Ok(())
    }
}

// DO NOT EDIT! This file was auto-generated by crates/re_types_builder/src/codegen/rust/to_archetype.rs

#![allow(unused_imports)]
#![allow(unused_parens)]
#![allow(clippy::clone_on_copy)]

use crate::CachedLatestAtResults;
use re_query2::{PromiseResolver, PromiseResult};
use re_types_core::{Archetype, Loggable as _};
use std::sync::Arc;

impl crate::ToArchetype<re_types::archetypes::DisconnectedSpace> for CachedLatestAtResults {
    #[inline]
    fn to_archetype(
        &self,
        resolver: &PromiseResolver,
    ) -> PromiseResult<crate::Result<re_types::archetypes::DisconnectedSpace>> {
        re_tracing::profile_function!(<re_types::archetypes::DisconnectedSpace>::name());

        // --- Required ---

        use re_types::components::DisconnectedSpace;
        let disconnected_space = match self.get_required(<DisconnectedSpace>::name()) {
            Ok(disconnected_space) => disconnected_space,
            Err(query_err) => return PromiseResult::Ready(Err(query_err)),
        };
        let disconnected_space = match disconnected_space.to_dense::<DisconnectedSpace>(resolver) {
            PromiseResult::Pending => return PromiseResult::Pending,
            PromiseResult::Error(promise_err) => return PromiseResult::Error(promise_err),
            PromiseResult::Ready(query_res) => match query_res {
                Ok(data) => {
                    let Some(first) = data.first().cloned() else {
                        return PromiseResult::Error(std::sync::Arc::new(
                            re_types_core::DeserializationError::missing_data(),
                        ));
                    };
                    first
                }
                Err(query_err) => return PromiseResult::Ready(Err(query_err)),
            },
        };

        // --- Recommended/Optional ---

        // ---

        let arch = re_types::archetypes::DisconnectedSpace { disconnected_space };

        PromiseResult::Ready(Ok(arch))
    }
}

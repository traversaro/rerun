// DO NOT EDIT! This file was auto-generated by crates/re_types_builder/src/codegen/rust/to_archetype.rs

#![allow(unused_imports)]
#![allow(unused_parens)]
#![allow(clippy::clone_on_copy)]

use crate::CachedLatestAtResults;
use re_query2::{PromiseResolver, PromiseResult};
use re_types_core::{Archetype, Loggable as _};
use std::sync::Arc;

impl crate::ToArchetype<re_types::archetypes::Points3D> for CachedLatestAtResults {
    #[inline]
    fn to_archetype(
        &self,
        resolver: &PromiseResolver,
    ) -> PromiseResult<crate::Result<re_types::archetypes::Points3D>> {
        re_tracing::profile_function!(<re_types::archetypes::Points3D>::name());

        // --- Required ---

        use re_types::components::Position3D;
        let positions = match self.get_required(<Position3D>::name()) {
            Ok(positions) => positions,
            Err(query_err) => return PromiseResult::Ready(Err(query_err)),
        };
        let positions = match positions.to_dense::<Position3D>(resolver) {
            PromiseResult::Pending => return PromiseResult::Pending,
            PromiseResult::Error(promise_err) => return PromiseResult::Error(promise_err),
            PromiseResult::Ready(query_res) => match query_res {
                Ok(data) => data.to_vec(),
                Err(query_err) => return PromiseResult::Ready(Err(query_err)),
            },
        };

        // --- Recommended/Optional ---

        use re_types::components::Radius;
        let radii = if let Some(radii) = self.get(<Radius>::name()) {
            match radii.to_dense::<Radius>(resolver) {
                PromiseResult::Pending => return PromiseResult::Pending,
                PromiseResult::Error(promise_err) => return PromiseResult::Error(promise_err),
                PromiseResult::Ready(query_res) => match query_res {
                    Ok(data) => Some(data.to_vec()),
                    Err(query_err) => return PromiseResult::Ready(Err(query_err)),
                },
            }
        } else {
            None
        };

        use re_types::components::Color;
        let colors = if let Some(colors) = self.get(<Color>::name()) {
            match colors.to_dense::<Color>(resolver) {
                PromiseResult::Pending => return PromiseResult::Pending,
                PromiseResult::Error(promise_err) => return PromiseResult::Error(promise_err),
                PromiseResult::Ready(query_res) => match query_res {
                    Ok(data) => Some(data.to_vec()),
                    Err(query_err) => return PromiseResult::Ready(Err(query_err)),
                },
            }
        } else {
            None
        };

        use re_types::components::Text;
        let labels = if let Some(labels) = self.get(<Text>::name()) {
            match labels.to_dense::<Text>(resolver) {
                PromiseResult::Pending => return PromiseResult::Pending,
                PromiseResult::Error(promise_err) => return PromiseResult::Error(promise_err),
                PromiseResult::Ready(query_res) => match query_res {
                    Ok(data) => Some(data.to_vec()),
                    Err(query_err) => return PromiseResult::Ready(Err(query_err)),
                },
            }
        } else {
            None
        };

        use re_types::components::ClassId;
        let class_ids = if let Some(class_ids) = self.get(<ClassId>::name()) {
            match class_ids.to_dense::<ClassId>(resolver) {
                PromiseResult::Pending => return PromiseResult::Pending,
                PromiseResult::Error(promise_err) => return PromiseResult::Error(promise_err),
                PromiseResult::Ready(query_res) => match query_res {
                    Ok(data) => Some(data.to_vec()),
                    Err(query_err) => return PromiseResult::Ready(Err(query_err)),
                },
            }
        } else {
            None
        };

        use re_types::components::KeypointId;
        let keypoint_ids = if let Some(keypoint_ids) = self.get(<KeypointId>::name()) {
            match keypoint_ids.to_dense::<KeypointId>(resolver) {
                PromiseResult::Pending => return PromiseResult::Pending,
                PromiseResult::Error(promise_err) => return PromiseResult::Error(promise_err),
                PromiseResult::Ready(query_res) => match query_res {
                    Ok(data) => Some(data.to_vec()),
                    Err(query_err) => return PromiseResult::Ready(Err(query_err)),
                },
            }
        } else {
            None
        };

        // ---

        let arch = re_types::archetypes::Points3D {
            positions,
            radii,
            colors,
            labels,
            class_ids,
            keypoint_ids,
        };

        PromiseResult::Ready(Ok(arch))
    }
}

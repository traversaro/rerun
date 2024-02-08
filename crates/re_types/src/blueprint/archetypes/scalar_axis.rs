// DO NOT EDIT! This file was auto-generated by crates/re_types_builder/src/codegen/rust/api.rs
// Based on "crates/re_types/definitions/rerun/blueprint/archetypes/scalar_axis.fbs".

#![allow(trivial_numeric_casts)]
#![allow(unused_imports)]
#![allow(unused_parens)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::iter_on_single_items)]
#![allow(clippy::map_flatten)]
#![allow(clippy::match_wildcard_for_single_variants)]
#![allow(clippy::needless_question_mark)]
#![allow(clippy::new_without_default)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::unnecessary_cast)]

use ::re_types_core::external::arrow2;
use ::re_types_core::ComponentName;
use ::re_types_core::SerializationResult;
use ::re_types_core::{ComponentBatch, MaybeOwnedComponentBatch};
use ::re_types_core::{DeserializationError, DeserializationResult};

/// **Archetype**: Configuration for the scalar axis of a plot.
#[derive(Clone, Debug, Default)]
pub struct ScalarAxis {
    /// The range of the axis.
    ///
    /// If unset, the range well be automatically determined based on the queried data.
    pub range: Option<crate::components::Range1D>,

    /// Whether to lock the range of the axis during zoom.
    pub lock_range_during_zoom: Option<crate::blueprint::components::LockRangeDuringZoom>,
}

impl ::re_types_core::SizeBytes for ScalarAxis {
    #[inline]
    fn heap_size_bytes(&self) -> u64 {
        self.range.heap_size_bytes() + self.lock_range_during_zoom.heap_size_bytes()
    }

    #[inline]
    fn is_pod() -> bool {
        <Option<crate::components::Range1D>>::is_pod()
            && <Option<crate::blueprint::components::LockRangeDuringZoom>>::is_pod()
    }
}

static REQUIRED_COMPONENTS: once_cell::sync::Lazy<[ComponentName; 0usize]> =
    once_cell::sync::Lazy::new(|| []);

static RECOMMENDED_COMPONENTS: once_cell::sync::Lazy<[ComponentName; 1usize]> =
    once_cell::sync::Lazy::new(|| ["rerun.blueprint.components.ScalarAxisIndicator".into()]);

static OPTIONAL_COMPONENTS: once_cell::sync::Lazy<[ComponentName; 3usize]> =
    once_cell::sync::Lazy::new(|| {
        [
            "rerun.blueprint.components.LockRangeDuringZoom".into(),
            "rerun.components.InstanceKey".into(),
            "rerun.components.Range1D".into(),
        ]
    });

static ALL_COMPONENTS: once_cell::sync::Lazy<[ComponentName; 4usize]> =
    once_cell::sync::Lazy::new(|| {
        [
            "rerun.blueprint.components.ScalarAxisIndicator".into(),
            "rerun.blueprint.components.LockRangeDuringZoom".into(),
            "rerun.components.InstanceKey".into(),
            "rerun.components.Range1D".into(),
        ]
    });

impl ScalarAxis {
    pub const NUM_COMPONENTS: usize = 4usize;
}

/// Indicator component for the [`ScalarAxis`] [`::re_types_core::Archetype`]
pub type ScalarAxisIndicator = ::re_types_core::GenericIndicatorComponent<ScalarAxis>;

impl ::re_types_core::Archetype for ScalarAxis {
    type Indicator = ScalarAxisIndicator;

    #[inline]
    fn name() -> ::re_types_core::ArchetypeName {
        "rerun.blueprint.archetypes.ScalarAxis".into()
    }

    #[inline]
    fn indicator() -> MaybeOwnedComponentBatch<'static> {
        static INDICATOR: ScalarAxisIndicator = ScalarAxisIndicator::DEFAULT;
        MaybeOwnedComponentBatch::Ref(&INDICATOR)
    }

    #[inline]
    fn required_components() -> ::std::borrow::Cow<'static, [ComponentName]> {
        REQUIRED_COMPONENTS.as_slice().into()
    }

    #[inline]
    fn recommended_components() -> ::std::borrow::Cow<'static, [ComponentName]> {
        RECOMMENDED_COMPONENTS.as_slice().into()
    }

    #[inline]
    fn optional_components() -> ::std::borrow::Cow<'static, [ComponentName]> {
        OPTIONAL_COMPONENTS.as_slice().into()
    }

    #[inline]
    fn all_components() -> ::std::borrow::Cow<'static, [ComponentName]> {
        ALL_COMPONENTS.as_slice().into()
    }

    #[inline]
    fn from_arrow_components(
        arrow_data: impl IntoIterator<Item = (ComponentName, Box<dyn arrow2::array::Array>)>,
    ) -> DeserializationResult<Self> {
        re_tracing::profile_function!();
        use ::re_types_core::{Loggable as _, ResultExt as _};
        let arrays_by_name: ::std::collections::HashMap<_, _> = arrow_data
            .into_iter()
            .map(|(name, array)| (name.full_name(), array))
            .collect();
        let range = if let Some(array) = arrays_by_name.get("rerun.components.Range1D") {
            <crate::components::Range1D>::from_arrow_opt(&**array)
                .with_context("rerun.blueprint.archetypes.ScalarAxis#range")?
                .into_iter()
                .next()
                .flatten()
        } else {
            None
        };
        let lock_range_during_zoom = if let Some(array) =
            arrays_by_name.get("rerun.blueprint.components.LockRangeDuringZoom")
        {
            <crate::blueprint::components::LockRangeDuringZoom>::from_arrow_opt(&**array)
                .with_context("rerun.blueprint.archetypes.ScalarAxis#lock_range_during_zoom")?
                .into_iter()
                .next()
                .flatten()
        } else {
            None
        };
        Ok(Self {
            range,
            lock_range_during_zoom,
        })
    }
}

impl ::re_types_core::AsComponents for ScalarAxis {
    fn as_component_batches(&self) -> Vec<MaybeOwnedComponentBatch<'_>> {
        re_tracing::profile_function!();
        use ::re_types_core::Archetype as _;
        [
            Some(Self::indicator()),
            self.range
                .as_ref()
                .map(|comp| (comp as &dyn ComponentBatch).into()),
            self.lock_range_during_zoom
                .as_ref()
                .map(|comp| (comp as &dyn ComponentBatch).into()),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    #[inline]
    fn num_instances(&self) -> usize {
        0
    }
}

impl ScalarAxis {
    pub fn new() -> Self {
        Self {
            range: None,
            lock_range_during_zoom: None,
        }
    }

    #[inline]
    pub fn with_range(mut self, range: impl Into<crate::components::Range1D>) -> Self {
        self.range = Some(range.into());
        self
    }

    #[inline]
    pub fn with_lock_range_during_zoom(
        mut self,
        lock_range_during_zoom: impl Into<crate::blueprint::components::LockRangeDuringZoom>,
    ) -> Self {
        self.lock_range_during_zoom = Some(lock_range_during_zoom.into());
        self
    }
}
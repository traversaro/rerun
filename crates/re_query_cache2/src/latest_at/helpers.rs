use re_data_store::{DataStore, LatestAtQuery};
use re_log_types::{EntityPath, RowId, TimeInt};
use re_types_core::Component;
use re_types_core::{external::arrow2::array::Array, ComponentName};

use crate::{CachedLatestAtComponentResults, Caches, PromiseResolver, PromiseResult};

// ---

impl CachedLatestAtComponentResults {
    /// Returns the component data as a dense vector.
    ///
    /// Logs a warning and returns `None` if the component is missing or cannot be deserialized.
    #[inline]
    pub fn dense<C: Component>(&self, resolver: &PromiseResolver) -> Option<&[C]> {
        let component_name = C::name();
        let level = re_log::Level::Warn;
        match self.to_dense::<C>(resolver).flatten() {
            PromiseResult::Pending => {
                re_log::debug_once!("Couldn't deserialize {component_name}: promise still pending",);
                None
            }
            PromiseResult::Ready(data) => Some(data),
            PromiseResult::Error(err) => {
                re_log::log_once!(
                    level,
                    "Couldn't deserialize {component_name}: {}",
                    re_error::format_ref(&*err),
                );
                None
            }
        }
    }

    /// Returns the component data as an arrow array.
    ///
    /// Logs a warning and returns `None` if the component is missing or cannot be deserialized.
    #[inline]
    pub fn raw(
        &self,
        resolver: &PromiseResolver,
        component_name: impl Into<ComponentName>,
    ) -> Option<Box<dyn Array>> {
        let component_name = component_name.into();
        let level = re_log::Level::Warn;
        match self.resolved(resolver) {
            PromiseResult::Pending => {
                re_log::debug_once!("Couldn't get {component_name}: promise still pending");
                None
            }
            PromiseResult::Ready(cell) => Some(cell.to_arrow()),
            PromiseResult::Error(err) => {
                re_log::log_once!(
                    level,
                    "Couldn't get {component_name}: {}",
                    re_error::format_ref(&*err),
                );
                None
            }
        }
    }

    /// Returns the component data of the single instance.
    ///
    /// This assumes that the row we get from the store only contains a single instance for this
    /// component; it will log a warning otherwise.
    ///
    /// This should only be used for "mono-components" such as `Transform` and `Tensor`.
    ///
    /// Logs a warning and returns `None` if the component is missing or cannot be deserialized.
    #[inline]
    pub fn mono<C: Component>(&self, resolver: &PromiseResolver) -> Option<C> {
        let component_name = C::name();
        let level = re_log::Level::Warn;
        match self.to_dense::<C>(resolver).flatten() {
            PromiseResult::Pending => {
                re_log::debug_once!("Couldn't deserialize {component_name}: promise still pending",);
                None
            }
            PromiseResult::Ready(data) if data.len() == 1 => Some(data[0].clone()),
            PromiseResult::Ready(data) => {
                re_log::log_once!(
                    level,
                    "Couldn't deserialize {component_name}: not a mono-batch (length: {})",
                    data.len(),
                );
                None
            }
            PromiseResult::Error(err) => {
                re_log::log_once!(
                    level,
                    "Couldn't deserialize {component_name}: {}",
                    re_error::format_ref(&*err),
                );
                None
            }
        }
    }

    /// Returns the component data of the single instance as an arrow array.
    ///
    /// This assumes that the row we get from the store only contains a single instance for this
    /// component; it will log a warning otherwise.
    ///
    /// This should only be used for "mono-components" such as `Transform` and `Tensor`.
    ///
    /// Logs a warning and returns `None` if the component is missing or cannot be deserialized.
    #[inline]
    pub fn mono_raw(
        &self,
        resolver: &PromiseResolver,
        component_name: impl Into<ComponentName>,
    ) -> Option<Box<dyn Array>> {
        let component_name = component_name.into();
        let level = re_log::Level::Warn;
        match self.resolved(resolver) {
            PromiseResult::Pending => {
                re_log::debug_once!("Couldn't get {component_name}: promise still pending");
                None
            }
            PromiseResult::Ready(cell) if cell.as_arrow_ref().len() == 1 => {
                Some(cell.as_arrow_ref().sliced(0, 1))
            }
            PromiseResult::Ready(cell) => {
                re_log::log_once!(
                    level,
                    "Couldn't get {component_name}: not a mono-batch (length: {})",
                    cell.as_arrow_ref().len(),
                );
                None
            }
            PromiseResult::Error(err) => {
                re_log::log_once!(
                    level,
                    "Couldn't get {component_name}: {}",
                    re_error::format_ref(&*err),
                );
                None
            }
        }
    }

    /// Returns the component data of the specified instance.
    ///
    /// Logs a warning and returns `None` if the component is missing or cannot be deserialized, or
    /// the index doesn't exist.
    #[inline]
    pub fn instance<C: Component>(&self, resolver: &PromiseResolver, index: usize) -> Option<C> {
        let component_name = C::name();
        let level = re_log::Level::Warn;
        match self.to_dense::<C>(resolver).flatten() {
            PromiseResult::Pending => {
                re_log::debug_once!("Couldn't deserialize {component_name}: promise still pending",);
                None
            }

            PromiseResult::Ready(data) => {
                // TODO(#5303): Figure out if/how we'd like to integrate clamping semantics into the
                // selection panel.
                //
                // For now, we simply always clamp, which is the closest to the legacy behavior that the UI
                // expects.
                let index = usize::min(index, data.len().saturating_sub(1));

                if data.len() > index {
                    Some(data[index].clone())
                } else {
                    re_log::log_once!(
                        level,
                        "Couldn't deserialize {component_name}: index not found (index: {index}, length: {})",
                        data.len(),
                    );
                    None
                }
            }

            PromiseResult::Error(err) => {
                re_log::log_once!(
                    level,
                    "Couldn't deserialize {component_name}: {}",
                    re_error::format_ref(&*err),
                );
                None
            }
        }
    }

    /// Returns the component data of the specified instance as an arrow array.
    ///
    /// Logs a warning and returns `None` if the component is missing or cannot be deserialized, or
    /// the index doesn't exist.
    #[inline]
    pub fn instance_raw(
        &self,
        resolver: &PromiseResolver,
        component_name: impl Into<ComponentName>,
        index: usize,
    ) -> Option<Box<dyn Array>> {
        let component_name = component_name.into();
        let level = re_log::Level::Warn;
        match self.resolved(resolver) {
            PromiseResult::Pending => {
                re_log::debug_once!("Couldn't get {component_name}: promise still pending");
                None
            }

            PromiseResult::Ready(cell) => {
                let len = cell.num_instances() as usize;

                // TODO(#5303): Figure out if/how we'd like to integrate clamping semantics into the
                // selection panel.
                //
                // For now, we simply always clamp, which is the closest to the legacy behavior that the UI
                // expects.
                let index = usize::min(index, len.saturating_sub(1));

                if len > index {
                    Some(cell.as_arrow_ref().sliced(index, 1))
                } else {
                    re_log::log_once!(
                        level,
                        "Couldn't deserialize {component_name}: index not found (index: {index}, length: {})",
                        len,
                    );
                    None
                }
            }

            PromiseResult::Error(err) => {
                re_log::log_once!(
                    level,
                    "Couldn't get {component_name}: {}",
                    re_error::format_ref(&*err),
                );
                None
            }
        }
    }
}

// ---

#[derive(Clone)]
pub struct CachedLatestAtMonoResult<C> {
    pub index: (TimeInt, RowId),
    pub value: C,
}

impl<C> CachedLatestAtMonoResult<C> {
    #[inline]
    pub fn data_time(&self) -> TimeInt {
        self.index.0
    }

    #[inline]
    pub fn row_id(&self) -> RowId {
        self.index.1
    }
}

impl<C: std::ops::Deref> std::ops::Deref for CachedLatestAtMonoResult<C> {
    type Target = C;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl Caches {
    /// Get the latest index and value for a given dense [`re_types_core::Component`].
    ///
    /// Returns `None` if the data is a promise that has yet to be resolved.
    ///
    /// This assumes that the row we get from the store only contains a single instance for this
    /// component; it will generate a log message of `level` otherwise.
    ///
    /// This should only be used for "mono-components" such as `Transform` and `Tensor`.
    ///
    /// This is a best-effort helper, it will merely log messages on failure.
    //
    // TODO(#5607): what should happen if the promise is still pending?
    pub fn latest_at_component_with_log_level<C: Component>(
        &self,
        store: &DataStore,
        resolver: &PromiseResolver,
        entity_path: &EntityPath,
        query: &LatestAtQuery,
        level: re_log::Level,
    ) -> Option<CachedLatestAtMonoResult<C>> {
        re_tracing::profile_function!();

        let results = self.latest_at(store, query, entity_path, [C::name()]);
        let result = results.get(C::name())?;

        let index @ (data_time, row_id) = *result.index();

        match result.to_dense::<C>(resolver).flatten() {
            PromiseResult::Pending => {
                re_log::debug_once!(
                    "Couldn't deserialize {entity_path}:{} @ {data_time:?}#{row_id}: promise still pending",
                    C::name(),
                );
                None
            }
            PromiseResult::Ready(data) if data.len() == 1 => Some(CachedLatestAtMonoResult {
                index,
                value: data[0].clone(),
            }),
            PromiseResult::Ready(data) => {
                re_log::log_once!(
                    level,
                    "Couldn't deserialize {entity_path}:{} @ {data_time:?}#{row_id}: not a mono-batch (length: {})",
                    C::name(),
                    data.len(),
                );
                None
            }
            PromiseResult::Error(err) => {
                re_log::log_once!(
                    level,
                    "Couldn't deserialize {entity_path} @ {data_time:?}#{row_id}:{}: {}",
                    C::name(),
                    re_error::format_ref(&*err),
                );
                None
            }
        }
    }

    /// Get the latest index and value for a given dense [`re_types_core::Component`].
    ///
    /// This assumes that the row we get from the store only contains a single instance for this
    /// component; it will log a warning otherwise.
    ///
    /// This should only be used for "mono-components" such as `Transform` and `Tensor`.
    ///
    /// This is a best-effort helper, it will merely log errors on failure.
    #[inline]
    pub fn latest_at_component<C: Component>(
        &self,
        store: &DataStore,
        resolver: &PromiseResolver,
        entity_path: &EntityPath,
        query: &LatestAtQuery,
    ) -> Option<CachedLatestAtMonoResult<C>> {
        self.latest_at_component_with_log_level(
            store,
            resolver,
            entity_path,
            query,
            re_log::Level::Warn,
        )
    }

    /// Get the latest index and value for a given dense [`re_types_core::Component`].
    ///
    /// This assumes that the row we get from the store only contains a single instance for this
    /// component; it will return None and log a debug message otherwise.
    ///
    /// This should only be used for "mono-components" such as `Transform` and `Tensor`.
    ///
    /// This is a best-effort helper, it will merely logs debug messages on failure.
    #[inline]
    pub fn latest_at_component_quiet<C: Component>(
        &self,
        store: &DataStore,
        resolver: &PromiseResolver,
        entity_path: &EntityPath,
        query: &LatestAtQuery,
    ) -> Option<CachedLatestAtMonoResult<C>> {
        self.latest_at_component_with_log_level(
            store,
            resolver,
            entity_path,
            query,
            re_log::Level::Debug,
        )
    }

    /// Call [`Self::latest_at_component`] at the given path, walking up the hierarchy until an instance is found.
    pub fn latest_at_component_at_closest_ancestor<C: Component>(
        &self,
        store: &DataStore,
        resolver: &PromiseResolver,
        entity_path: &EntityPath,
        query: &LatestAtQuery,
    ) -> Option<(EntityPath, CachedLatestAtMonoResult<C>)> {
        re_tracing::profile_function!();

        let mut cur_entity_path = Some(entity_path.clone());
        while let Some(entity_path) = cur_entity_path {
            if let Some(result) =
                self.latest_at_component::<C>(store, resolver, &entity_path, query)
            {
                return Some((entity_path, result));
            }
            cur_entity_path = entity_path.parent();
        }
        None
    }
}

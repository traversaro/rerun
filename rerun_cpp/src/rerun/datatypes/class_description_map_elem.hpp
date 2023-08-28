// NOTE: This file was autogenerated by re_types_builder; DO NOT EDIT.
// Based on "crates/re_types/definitions/rerun/datatypes/class_description_map_elem.fbs"

#pragma once

#include "../result.hpp"
#include "class_description.hpp"
#include "class_id.hpp"

#include <cstdint>
#include <memory>
#include <utility>

namespace arrow {
    class DataType;
    class MemoryPool;
    class StructBuilder;
} // namespace arrow

namespace rerun {
    namespace datatypes {
        /// A helper type for mapping class IDs to class descriptions.
        ///
        /// This is internal to the `AnnotationContext` structure.
        struct ClassDescriptionMapElem {
            rerun::datatypes::ClassId class_id;

            rerun::datatypes::ClassDescription class_description;

          public:
            // Extensions to generated type defined in 'class_description_map_elem_ext.cpp'

            ClassDescriptionMapElem(ClassDescription _class_description)
                : class_id(_class_description.info.id),
                  class_description(std::move(_class_description)) {}

          public:
            ClassDescriptionMapElem() = default;

            /// Returns the arrow data type this type corresponds to.
            static const std::shared_ptr<arrow::DataType>& arrow_datatype();

            /// Creates a new array builder with an array of this type.
            static Result<std::shared_ptr<arrow::StructBuilder>> new_arrow_array_builder(
                arrow::MemoryPool* memory_pool
            );

            /// Fills an arrow array builder with an array of this type.
            static Error fill_arrow_array_builder(
                arrow::StructBuilder* builder, const ClassDescriptionMapElem* elements,
                size_t num_elements
            );
        };
    } // namespace datatypes
} // namespace rerun
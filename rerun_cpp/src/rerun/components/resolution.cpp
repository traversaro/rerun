// DO NOT EDIT! This file was auto-generated by crates/re_types_builder/src/codegen/cpp/mod.rs
// Based on "crates/re_types/definitions/rerun/components/resolution.fbs".

#include "resolution.hpp"

#include "../datatypes/vec2d.hpp"

#include <arrow/builder.h>
#include <arrow/type_fwd.h>

namespace rerun::components {
    const char Resolution::NAME[] = "rerun.components.Resolution";

    const std::shared_ptr<arrow::DataType>& Resolution::arrow_datatype() {
        static const auto datatype = rerun::datatypes::Vec2D::arrow_datatype();
        return datatype;
    }

    rerun::Error Resolution::fill_arrow_array_builder(
        arrow::FixedSizeListBuilder* builder, const Resolution* elements, size_t num_elements
    ) {
        static_assert(sizeof(rerun::datatypes::Vec2D) == sizeof(Resolution));
        RR_RETURN_NOT_OK(rerun::datatypes::Vec2D::fill_arrow_array_builder(
            builder,
            reinterpret_cast<const rerun::datatypes::Vec2D*>(elements),
            num_elements
        ));

        return Error::ok();
    }

    Result<rerun::DataCell> Resolution::to_data_cell(
        const Resolution* instances, size_t num_instances
    ) {
        // TODO(andreas): Allow configuring the memory pool.
        arrow::MemoryPool* pool = arrow::default_memory_pool();

        ARROW_ASSIGN_OR_RAISE(auto builder, arrow::MakeBuilder(arrow_datatype(), pool))
        if (instances && num_instances > 0) {
            RR_RETURN_NOT_OK(Resolution::fill_arrow_array_builder(
                static_cast<arrow::FixedSizeListBuilder*>(builder.get()),
                instances,
                num_instances
            ));
        }
        std::shared_ptr<arrow::Array> array;
        ARROW_RETURN_NOT_OK(builder->Finish(&array));

        return rerun::DataCell::create(
            Resolution::NAME,
            Resolution::arrow_datatype(),
            std::move(array)
        );
    }
} // namespace rerun::components

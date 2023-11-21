// DO NOT EDIT! This file was auto-generated by crates/re_types_builder/src/codegen/cpp/mod.rs
// Based on "crates/re_types/definitions/rerun/datatypes/tensor_buffer.fbs".

#pragma once

#include "../collection.hpp"
#include "../half.hpp"
#include "../result.hpp"
#include "../type_traits.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <utility>

namespace arrow {
    class DataType;
    class DenseUnionBuilder;
} // namespace arrow

namespace rerun::datatypes {
    namespace detail {
        /// \private
        enum class TensorBufferTag : uint8_t {
            /// Having a special empty state makes it possible to implement move-semantics. We need to be able to leave the object in a state which we can run the destructor on.
            None = 0,
            U8,
            U16,
            U32,
            U64,
            I8,
            I16,
            I32,
            I64,
            F16,
            F32,
            F64,
            JPEG,
            NV12,
        };

        /// \private
        union TensorBufferData {
            rerun::Collection<uint8_t> u8;

            rerun::Collection<uint16_t> u16;

            rerun::Collection<uint32_t> u32;

            rerun::Collection<uint64_t> u64;

            rerun::Collection<int8_t> i8;

            rerun::Collection<int16_t> i16;

            rerun::Collection<int32_t> i32;

            rerun::Collection<int64_t> i64;

            rerun::Collection<rerun::half> f16;

            rerun::Collection<float> f32;

            rerun::Collection<double> f64;

            rerun::Collection<uint8_t> jpeg;

            rerun::Collection<uint8_t> nv12;

            TensorBufferData() {
                std::memset(reinterpret_cast<void*>(this), 0, sizeof(TensorBufferData));
            }

            ~TensorBufferData() {}

            void swap(TensorBufferData& other) noexcept {
                // This bitwise swap would fail for self-referential types, but we don't have any of those.
                char temp[sizeof(TensorBufferData)];
                void* otherbytes = reinterpret_cast<void*>(&other);
                void* thisbytes = reinterpret_cast<void*>(this);
                std::memcpy(temp, thisbytes, sizeof(TensorBufferData));
                std::memcpy(thisbytes, otherbytes, sizeof(TensorBufferData));
                std::memcpy(otherbytes, temp, sizeof(TensorBufferData));
            }
        };
    } // namespace detail

    /// **Datatype**: The underlying storage for a `Tensor`.
    ///
    /// Tensor elements are stored in a contiguous buffer of a single type.
    struct TensorBuffer {
        TensorBuffer() : _tag(detail::TensorBufferTag::None) {}

        /// Copy constructor
        TensorBuffer(const TensorBuffer& other) : _tag(other._tag) {
            switch (other._tag) {
                case detail::TensorBufferTag::U8: {
                    using TypeAlias = rerun::Collection<uint8_t>;
                    new (&_data.u8) TypeAlias(other._data.u8);
                } break;
                case detail::TensorBufferTag::U16: {
                    using TypeAlias = rerun::Collection<uint16_t>;
                    new (&_data.u16) TypeAlias(other._data.u16);
                } break;
                case detail::TensorBufferTag::U32: {
                    using TypeAlias = rerun::Collection<uint32_t>;
                    new (&_data.u32) TypeAlias(other._data.u32);
                } break;
                case detail::TensorBufferTag::U64: {
                    using TypeAlias = rerun::Collection<uint64_t>;
                    new (&_data.u64) TypeAlias(other._data.u64);
                } break;
                case detail::TensorBufferTag::I8: {
                    using TypeAlias = rerun::Collection<int8_t>;
                    new (&_data.i8) TypeAlias(other._data.i8);
                } break;
                case detail::TensorBufferTag::I16: {
                    using TypeAlias = rerun::Collection<int16_t>;
                    new (&_data.i16) TypeAlias(other._data.i16);
                } break;
                case detail::TensorBufferTag::I32: {
                    using TypeAlias = rerun::Collection<int32_t>;
                    new (&_data.i32) TypeAlias(other._data.i32);
                } break;
                case detail::TensorBufferTag::I64: {
                    using TypeAlias = rerun::Collection<int64_t>;
                    new (&_data.i64) TypeAlias(other._data.i64);
                } break;
                case detail::TensorBufferTag::F16: {
                    using TypeAlias = rerun::Collection<rerun::half>;
                    new (&_data.f16) TypeAlias(other._data.f16);
                } break;
                case detail::TensorBufferTag::F32: {
                    using TypeAlias = rerun::Collection<float>;
                    new (&_data.f32) TypeAlias(other._data.f32);
                } break;
                case detail::TensorBufferTag::F64: {
                    using TypeAlias = rerun::Collection<double>;
                    new (&_data.f64) TypeAlias(other._data.f64);
                } break;
                case detail::TensorBufferTag::JPEG: {
                    using TypeAlias = rerun::Collection<uint8_t>;
                    new (&_data.jpeg) TypeAlias(other._data.jpeg);
                } break;
                case detail::TensorBufferTag::NV12: {
                    using TypeAlias = rerun::Collection<uint8_t>;
                    new (&_data.nv12) TypeAlias(other._data.nv12);
                } break;
                case detail::TensorBufferTag::None: {
                } break;
            }
        }

        TensorBuffer& operator=(const TensorBuffer& other) noexcept {
            TensorBuffer tmp(other);
            this->swap(tmp);
            return *this;
        }

        TensorBuffer(TensorBuffer&& other) noexcept : TensorBuffer() {
            this->swap(other);
        }

        TensorBuffer& operator=(TensorBuffer&& other) noexcept {
            this->swap(other);
            return *this;
        }

        ~TensorBuffer() {
            switch (this->_tag) {
                case detail::TensorBufferTag::None: {
                    // Nothing to destroy
                } break;
                case detail::TensorBufferTag::U8: {
                    using TypeAlias = rerun::Collection<uint8_t>;
                    _data.u8.~TypeAlias();
                } break;
                case detail::TensorBufferTag::U16: {
                    using TypeAlias = rerun::Collection<uint16_t>;
                    _data.u16.~TypeAlias();
                } break;
                case detail::TensorBufferTag::U32: {
                    using TypeAlias = rerun::Collection<uint32_t>;
                    _data.u32.~TypeAlias();
                } break;
                case detail::TensorBufferTag::U64: {
                    using TypeAlias = rerun::Collection<uint64_t>;
                    _data.u64.~TypeAlias();
                } break;
                case detail::TensorBufferTag::I8: {
                    using TypeAlias = rerun::Collection<int8_t>;
                    _data.i8.~TypeAlias();
                } break;
                case detail::TensorBufferTag::I16: {
                    using TypeAlias = rerun::Collection<int16_t>;
                    _data.i16.~TypeAlias();
                } break;
                case detail::TensorBufferTag::I32: {
                    using TypeAlias = rerun::Collection<int32_t>;
                    _data.i32.~TypeAlias();
                } break;
                case detail::TensorBufferTag::I64: {
                    using TypeAlias = rerun::Collection<int64_t>;
                    _data.i64.~TypeAlias();
                } break;
                case detail::TensorBufferTag::F16: {
                    using TypeAlias = rerun::Collection<rerun::half>;
                    _data.f16.~TypeAlias();
                } break;
                case detail::TensorBufferTag::F32: {
                    using TypeAlias = rerun::Collection<float>;
                    _data.f32.~TypeAlias();
                } break;
                case detail::TensorBufferTag::F64: {
                    using TypeAlias = rerun::Collection<double>;
                    _data.f64.~TypeAlias();
                } break;
                case detail::TensorBufferTag::JPEG: {
                    using TypeAlias = rerun::Collection<uint8_t>;
                    _data.jpeg.~TypeAlias();
                } break;
                case detail::TensorBufferTag::NV12: {
                    using TypeAlias = rerun::Collection<uint8_t>;
                    _data.nv12.~TypeAlias();
                } break;
            }
        }

      public:
        // Extensions to generated type defined in 'tensor_buffer_ext.cpp'

        /// Construct a `TensorBuffer` from a `Collection<uint8_t>`.
        TensorBuffer(Collection<uint8_t> u8) : TensorBuffer(TensorBuffer::u8(std::move(u8))) {}

        /// Construct a `TensorBuffer` from a `Collection<uint16_t>`.
        TensorBuffer(Collection<uint16_t> u16) : TensorBuffer(TensorBuffer::u16(std::move(u16))) {}

        /// Construct a `TensorBuffer` from a `Collection<uint32_t>`.
        TensorBuffer(Collection<uint32_t> u32) : TensorBuffer(TensorBuffer::u32(std::move(u32))) {}

        /// Construct a `TensorBuffer` from a `Collection<uint64_t>`.
        TensorBuffer(Collection<uint64_t> u64) : TensorBuffer(TensorBuffer::u64(std::move(u64))) {}

        /// Construct a `TensorBuffer` from a `Collection<int8_t>`.
        TensorBuffer(Collection<int8_t> i8) : TensorBuffer(TensorBuffer::i8(std::move(i8))) {}

        /// Construct a `TensorBuffer` from a `Collection<int16_t>`.
        TensorBuffer(Collection<int16_t> i16) : TensorBuffer(TensorBuffer::i16(std::move(i16))) {}

        /// Construct a `TensorBuffer` from a `Collection<int32_t>`.
        TensorBuffer(Collection<int32_t> i32) : TensorBuffer(TensorBuffer::i32(std::move(i32))) {}

        /// Construct a `TensorBuffer` from a `Collection<int64_t>`.
        TensorBuffer(Collection<int64_t> i64) : TensorBuffer(TensorBuffer::i64(std::move(i64))) {}

        /// Construct a `TensorBuffer` from a `Collection<half>`.
        TensorBuffer(Collection<rerun::half> f16)
            : TensorBuffer(TensorBuffer::f16(std::move(f16))) {}

        /// Construct a `TensorBuffer` from a `Collection<float>`.
        TensorBuffer(Collection<float> f32) : TensorBuffer(TensorBuffer::f32(std::move(f32))) {}

        /// Construct a `TensorBuffer` from a `Collection<double>`.
        TensorBuffer(Collection<double> f64) : TensorBuffer(TensorBuffer::f64(std::move(f64))) {}

        /// Construct a `TensorBuffer` from any container type that has a `value_type` member and for which a
        /// `rerun::ContainerAdapter` exists.
        ///
        /// This constructor assumes the type of tensor buffer you want to use is defined by `TContainer::value_type`
        /// and then forwards the argument as-is to the appropriate `rerun::Container` constructor.
        /// \see rerun::ContainerAdapter, rerun::Container
        template <typename TContainer, typename value_type = traits::value_type_of_t<TContainer>>
        TensorBuffer(TContainer&& container)
            : TensorBuffer(Collection<value_type>(std::forward<TContainer>(container))) {}

        /// Number of elements in the buffer.
        ///
        /// You may NOT call this for JPEG buffers.
        size_t num_elems() const;

        void swap(TensorBuffer& other) noexcept {
            std::swap(this->_tag, other._tag);
            this->_data.swap(other._data);
        }

        static TensorBuffer u8(rerun::Collection<uint8_t> u8) {
            TensorBuffer self;
            self._tag = detail::TensorBufferTag::U8;
            new (&self._data.u8) rerun::Collection<uint8_t>(std::move(u8));
            return self;
        }

        static TensorBuffer u16(rerun::Collection<uint16_t> u16) {
            TensorBuffer self;
            self._tag = detail::TensorBufferTag::U16;
            new (&self._data.u16) rerun::Collection<uint16_t>(std::move(u16));
            return self;
        }

        static TensorBuffer u32(rerun::Collection<uint32_t> u32) {
            TensorBuffer self;
            self._tag = detail::TensorBufferTag::U32;
            new (&self._data.u32) rerun::Collection<uint32_t>(std::move(u32));
            return self;
        }

        static TensorBuffer u64(rerun::Collection<uint64_t> u64) {
            TensorBuffer self;
            self._tag = detail::TensorBufferTag::U64;
            new (&self._data.u64) rerun::Collection<uint64_t>(std::move(u64));
            return self;
        }

        static TensorBuffer i8(rerun::Collection<int8_t> i8) {
            TensorBuffer self;
            self._tag = detail::TensorBufferTag::I8;
            new (&self._data.i8) rerun::Collection<int8_t>(std::move(i8));
            return self;
        }

        static TensorBuffer i16(rerun::Collection<int16_t> i16) {
            TensorBuffer self;
            self._tag = detail::TensorBufferTag::I16;
            new (&self._data.i16) rerun::Collection<int16_t>(std::move(i16));
            return self;
        }

        static TensorBuffer i32(rerun::Collection<int32_t> i32) {
            TensorBuffer self;
            self._tag = detail::TensorBufferTag::I32;
            new (&self._data.i32) rerun::Collection<int32_t>(std::move(i32));
            return self;
        }

        static TensorBuffer i64(rerun::Collection<int64_t> i64) {
            TensorBuffer self;
            self._tag = detail::TensorBufferTag::I64;
            new (&self._data.i64) rerun::Collection<int64_t>(std::move(i64));
            return self;
        }

        static TensorBuffer f16(rerun::Collection<rerun::half> f16) {
            TensorBuffer self;
            self._tag = detail::TensorBufferTag::F16;
            new (&self._data.f16) rerun::Collection<rerun::half>(std::move(f16));
            return self;
        }

        static TensorBuffer f32(rerun::Collection<float> f32) {
            TensorBuffer self;
            self._tag = detail::TensorBufferTag::F32;
            new (&self._data.f32) rerun::Collection<float>(std::move(f32));
            return self;
        }

        static TensorBuffer f64(rerun::Collection<double> f64) {
            TensorBuffer self;
            self._tag = detail::TensorBufferTag::F64;
            new (&self._data.f64) rerun::Collection<double>(std::move(f64));
            return self;
        }

        static TensorBuffer jpeg(rerun::Collection<uint8_t> jpeg) {
            TensorBuffer self;
            self._tag = detail::TensorBufferTag::JPEG;
            new (&self._data.jpeg) rerun::Collection<uint8_t>(std::move(jpeg));
            return self;
        }

        static TensorBuffer nv12(rerun::Collection<uint8_t> nv12) {
            TensorBuffer self;
            self._tag = detail::TensorBufferTag::NV12;
            new (&self._data.nv12) rerun::Collection<uint8_t>(std::move(nv12));
            return self;
        }

        /// Return a pointer to u8 if the union is in that state, otherwise `nullptr`.
        const rerun::Collection<uint8_t>* get_u8() const {
            if (_tag == detail::TensorBufferTag::U8) {
                return &_data.u8;
            } else {
                return nullptr;
            }
        }

        /// Return a pointer to u16 if the union is in that state, otherwise `nullptr`.
        const rerun::Collection<uint16_t>* get_u16() const {
            if (_tag == detail::TensorBufferTag::U16) {
                return &_data.u16;
            } else {
                return nullptr;
            }
        }

        /// Return a pointer to u32 if the union is in that state, otherwise `nullptr`.
        const rerun::Collection<uint32_t>* get_u32() const {
            if (_tag == detail::TensorBufferTag::U32) {
                return &_data.u32;
            } else {
                return nullptr;
            }
        }

        /// Return a pointer to u64 if the union is in that state, otherwise `nullptr`.
        const rerun::Collection<uint64_t>* get_u64() const {
            if (_tag == detail::TensorBufferTag::U64) {
                return &_data.u64;
            } else {
                return nullptr;
            }
        }

        /// Return a pointer to i8 if the union is in that state, otherwise `nullptr`.
        const rerun::Collection<int8_t>* get_i8() const {
            if (_tag == detail::TensorBufferTag::I8) {
                return &_data.i8;
            } else {
                return nullptr;
            }
        }

        /// Return a pointer to i16 if the union is in that state, otherwise `nullptr`.
        const rerun::Collection<int16_t>* get_i16() const {
            if (_tag == detail::TensorBufferTag::I16) {
                return &_data.i16;
            } else {
                return nullptr;
            }
        }

        /// Return a pointer to i32 if the union is in that state, otherwise `nullptr`.
        const rerun::Collection<int32_t>* get_i32() const {
            if (_tag == detail::TensorBufferTag::I32) {
                return &_data.i32;
            } else {
                return nullptr;
            }
        }

        /// Return a pointer to i64 if the union is in that state, otherwise `nullptr`.
        const rerun::Collection<int64_t>* get_i64() const {
            if (_tag == detail::TensorBufferTag::I64) {
                return &_data.i64;
            } else {
                return nullptr;
            }
        }

        /// Return a pointer to f16 if the union is in that state, otherwise `nullptr`.
        const rerun::Collection<rerun::half>* get_f16() const {
            if (_tag == detail::TensorBufferTag::F16) {
                return &_data.f16;
            } else {
                return nullptr;
            }
        }

        /// Return a pointer to f32 if the union is in that state, otherwise `nullptr`.
        const rerun::Collection<float>* get_f32() const {
            if (_tag == detail::TensorBufferTag::F32) {
                return &_data.f32;
            } else {
                return nullptr;
            }
        }

        /// Return a pointer to f64 if the union is in that state, otherwise `nullptr`.
        const rerun::Collection<double>* get_f64() const {
            if (_tag == detail::TensorBufferTag::F64) {
                return &_data.f64;
            } else {
                return nullptr;
            }
        }

        /// Return a pointer to jpeg if the union is in that state, otherwise `nullptr`.
        const rerun::Collection<uint8_t>* get_jpeg() const {
            if (_tag == detail::TensorBufferTag::JPEG) {
                return &_data.jpeg;
            } else {
                return nullptr;
            }
        }

        /// Return a pointer to nv12 if the union is in that state, otherwise `nullptr`.
        const rerun::Collection<uint8_t>* get_nv12() const {
            if (_tag == detail::TensorBufferTag::NV12) {
                return &_data.nv12;
            } else {
                return nullptr;
            }
        }

        /// Returns the arrow data type this type corresponds to.
        static const std::shared_ptr<arrow::DataType>& arrow_datatype();

        /// Fills an arrow array builder with an array of this type.
        static rerun::Error fill_arrow_array_builder(
            arrow::DenseUnionBuilder* builder, const TensorBuffer* elements, size_t num_elements
        );

      private:
        detail::TensorBufferTag _tag;
        detail::TensorBufferData _data;
    };
} // namespace rerun::datatypes

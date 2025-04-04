#ifndef __BUFFER_HPP__
#define __BUFFER_HPP__

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Avoid including heavy non-standard headers in a header file.
// #include <bits/stdc++.h>
// #include <boost/program_options.hpp>

#define __XRT__

#ifdef __XRT__
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_device.h"
#endif

#include "debug_utils.hpp"

/*
This is a buffer wrapper that maps to a bo_buffer or other memory without performing a deep copy.
A copy (or mapping) does not duplicate the underlying memory; it only maps the pointer.
*/

// --------------------- Base class: bytes --------------------- //
class bytes {
protected:
    uint8_t* data_;
    size_t size_;
    bool is_owner_;
#ifdef __XRT__
    bool is_bo_owner_;
    xrt::bo* bo_;
#endif

public:
    // Default constructor (no data)
    bytes() : data_(nullptr), size_(0), is_owner_(false)
#ifdef __XRT__
        , is_bo_owner_(false), bo_(nullptr)
#endif
    {}

    // Copy constructor: shallow copy, does not take ownership.
    bytes(const bytes& other)
        : data_(other.data_), size_(other.size_), is_owner_(false)
#ifdef __XRT__
        , is_bo_owner_(false), bo_(nullptr)
#endif
    {}

    // Move constructor: transfers pointer and flags, then clears source.
    bytes(bytes&& other) noexcept
        : data_(other.data_), size_(other.size_), is_owner_(other.is_owner_)
#ifdef __XRT__
        , is_bo_owner_(other.is_bo_owner_), bo_(other.bo_)
#endif
    {
        other.data_ = nullptr;
        other.size_ = 0;
        other.is_owner_ = false;
#ifdef __XRT__
        other.is_bo_owner_ = false;
        other.bo_ = nullptr;
#endif
    }

    // Construct a buffer of given size (allocates memory and owns it).
    bytes(size_t size)
        : data_(new uint8_t[size]), size_(size), is_owner_(true)
#ifdef __XRT__
        , is_bo_owner_(false), bo_(nullptr)
#endif
    {}

    // Construct a buffer that maps an existing pointer (does not take ownership).
    bytes(uint8_t* data, size_t size)
        : data_(data), size_(size), is_owner_(false)
#ifdef __XRT__
        , is_bo_owner_(false), bo_(nullptr)
#endif
    {}

#ifdef __XRT__
    // Construct from an xrt::bo (maps the bo; does not claim ownership).
    bytes(xrt::bo& bo)
        : data_(bo.map<uint8_t*>()), size_(bo.size()), is_owner_(false),
          is_bo_owner_(false), bo_(&bo)
    {}

    // Construct a bo-backed buffer (allocates a new bo; takes ownership of the bo).
    bytes(size_t size, xrt::device& device, xrt::kernel& kernel, int group_id)
        : size_(size), is_owner_(false), is_bo_owner_(true)
    {
        bo_ = new xrt::bo(device, size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(group_id));
        data_ = bo_->map<uint8_t*>();
    }
#endif

    // Destructor: free owned memory or bo.
    virtual ~bytes() {
        if (is_owner_ && data_) {
            delete[] data_;
        }
#ifdef __XRT__
        if (is_bo_owner_ && bo_) {
            delete bo_;
        }
#endif
    }

    // Assignment operator: shallow copy (no ownership transfer)
    bytes& operator=(const bytes& other) {
        if (this != &other) {
            if (is_owner_ && data_) {
                delete[] data_;
            }
            data_ = other.data_;
            size_ = other.size_;
            is_owner_ = false;
#ifdef __XRT__
            is_bo_owner_ = false;
            bo_ = other.bo_;
#endif
        }
        return *this;
    }

    // Element access operators
    uint8_t& operator[](size_t index) {
        assert(data_ != nullptr);
        return data_[index];
    }

    const uint8_t& operator[](size_t index) const {
        assert(data_ != nullptr);
        return data_[index];
    }

    // Accessors
    size_t size() const { return size_; }
    uint8_t* data() const { return data_; }
    uint8_t* bdata() const { return data_; }
    uint8_t* begin() const { return data_; }
    uint8_t* end() const { return data_ + size_; }

    // Resize (only allowed if not a bo-mapped buffer)
    void resize(size_t size) {
#ifdef __XRT__
        assert(is_bo_owner_ == false && "Cannot resize a bo-mapped buffer");
#endif
        // special case: size is 0 and data is nullptr: a empty buffer
        // case 1: a owner buffer, delete the memory
        // case 2: a non-owner buffer, throw an error
        if ((data_ != nullptr) && (is_owner_ == false)) {
            throw std::runtime_error("Cannot resize a non-owner buffer");
        }
        if (is_owner_ && data_) {
            delete[] data_;
        }
        data_ = new uint8_t[size];
        size_ = size;
        is_owner_ = true;
    }

    // Free the buffer memory (only allowed if owned)
    void free() {
#ifdef __XRT__
        assert(is_bo_owner_ == false && "Cannot free a bo-mapped buffer");
#endif
        if (is_owner_ && data_) {
            delete[] data_;
        }
        data_ = nullptr;
        size_ = 0;
        is_owner_ = false;
#ifdef __XRT__
        is_bo_owner_ = false;
        bo_ = nullptr;
#endif
    }

    // Reserve and release are aliases for resize/free.
    void reserve(size_t size) { resize(size); }
    void release() { free(); }

    // // Cast the underlying data pointer to type T.
    // template<typename T>
    // T* cast_to() {
    //     return reinterpret_cast<T*>(data_);
    // }

    // Helper function to print the buffer.
    bool is_owner() const { return is_owner_; }
    bool is_bo_owner() const { return is_bo_owner_; }

#ifdef __XRT__
    // XRT-specific functions.
    void sync_to_device() {
        assert(bo_ != nullptr);
        bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void sync_from_device() {
        assert(bo_ != nullptr);
        bo_->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }

    xrt::bo& bo() {
        assert(bo_ != nullptr);
        return *bo_;
    }
#endif
};

// --------------------- Derived class template: buffer<T> --------------------- //
// This class wraps a data type T over the underlying byte buffer.
template<typename T>
class buffer : public bytes {
public:
    // Default constructor.
    buffer() : bytes() {}

    // Construct a buffer for count elements (allocates memory).
    buffer(size_t count) : bytes(count * sizeof(T)) {}

    // Construct from existing T* data (shallow mapping; does not take ownership).
    buffer(T* data, size_t count)
        : bytes(reinterpret_cast<uint8_t*>(data), count * sizeof(T)) {}

    // Shallow copy constructor.
    buffer(const buffer& other) : bytes(other) {}

#ifdef __XRT__
    // Construct from an xrt::bo.
    buffer(xrt::bo& bo) : bytes(bo) {}

    // Construct a bo-backed buffer for count elements.
    buffer(size_t count, xrt::device& device, xrt::kernel& kernel, int group_id)
        : bytes(count * sizeof(T), device, kernel, group_id) {}
#endif

    // --- New: Constructors from std::vector --- //

    // Construct from a const lvalue reference to a std::vector<T>.
    // This creates a shallow mapping (no deep copy); the caller must ensure that
    // the vector outlives this buffer.
    buffer(const std::vector<T>& vec)
        : bytes(reinterpret_cast<uint8_t*>(const_cast<T*>(vec.data())), vec.size() * sizeof(T))
    {
        // Ownership is not taken.
    }

    // Right-value (rvalue) constructor from std::vector<T>.
    // WARNING: This also creates a shallow mapping.
    // The caller must ensure that the vector is not used (and remains valid)
    // after constructing this buffer.
    buffer(std::vector<T>&& vec)
        : bytes(reinterpret_cast<uint8_t*>(vec.data()), vec.size() * sizeof(T))
    {
        // Do NOT attempt to adopt the vector's memory since std::vector does not provide a release() method.
        // Instead, document that the caller is responsible for maintaining the vector's lifetime.
    }

    // --- New: copy_from() overload that copies data from a std::vector --- //
    // Copies the contents of the vector into the buffer.
    // The buffer must have been allocated with the same number of elements.
    void copy_from(const std::vector<T>& vec) {
        if (vec.size() * sizeof(T) != this->size_) {
            throw std::runtime_error("Size mismatch in copy_from(vector)");
        }
        std::memcpy(data_, vec.data(), size_);
    }

    // --- Templated cast_to() --- //
    // Return a buffer<U> that reinterprets the underlying data as type U.
    template<typename U>
    buffer<U> cast_to() {
        // Calculate the number of U elements that can be mapped.
        size_t newCount = size_ / sizeof(U);
        return buffer<U>(reinterpret_cast<U*>(data_), newCount);
    }

    // Return a bytes view of this buffer (shallow copy).
    const bytes as_bytes() const {
        return *this;
    }

    // Element access operators.
    T& operator[](size_t index) {
        assert(data_ != nullptr);
        return reinterpret_cast<T*>(data_)[index];
    }

    const T& operator[](size_t index) const {
        assert(data_ != nullptr);
        return reinterpret_cast<T*>(data_)[index];
    }

    // Return the number of T elements in the buffer.
    size_t size() const { return size_ / sizeof(T); }

    // Return pointer to T data.
    T* data() const { return reinterpret_cast<T*>(data_); }
    T* begin() const { return reinterpret_cast<T*>(data_); }
    T* end() const { return reinterpret_cast<T*>(data_) + size(); }

    // Resize the buffer to hold count elements (only allowed if not bo-mapped).
    void resize(size_t count) {
        bytes::resize(count * sizeof(T));
    }

    // Reserve capacity for count elements.
    void reserve(size_t count) {
        bytes::reserve(count * sizeof(T));
    }

    // Set all elements to the given value.
    void memset(T value) {
        T* ptr = data();
        for (size_t i = 0; i < size(); i++) {
            ptr[i] = value;
        }
    }

    // Copy data from another bytes object (must have the same byte size).
    void copy_from(const bytes& other) {
        if (size_ != other.size()) {
            throw std::runtime_error("Size mismatch in copy_from(bytes)");
        }
        std::memcpy(data_, other.data(), size_);
    }

    // Copy data from another buffer (must have the same number of elements).
    void copy_from(const buffer<T>& other) {
        if (size() != other.size()) {
            throw std::runtime_error("Size mismatch in copy_from(buffer)");
        }
        std::memcpy(data_, other.bdata(), size_ * sizeof(T)); // size_ is in bytes.
    }

    // Copy data from a pointer and provided size.
    void copy_from(T* data, size_t size) {
        if (size > this->size()) {
            throw std::runtime_error("Size mismatch in copy_from(pointer)");
        }
        std::memcpy(data_, data, size * sizeof(T));
    }

    bytes& as_bytes() {
        return *this;
    }

};

#endif // __BUFFER_HPP__
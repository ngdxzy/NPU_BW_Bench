#ifndef __VECTOR_VIEW_HPP__
#define __VECTOR_VIEW_HPP__

#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>
#include "debug_utils.hpp"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_device.h"
/*
This is a vector wrapper that allows a vector like object that can be mapped to a bo_buffer

*/
template<typename T>
class vector {
private:
    T* data_;
    size_t size_;
    bool is_owner_;
    bool is_bo_owner_;
    xrt::bo* bo_;

public:
    // Constructor from a pointer range
    // If copy is true, the vector owns the data
    vector(T* start, T* end, bool copy = true) : data_(start), size_(end - start), is_owner_(copy), is_bo_owner_(false), bo_(nullptr) {
        if (copy) {
            data_ = new T[size_];
            memcpy(data_, start, size_ * sizeof(T));
        }
        else{
            data_ = start;
            size_ = end - start;
            is_owner_ = false;
        }
    }
    // ********************** Constructors **********************
    // Constructor from a bo
    vector(xrt::bo& bo) : data_(bo.map<T*>()), size_(bo.size() / sizeof(T)), is_owner_(false), is_bo_owner_(false), bo_(&bo) {
        LOG_VERBOSE(3, "Vector created from bo");
    }

    // Constructor from create_bo_vector
    vector(size_t size, xrt::device& device, xrt::kernel& kernel, int group_id) : size_(size), is_owner_(false), is_bo_owner_(true) {
        bo_ = new xrt::bo(device, size * sizeof(T), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(group_id));
        data_ = bo_->map<T*>();
        LOG_VERBOSE(3, "Vector created together with bo");
    }

    // Constructor from a std::vector
    // If copy is true, the vector owns the data
    vector(std::vector<T> vec, bool copy = true) : data_(vec.data()), size_(vec.size()), is_owner_(copy), is_bo_owner_(false), bo_(nullptr) {
        if (copy) {
            data_ = new T[size_];
            memcpy(data_, vec.data(), size_ * sizeof(T));
        }
        else{
            data_ = vec.data();
            size_ = vec.size();
        }
    }

    // constructor from another vector
    vector(const vector<T>& vec) : data_(vec.data_), size_(vec.size_), is_owner_(false), is_bo_owner_(false), bo_(vec.bo_) {}

    // constructor from a right value
    vector(vector<T>&& vec) : data_(vec.data_), size_(vec.size_), is_owner_(vec.is_owner_), is_bo_owner_(vec.is_bo_owner_), bo_(vec.bo_) {
        vec.data_ = nullptr;
        vec.size_ = 0;
        vec.is_owner_ = false;
        vec.is_bo_owner_ = false;
        vec.bo_ = nullptr;
    }

    // Constructor from a size
    // The vector owns the data
    vector(size_t size) : data_(new T[size]), size_(size), is_owner_(true), is_bo_owner_(false), bo_(nullptr) {}

    // Constructor to initialize with size and value
    vector(size_t size, const T& value) : data_(new T[size]), size_(size), is_owner_(true), is_bo_owner_(false), bo_(nullptr) {
        for (size_t i = 0; i < size; i++) {
            data_[i] = value;
        }
    }

    // Constructor from a pointer range
    // If copy is true, the vector owns the data
    vector(T* data, size_t size, bool copy = true) : data_(data), size_(size), is_owner_(copy), is_bo_owner_(false), bo_(nullptr) {
        if (copy) {
            data_ = new T[size];
            memcpy(data_, data, size * sizeof(T));
        }
        else{
            data_ = data;
            size_ = size;
        }
    }

    // Default constructor
    // The vector does not own the data
    vector() : data_(nullptr), size_(0), is_owner_(false), is_bo_owner_(false), bo_(nullptr) {}


    // Destructor
    ~vector() {
        LOG_VERBOSE_IF(3, is_owner_ || is_bo_owner_, "Deleting vector with is_owner: " << is_owner_ << " and is_bo_owner: " << is_bo_owner_ << " and data: " << data_ << " and size: " << size_);
        if (is_owner_) {
            if (data_ != nullptr){
                // std::cout << "deleting vector" << data_ << std::endl;
                delete[] data_;
                data_ = nullptr;
            }
        }
        if (is_bo_owner_){
            delete bo_;
        }
    }

    // ********************** Accessors **********************
    bool is_owner() const { return is_owner_; }
    bool is_bo_owner() const { return is_bo_owner_; }

    xrt::bo& bo() const { assert(bo_ != nullptr); return *bo_; }

    size_t size() const { return size_; }
    size_t size_bytes() const { return size_ * sizeof(T); }
    T* data() const { return data_; }
    T* begin() const { return data_; }
    T* end() const { return data_ + size_; }
    // Access operator
    T& operator[](size_t index) {
        assert(this->data_ != nullptr);
        if (index >= size_) throw std::out_of_range("Index out of range");
        return data_[index];
    }

    // Const access operator
    const T& operator[](size_t index) const {
        assert(this->data_ != nullptr);
        if (index >= size_) throw std::out_of_range("Index out of range");
        return data_[index];
    }

    // Assignment operator
    // The vector does not own the data, pointer is copied
    vector<T>& operator=(const vector<T>& vec) {
        assert(is_bo_owner_ == false);
        if (is_owner_) {
            delete[] data_;
        }
        data_ = vec.data_;
        size_ = vec.size_;
        is_owner_ = false;
        bo_ = vec.bo_;
        is_bo_owner_ = false;
        return *this;
    }


    // Remap the vector to a bo
    // If size is -1, the vector size is set to the size of the new pointer
    void remap(xrt::bo& bo) {
        assert(is_bo_owner_ == false);
        if (is_owner_) {
            delete[] data_;
        }
        data_ = bo.map<T*>();
        size_ = bo.size() / sizeof(T);
        is_owner_ = false;
        bo_ = &bo;
    }

    // Remap the vector to a new pointer
    // If size is -1, the vector size is set to the size of the new pointer
    void remap(T* ptr, size_t size = -1) {
        assert(is_bo_owner_ == false);
        if (is_owner_) {
            delete[] data_;
        }
        data_ = ptr;
        if (size != -1) {
            size_ = size;
        }
        is_owner_ = false;
    }

    // Deep copy from another vector
    void copy_from(const vector<T>& vec) {
        assert(size_ == vec.size());
        assert(this->data_ != nullptr);
        memcpy(data_, vec.data(), size_ * sizeof(T));
    }

    // Deep copy from a pointer
    void copy_from(T* p) {
        assert(this->data_ != nullptr);
        memcpy(data_, p, size_ * sizeof(T));
    }

    void copy_from(T* p, size_t size, size_t offset = 0) {
        assert(this->data_ != nullptr);
        memcpy(data_ + offset, p, size * sizeof(T));
    }

    void resize(size_t size){
        assert(is_bo_owner_ == false);
        if (is_owner_){
            delete[] data_;
        }
        data_ = new T[size];
        size_ = size;
        is_owner_ = true;
    }

    // Read from a file
    // The vector size is set to the original size
    void from_file(const std::string& filename) {
        assert(this->data_ != nullptr);
        std::ifstream file(filename, std::ios::in | std::ios::binary);
        if (!file.is_open()) {
            throw std::ios_base::failure("Failed to open file.");
        }
        file.read(reinterpret_cast<char*>(data_), size_ * sizeof(T));
        file.close();
    }

    // Read from a file
    // The vector size is set to the new size (in elements)
    void from_file(const std::string& filename, size_t size) {
        assert(this->data_ != nullptr);
        LOG_VERBOSE(3, "Opening file: " << filename);
        size_ = size;
        std::ifstream file(filename, std::ios::in | std::ios::binary);
        if (!file.is_open()) {
            throw std::ios_base::failure("Failed to open file.");
        }
        file.read(reinterpret_cast<char*>(data_), size_ * sizeof(T));
        file.close();
    }


    // Read from a file
    // The vector size is set to the new size (in elements)
    // The file is read from the offset (in elements)
    void from_file(const std::string& filename, size_t size, size_t offset) {
        assert(this->data_ != nullptr);
        LOG_VERBOSE(3, "Opening file: " << filename);
        size_ = size;
        std::ifstream file(filename, std::ios::in | std::ios::binary);
        if (!file.is_open()) {
            throw std::ios_base::failure("Failed to open file.");
        }
        file.seekg(offset * sizeof(T));
        file.read(reinterpret_cast<char*>(data_), size_ * sizeof(T));
        file.close();
    }

    void acquire(int size){
        assert(is_bo_owner_ == false);
        if (is_owner_){
            delete[] data_;
        }
        data_ = new T[size];
        size_ = size;
        is_owner_ = true;
    }

    void release(){
        assert(is_bo_owner_ == false);
        if (is_owner_){
            delete[] data_;
        }
        data_ = nullptr;
        size_ = 0;
        is_owner_ = false;
    }

    void memset(T value){
        assert(this->data_ != nullptr);
        for (size_t i = 0; i < size_; i++){
            data_[i] = value;
        }
    }

    void sync_to_device(){
        assert(bo_ != nullptr);
        bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void sync_from_device(){
        assert(bo_ != nullptr);
        bo_->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }
    
    // dummy function to sync from device
    void sync_to_host(){
        sync_from_device();
    }
};

#endif
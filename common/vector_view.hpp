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

template<typename T>
class vector {
private:
    T* data_;
    size_t size_;
    bool is_owner_;

public:
    // Constructor from a pointer range
    // If copy is true, the vector owns the data
    vector(T* start, T* end, bool copy = true) : data_(start), size_(end - start), is_owner_(copy) {
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

    // Constructor from a std::vector
    // If copy is true, the vector owns the data
    vector(std::vector<T> vec, bool copy = true) : data_(vec.data()), size_(vec.size()), is_owner_(copy) {
        if (copy) {
            data_ = new T[size_];
            memcpy(data_, vec.data(), size_ * sizeof(T));
        }
        else{
            data_ = vec.data();
            size_ = vec.size();
        }
    }

    // Constructor to initialize with size and value
    vector(size_t size, const T& value) : data_(new T[size]), size_(size), is_owner_(true) {
        for (size_t i = 0; i < size; i++) {
            data_[i] = value;
        }
    }


    vector(size_t size, const double & value) : data_(new T[size]), size_(size), is_owner_(true) {
        for (size_t i = 0; i < size; i++) {
            data_[i] = (T)value;
        }
    }

    vector(size_t size, const float & value) : data_(new T[size]), size_(size), is_owner_(true) {
        for (size_t i = 0; i < size; i++) {
            data_[i] = (T)value;
        }
    }

    // Copy constructor
    vector(const vector<T>& other) : data_(other.data_), size_(other.size_), is_owner_(false) {
    }

    // Constructor from a size
    // The vector owns the data
    vector(size_t size) : data_(new T[size]), size_(size), is_owner_(true) {}

    // Constructor from another vector
    // The vector does not own the data
    vector(vector<T>& vec) : data_(vec.data()), size_(vec.size()), is_owner_(false) {}

    // Constructor from a pointer range
    // If copy is true, the vector owns the data
    vector(T* data, size_t size, bool copy = true) : data_(data), size_(size), is_owner_(copy) {
        if (copy) {
            data_ = new T[size];
            memcpy(data_, data, size * sizeof(T));
        }
        else{
            data_ = data;
            size_ = size;
            is_owner_ = false;
        }
    }

    // Default constructor
    // The vector does not own the data
    vector() : data_(nullptr), size_(0), is_owner_(false) {}

    // Constructor from a file
    // The vector owns the data
    // The vector size is set to the file size / sizeof(T)
    vector(const std::string& filename) {
        std::ifstream file(filename, std::ios::in | std::ios::binary);

        // Check if the file was successfully opened
        if (!file.is_open()) {
            throw std::ios_base::failure("Failed to open file " + filename);
        }
        // Move to the end of the file
        file.seekg(0, std::ios::end);

        // Get the position, which is the file size
        std::size_t fileSize = file.tellg();

        // Return the file pointer to the beginning (optional, if needed later)
        file.seekg(0, std::ios::beg);

        size_ = fileSize / sizeof(T);
        data_ = new T[size_];
        is_owner_ = true;
        file.read(reinterpret_cast<char*>(data_), size_ * sizeof(T));
        file.close();
    }

    // Destructor
    ~vector() {
        if (is_owner_) {
            if (data_ != nullptr){
                // std::cout << "deleting vector" << data_ << std::endl;
                delete[] data_;
                data_ = nullptr;
            }
        }
    }

    bool is_owner() const { return is_owner_; }

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
        if (is_owner_) {
            delete[] data_;
        }
        data_ = vec.data_;
        size_ = vec.size_;
        is_owner_ = false;
        return *this;
    }

    size_t size() const { return size_; }
    size_t size_bytes() const { return size_ * sizeof(T); }
    T* data() const { return data_; }
    T* begin() const { return data_; }
    T* end() const { return data_ + size_; }

    // Remap the vector to a new pointer
    // If size is -1, the vector size is set to the size of the new pointer
    void remap(T* ptr, size_t size = -1) {
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
        if (is_owner_){
            delete[] data_;
        }
        data_ = new T[size];
        size_ = size;
        is_owner_ = true;
    }

    void release(){
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

};

#endif

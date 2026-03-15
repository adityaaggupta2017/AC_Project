#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <fstream>
#include <iomanip>

// CUDA error checking
inline void cuda_check(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
    exit(1);
  }
}

// Convert uint64_t to hex string
inline std::string u64_hex(uint64_t val) {
  std::ostringstream oss;
  oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << val;
  return oss.str();
}

// Append a row to CSV file
inline void csv_append_row(const std::string& filename, const std::string& row) {
  std::ofstream f(filename, std::ios::app);
  if (!f.is_open()) {
    fprintf(stderr, "Error: cannot open file %s\n", filename.c_str());
    return;
  }
  f << row << "\n";
  f.close();
}

// Initialize CSV file with header
inline void csv_init(const std::string& filename) {
  std::ofstream f(filename);
  if (!f.is_open()) {
    fprintf(stderr, "Error: cannot create file %s\n", filename.c_str());
    return;
  }
  f << "cipher,platform,variant,unknown_bits,keys_tested,seconds,keys_per_sec,found_key\n";
  f.close();
}

// Convert byte array to hex string
inline std::string bytes_to_hex(const uint8_t* data, int len) {
  std::ostringstream oss;
  oss << "0x";
  for (int i = 0; i < len; i++) {
    oss << std::hex << std::setw(2) << std::setfill('0') << (int)data[i];
  }
  return oss.str();
}

// Parse hex string to byte array
inline void hex_to_bytes(const char* hex_str, uint8_t* out, int out_len) {
  if (strncmp(hex_str, "0x", 2) == 0) {
    hex_str += 2;
  }
  
  int len = strlen(hex_str);
  if (len > out_len * 2) {
    len = out_len * 2;
  }
  
  for (int i = 0; i < out_len; i++) {
    out[i] = 0;
  }
  
  for (int i = 0; i < len; i += 2) {
    int hi = (hex_str[i] >= 'a') ? (hex_str[i] - 'a' + 10) : 
             (hex_str[i] >= 'A') ? (hex_str[i] - 'A' + 10) : (hex_str[i] - '0');
    int lo = (i + 1 < len) ?
             ((hex_str[i+1] >= 'a') ? (hex_str[i+1] - 'a' + 10) :
              (hex_str[i+1] >= 'A') ? (hex_str[i+1] - 'A' + 10) : (hex_str[i+1] - '0')) : 0;
    out[i / 2] = (hi << 4) | lo;
  }
}

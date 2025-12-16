#pragma once
#include "csr.hpp"
#include "pim_config.h"

/*
 * PIM Emulator Entry Point
 * 
 * High-level API that encapsulates the PIM-Emu pipeline.
 * Handles filter, quantization, and format transformations based on parameters.
 */

/**
 * Apply PIM filtering only (no quantization).
 * 
 * @param X Input CSR matrix
 * @param params PIM parameters specifying filter mode and threshold
 * @return Filtered CSR matrix
 */
CSR pim_filter_only(const CSR& X, const PIMParams& params);

/**
 * Apply PIM filtering and quantization.
 * (Future implementation)
 * 
 * @param X Input CSR matrix
 * @param params PIM parameters specifying filter mode, threshold, and quant mode
 * @return Processed CSR matrix
 */
CSR pim_filter_and_quant(const CSR& X, const PIMParams& params);


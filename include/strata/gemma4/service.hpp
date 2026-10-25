#pragma once

// Single-residence, token-ID protocol. The caller owns tokenization and I/O.
// All embedding, attention, FFN, projection, picking and MTP execute on CUDA.
// This entry is independent of the fixed-slot historical benchmark drivers.
int strata_gemma4_native_serve(int argc, char** argv);

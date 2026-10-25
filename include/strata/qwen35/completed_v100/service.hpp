#pragma once

// Native line-protocol service for Dense Qwen3.5 and Qwen3.6 GGUF models.
// The caller owns argv and invokes this on one command-owner thread.
int strata_qwen_v100_completed_serve(int argc, char** argv);

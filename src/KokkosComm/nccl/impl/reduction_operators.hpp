// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include <nccl.h>
#include "nccl/nccl_space.hpp"

namespace KokkosComm {
namespace Impl {
  constexpr ncclRedOp_t Sum_    (NcclSpace const&) noexcept { return ncclSum; }
  constexpr ncclRedOp_t Prod_   (NcclSpace const&) noexcept { return ncclProd; }
  constexpr ncclRedOp_t Max_    (NcclSpace const&) noexcept { return ncclMax; }
  constexpr ncclRedOp_t Min_    (NcclSpace const&) noexcept { return ncclMin; }
  constexpr ncclRedOp_t Average_(NcclSpace const&) noexcept { return ncclAvg; }
}
}

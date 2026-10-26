// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <nccl.h>

#include "concepts.hpp"
#include "nccl/nccl_space.hpp"

namespace KokkosComm{
  // Fixed width integer types
  template<> inline const MPI_Datatype datatype<NcclSpace, std::int8_t>   = ncclInt8;
  template<> inline const MPI_Datatype datatype<NcclSpace, std::uint8_t>  = ncclUint8;
  template<> inline const MPI_Datatype datatype<NcclSpace, std::int32_t>  = ncclInt32;
  template<> inline const MPI_Datatype datatype<NcclSpace, std::uint32_t> = ncclUint32;
  template<> inline const MPI_Datatype datatype<NcclSpace, std::int64_t>  = ncclInt64;
  template<> inline const MPI_Datatype datatype<NcclSpace, std::uint64_t> = ncclUint64;

  // Special cases : char and std::byte
  template<> inline const MPI_Datatype datatype<NcclSpace, char>          = ncclChar; 
  
  // Floating point types
  template<> inline const MPI_Datatype datatype<NcclSpace, float>         = ncclFloat;
  template<> inline const MPI_Datatype datatype<NcclSpace, double>        = ncclDouble;
}

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include "../../concepts.hpp"
#include "../mpi_space.hpp"

namespace KokkosComm{
  // Fixed width integer types
  template<> inline const MPI_Datatype datatype<MpiSpace, std::int8_t>     = MPI_INT8_T;
  template<> inline const MPI_Datatype datatype<MpiSpace, std::uint8_t>    = MPI_UINT8_T;
  template<> inline const MPI_Datatype datatype<MpiSpace, std::int16_t>    = MPI_INT16_T;
  template<> inline const MPI_Datatype datatype<MpiSpace, std::uint16_t>   = MPI_UINT16_T;
  template<> inline const MPI_Datatype datatype<MpiSpace, std::int32_t>    = MPI_INT32_T;
  template<> inline const MPI_Datatype datatype<MpiSpace, std::uint32_t>   = MPI_UINT32_T;
  template<> inline const MPI_Datatype datatype<MpiSpace, std::int64_t>    = MPI_INT64_T;
  template<> inline const MPI_Datatype datatype<MpiSpace, std::uint64_t>   = MPI_UINT64_T;

  // Special cases : char and std::byte
  template<> inline const MPI_Datatype datatype<MpiSpace, std::byte>       = MPI_BYTE; 
  template<> inline const MPI_Datatype datatype<MpiSpace, char>            = MPI_CHAR; 
  
  // Floating point types
  template<> inline const MPI_Datatype datatype<MpiSpace, float>           = MPI_FLOAT;
  template<> inline const MPI_Datatype datatype<MpiSpace, double>          = MPI_DOUBLE;
  template<> inline const MPI_Datatype datatype<MpiSpace, long double>     = MPI_LONG_DOUBLE;

  // Complex numbers
#if defined(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI)
  template<> inline const MPI_Datatype datatype<MpiSpace, Kokkos::complex<float>>   = MPI_CXX_COMPLEX;
  template<> inline const MPI_Datatype datatype<MpiSpace, Kokkos::complex<double>>  = MPI_CXX_DOUBLE_COMPLEX;
#else
  template<> inline const MPI_Datatype datatype<MpiSpace, Kokkos::complex<float>>   = MPI_COMPLEX;
  template<> inline const MPI_Datatype datatype<MpiSpace, Kokkos::complex<double>>  = MPI_DOUBLE_COMPLEX;
#endif
}

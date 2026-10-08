// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include "concepts.hpp"
#include "mpi/mpi_space.hpp"

namespace KokkosComm{
namespace Impl{
 
  // Fixed width integer types
  template<> inline constexpr MPI_Datatype datatype<MPISpace, std::int8_t>     = MPI_INT8_T;
  template<> inline constexpr MPI_Datatype datatype<MPISpace, std::uint8_t>    = MPI_UINT8_T;
  template<> inline constexpr MPI_Datatype datatype<MPISpace, std::int16_t>    = MPI_INT16_T;
  template<> inline constexpr MPI_Datatype datatype<MPISpace, std::uint16_t>   = MPI_UINT16_T;
  template<> inline constexpr MPI_Datatype datatype<MPISpace, std::int32_t>    = MPI_INT32_T;
  template<> inline constexpr MPI_Datatype datatype<MPISpace, std::uint32_t>   = MPI_UINT32_T;
  template<> inline constexpr MPI_Datatype datatype<MPISpace, std::int64_t>    = MPI_INT64_T;
  template<> inline constexpr MPI_Datatype datatype<MPISpace, std::uint64_t>   = MPI_UINT64_T;

  // Special cases : char and std::byte
  template<> inline constexpr MPI_Datatype datatype<MPISpace, std::byte>       = MPI_BYTE; 
  template<> inline constexpr MPI_Datatype datatype<MPISpace, char>            = MPI_CHAR; 
  
  // Floating point types
  template<> inline constexpr MPI_Datatype datatype<MPISpace, float>           = MPI_FLOAT;
  template<> inline constexpr MPI_Datatype datatype<MPISpace, double>          = MPI_DOUBLE;
  template<> inline constexpr MPI_Datatype datatype<MPISpace, long double>     = MPI_LONG_DOUBLE;

  // Complex numbers
#if defined(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI)
  template<> inline constexpr MPI_Datatype datatype<MPISpace, Kokkos::complex<float>>   = MPI_CXX_COMPLEX;
  template<> inline constexpr MPI_Datatype datatype<MPISpace, Kokkos::complex<double>>  = MPI_CXX_DOUBLE_COMPLEX;
#else
  template<> inline constexpr MPI_Datatype datatype<MPISpace, Kokkos::complex<float>>   = MPI_COMPLEX;
  template<> inline constexpr MPI_Datatype datatype<MPISpace, Kokkos::complex<double>>  = MPI_DOUBLE_COMPLEX;
#endif
}
}

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include "concepts.hpp"

namespace KokkosComm {
namespace Impl
{
  template<typename...>
  struct undefined_t;
}

/// @brief Specializable template variable mapping a C++-native data type to it's communication space equivalent
///
/// @note If the mapping does not exist, the variable can't be instantiated
///
/// Non-system data types (i.e. the data types not natively supported by `C`) are not convertible. This notably
/// includes user-defined types.
///
/// When `C` is:
/// - `MpiSpace`, returns the corresponding `MPI_Datatype` type.
/// - `NcclSpace`, returns the corresponding `ncclDataType_t` type.
///
/// @tparam C The target communication space backend to use for data type conversion.
/// @tparam T The C++-native data type to convert.
/// @returns The communication space representation of the C++-native data type.
template<CommunicationSpace C, typename T>
extern Impl::undefined_t<C,T> datatype;

/// @returns The communication space representation of the Kokkos View value type.
template <CommunicationSpace C, KokkosView V>
[[nodiscard]] constexpr auto datatype_for([[maybe_unused]] const V& view) -> typename C::datatype_type {
  return datatype<C, std::remove_cvref_t<typename V::value_type>>;
}

/// @returns The communication space representation of the Kokkos View value type.
template <CommunicationSpace C, KokkosView V>
[[nodiscard]] constexpr auto datatype_for([[maybe_unused]] C&& comm, [[maybe_unused]] const V& view) ->
    typename C::datatype_type {
  return datatype<C, std::remove_cvref_t<typename V::value_type>>;
}

}  // namespace KokkosComm

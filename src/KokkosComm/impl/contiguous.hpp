// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstddef>
#include <string>
#include <utility>

#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>

namespace KokkosComm::Impl {

template <KokkosView View>
using contiguous_view_t = Kokkos::View<
      typename View::non_const_data_type,
      typename View::execution_space::array_layout,
      typename View::memory_space>;

template <typename Exec, typename View, std::size_t... I>
auto allocate_contiguous_for_impl(const Exec& exec, const std::string& label, const View& v, std::index_sequence<I...>) 
-> contiguous_view_t<View> 
{
  return contiguous_view_t<View>(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, label), v.extent(Is)...);
}

/// @brief Allocate a contiguous View suitable for packing a non-contiguous View.
/// @tparam Exec A Kokkos Execution Space type.
/// @tparam View A Kokkos View type.
/// @param exec The execution space instance in which to perform the view allocation.
/// @param v The View to make a suitable contiguous allocation for.
/// @param label The label to give to the allocated contiguous View. Defaults to "contiguous_view".
template <KokkosExecutionSpace Exec, KokkosView View>
auto allocate_contiguous_for(const Exec& exec, const std::string& label, const View& v) -> contiguous_view_t<View> {
  // Unpack `v` extents into the `ContigView` constructor
  return allocate_contiguous_for_impl(exec, label, v, std::make_index_sequence<rank<View>()>{});
}

}  // namespace KokkosComm::Impl

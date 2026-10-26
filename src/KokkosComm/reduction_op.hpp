// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>
#include <KokkosComm/concepts.hpp>

namespace KokkosComm {

namespace {

// clang-format off
#define DECL_REDUCTION_OP_FOR(tag)                                                            \
    struct tag##_fn {                                                                         \
        template <CommunicationSpace Cs>                                                      \
        requires requires(Cs&& cs) { tag##_(std::forward<Cs>(cs)); }                          \
        constexpr auto operator()(Cs&& cs) const                                              \
            noexcept(noexcept(tag##_(std::forward<Cs>(cs))))                                  \
        {                                                                                     \
            return tag##_(std::forward<Cs>(cs));                                              \
        }                                                                                     \
                                                                                              \
        template <CommunicationSpace Cs>                                                      \
        requires (!requires(Cs&& cs) { tag##_(std::forward<Cs>(cs)); })                       \
        constexpr auto operator()(Cs&& cs) const                                              \
        {                                                                                     \
          static_assert(std::is_void_v<Cs>, "KokkosComm::" #tag                               \
              " : operator not implemented for the provided communication space");            \
        }                                                                                     \ 
    } inline constexpr tag{};                                                                 \                                                        
                                                                                              \
    template<> inline constexpr bool Impl::is_reduction_operator_v<tag##_fn> = true
}  // namespace
// clang-format on

DECL_REDUCTION_OP_FOR(BAnd);
DECL_REDUCTION_OP_FOR(BOr);
DECL_REDUCTION_OP_FOR(BXor);
DECL_REDUCTION_OP_FOR(LAnd);
DECL_REDUCTION_OP_FOR(LOr);
DECL_REDUCTION_OP_FOR(LXor);
DECL_REDUCTION_OP_FOR(Max);
DECL_REDUCTION_OP_FOR(MaxLoc);
DECL_REDUCTION_OP_FOR(Min);
DECL_REDUCTION_OP_FOR(MinLoc);
DECL_REDUCTION_OP_FOR(Sum);
DECL_REDUCTION_OP_FOR(Prod);
DECL_REDUCTION_OP_FOR(Average);

}  // namespace KokkosComm

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include "../mpi_space.hpp"

namespace KokkosComm {
namespace Impl 
{
  constexpr MPI_Op BAnd_  (MpiSpace const&) noexcept { return  MPI_BAND;   }
  constexpr MPI_Op BOr_   (MpiSpace const&) noexcept { return  MPI_BOR;    }
  constexpr MPI_Op BXor_  (MpiSpace const&) noexcept { return  MPI_BXOR;   }
  constexpr MPI_Op LAnd_  (MpiSpace const&) noexcept { return  MPI_LAND;   }
  constexpr MPI_Op LOr_   (MpiSpace const&) noexcept { return  MPI_LOR;    }
  constexpr MPI_Op LXor_  (MpiSpace const&) noexcept { return  MPI_LXOR;   }
  constexpr MPI_Op Max_   (MpiSpace const&) noexcept { return  MPI_MAX;    }
  constexpr MPI_Op MaxLoc_(MpiSpace const&) noexcept { return  MPI_MAXLOC; }
  constexpr MPI_Op Min_   (MpiSpace const&) noexcept { return  MPI_MIN;    }
  constexpr MPI_Op MinLoc_(MpiSpace const&) noexcept { return  MPI_MINLOC; }
  constexpr MPI_Op Sum_   (MpiSpace const&) noexcept { return  MPI_SUM;    }
  constexpr MPI_Op Prod_  (MpiSpace const&) noexcept { return  MPI_PROD;   }
}
}

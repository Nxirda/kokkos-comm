// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include "mpi/mpi_space.hpp"

namespace KokkosComm {
namespace Impl 
{
  constexpr MPI_Op BAnd_  (mpi_space const&) noexcept { return  MPI_BAND;   }
  constexpr MPI_Op BOr_   (mpi_space const&) noexcept { return  MPI_BOR;    }
  constexpr MPI_Op BXor_  (mpi_space const&) noexcept { return  MPI_BXOR;   }
  constexpr MPI_Op LAnd_  (mpi_space const&) noexcept { return  MPI_LAND;   }
  constexpr MPI_Op LOr_   (mpi_space const&) noexcept { return  MPI_LOR;    }
  constexpr MPI_Op LXor_  (mpi_space const&) noexcept { return  MPI_LXOR;   }
  constexpr MPI_Op Max_   (mpi_space const&) noexcept { return  MPI_MAX;    }
  constexpr MPI_Op MaxLoc_(mpi_space const&) noexcept { return  MPI_MAXLOC; }
  constexpr MPI_Op Min_   (mpi_space const&) noexcept { return  MPI_MIN;    }
  constexpr MPI_Op MinLoc_(mpi_space const&) noexcept { return  MPI_MINLOC; }
  constexpr MPI_Op Sum_   (mpi_space const&) noexcept { return  MPI_SUM;    }
  constexpr MPI_Op Prod_  (mpi_space const&) noexcept { return  MPI_PROD;   }
}
}

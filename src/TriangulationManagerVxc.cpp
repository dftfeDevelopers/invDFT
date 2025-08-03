// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
/** @file triangulationManagerVxc.cc
 *
 *  @brief Source file for triangulationManager.h
 *
 *  @author Bikash Kanungo, Vishal Subramanian
 */

#include <TriangulationManagerVxc.h>
#include <constants.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include <meshGenUtils.h>

//#include "generateMesh.cc"
//#include "restartUtils.cc"

namespace invDFT {
namespace {
void getSystemExtent(const std::vector<std::vector<double>> &atomLocations,
                     const unsigned int coordIndexOffset,
                     const double innerDomainSize, std::vector<double> &lo,
                     std::vector<double> &hi) {
  const unsigned int N = atomLocations.size();
  lo.resize(3, 0.0);
  hi.resize(3, 0.0);
  for (unsigned int i = 0; i < 3; ++i) {
    lo[i] = atomLocations[0][i + coordIndexOffset];
    hi[i] = atomLocations[0][i + coordIndexOffset];
  }

  for (unsigned int i = 0; i < N; ++i) {
    for (unsigned int j = 0; j < 3; ++j) {
      if (atomLocations[i][j + coordIndexOffset] - innerDomainSize < lo[j])
        lo[j] = atomLocations[i][j + coordIndexOffset] - innerDomainSize;

      if (atomLocations[i][j + coordIndexOffset] + innerDomainSize > hi[j])
        hi[j] = atomLocations[i][j + coordIndexOffset] + innerDomainSize;
    }
  }
}

double
getMinCellSize(const parallel::distributed::Triangulation<3> &parallelMesh,
               const MPI_Comm &mpi_comm_domain) {
  unsigned int iCell = 0;
  double minCellDia = 0.0;
  for (auto &cellIter : parallelMesh.active_cell_iterators()) {
    if (iCell == 0) {
      minCellDia = cellIter->diameter();
    }
    if (minCellDia > cellIter->diameter())
      minCellDia = cellIter->diameter();

    iCell++;
  }
  MPI_Allreduce(MPI_IN_PLACE, &minCellDia, 1, MPI_DOUBLE, MPI_MIN,
                mpi_comm_domain);

  return minCellDia;
}

} // namespace

//
//
// constructor
//
TriangulationManagerVxc::TriangulationManagerVxc(
    const MPI_Comm &mpi_comm_parent, const MPI_Comm &mpi_comm_domain,
    const MPI_Comm &interpoolcomm, const MPI_Comm &interbandgroup_comm,
    const dftfe::dftParameters &dftParams,
    const inverseDFTParameters &inverseDFTParams,
    const dealii::parallel::distributed::Triangulation<3, 3>::Settings
        repartitionFlag)
    : d_mpi_comm_parent(mpi_comm_parent), d_mpi_comm_domain(mpi_comm_domain),
      d_interpoolcomm(interpoolcomm),
      d_interbandgroup_comm(interbandgroup_comm),
      d_parallelTriangulationUnmovedVxc(
          mpi_comm_domain, dealii::Triangulation<3, 3>::none, repartitionFlag),
      d_parallelTriangulationMovedVxc(
          mpi_comm_domain, dealii::Triangulation<3, 3>::none, repartitionFlag),
      d_serialTriangulationVxc(MPI_COMM_SELF),
      d_serialTriangulationElectrostaticsVxc(MPI_COMM_SELF),
      d_electrostaticsTriangulationRhoVxc(mpi_comm_domain),
      d_electrostaticsTriangulationDispVxc(mpi_comm_domain),
      d_electrostaticsTriangulationForceVxc(mpi_comm_domain),
      d_dftParams(dftParams), d_inverseDFTParams(inverseDFTParams),
      pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0)),
      computing_timer(pcout, TimerOutput::never, TimerOutput::wall_times) {}

//
// destructor
//
TriangulationManagerVxc::~TriangulationManagerVxc() {}

void TriangulationManagerVxc::generateParallelUnmovedMeshVxc(
    const std::vector<std::vector<double>> &atomPositions,
    dftfe::triangulationManager &dftTria) {
  std::vector<double> minCoord, maxCoord;
  // TODO var innerDomainSize is hard coded, read it from dftParams
  double innerDomainSize = d_inverseDFTParams.VxcInnerDomain;
  getSystemExtent(atomPositions, 2, innerDomainSize, minCoord, maxCoord);
  d_parallelTriangulationUnmovedVxc.clear();
  // copy the mesh
  //    generateMeshWithManualRepartitioning(d_parallelTriangulationUnmovedVxc);

  dftTria.generateMesh(d_parallelTriangulationUnmovedVxc,
                       d_serialTriangulationVxc,
                       d_parallelTriaCurrentRefinement,
                       d_serialTriaCurrentRefinement, false, true);
  /*
      AssertThrow(
        parallelMeshUnmoved.n_active_cells() ==
          d_parallelTriangulationUnmovedVxc.n_active_cells(),
        ExcMessage(
          "DFT-FE error:  Vxc mesh partitioning is not consistent with the wave
     function mesh "));

      AssertThrow(
        parallelMeshUnmoved.n_active_cells() ==
          d_parallelTriangulationUnmovedVxc.n_active_cells(),
        ExcMessage(
          "DFT-FE error:  Vxc mesh partitioning is not consistent with the wave
     function mesh "));
  */
  double minCellSize =
      getMinCellSize(d_parallelTriangulationUnmovedVxc, d_mpi_comm_domain);
  bool meshSatisfied = false;
  // TODO change the flag for refinement to ensure no change in parallel
  // layout

  unsigned int refineStepIter = 0;
  while (!meshSatisfied) {
    pcout << " refinement step = " << refineStepIter << "\n";
    meshSatisfied = true;
    for (auto &cellIter :
         d_parallelTriangulationUnmovedVxc.active_cell_iterators()) {
      if (cellIter->is_locally_owned()) {
        dealii::Point<3, double> center = cellIter->center();

        bool refineThisCell = true;
        for (unsigned int iDim = 0; iDim < 3; iDim++) {
          if ((center[iDim] > maxCoord[iDim]) ||
              (center[iDim] < minCoord[iDim])) {
            refineThisCell = false;
            break;
          }
        }
        if (refineThisCell &&
            (cellIter->minimum_vertex_distance() > (minCellSize + 1e-4))) {
          cellIter->set_refine_flag();
          meshSatisfied = false;
        }
      }
    }
    // DO not call repartition(). this ensures that the partitioning is
    // consistent
    d_parallelTriangulationUnmovedVxc.execute_coarsening_and_refinement();
    d_parallelTriangulationUnmovedVxc.repartition();
    meshSatisfied =
        Utilities::MPI::min((unsigned int)meshSatisfied, d_mpi_comm_domain);

    refineStepIter++;
  };

  if (d_dftParams.verbosity >= 4)
    pcout << std::endl
          << "Final triangulation number of elements: "
          << d_parallelTriangulationUnmovedVxc.n_global_active_cells()
          << std::endl;

  double minElemLength = d_dftParams.meshSizeOuterDomain;
  double maxElemLength = 0.0;
  typename parallel::distributed::Triangulation<3>::active_cell_iterator cell,
      endc;
  cell = d_parallelTriangulationUnmovedVxc.begin_active();
  endc = d_parallelTriangulationUnmovedVxc.end();
  unsigned int numLocallyOwnedCells = 0;
  for (; cell != endc; ++cell) {
    if (cell->is_locally_owned()) {
      numLocallyOwnedCells++;
      if (cell->minimum_vertex_distance() < minElemLength)
        minElemLength = cell->minimum_vertex_distance();

      if (cell->minimum_vertex_distance() > maxElemLength)
        maxElemLength = cell->minimum_vertex_distance();
    }
  }

  minElemLength = Utilities::MPI::min(minElemLength, d_mpi_comm_domain);
  maxElemLength = Utilities::MPI::max(maxElemLength, d_mpi_comm_domain);

  //
  // print out adaptive mesh metrics and check mesh generation synchronization
  // across pools
  //
  if (d_dftParams.verbosity >= 4) {
    pcout << "Vxc Triangulation generation summary: " << std::endl
          << " num elements: "
          << d_parallelTriangulationUnmovedVxc.n_global_active_cells()
          << ", min element length: " << minElemLength
          << ", max element length: " << maxElemLength << std::endl;
  }
}

void TriangulationManagerVxc::generateParallelMovedMeshVxc(
    const parallel::distributed::Triangulation<3> &parallelMeshUnmoved,
    const parallel::distributed::Triangulation<3> &parallelMeshMoved) {
  //        d_parallelTriangulationUnmovedVxc.clear();
  //        d_parallelTriangulationUnmovedVxc.copy_triangulation(parallelMeshUnmoved);
  //
  //
  //        d_parallelTriangulationMovedVxc.clear();
  //        d_parallelTriangulationMovedVxc.copy_triangulation(parallelMeshMoved);
  //
  //        if (d_dftParams.verbosity >= 4)
  //          pcout << std::endl
  //                << "Final triangulation number of elements: "
  //                << d_parallelTriangulationMovedVxc.n_global_active_cells()
  //                << std::endl;

  d_parallelTriangulationMovedVxc.copy_triangulation(
      d_parallelTriangulationUnmovedVxc);

  const std::vector<bool> locally_owned_vertices =
      dealii::GridTools::get_locally_owned_vertices(
          d_parallelTriangulationMovedVxc);

  std::vector<bool> vertex_moved(d_parallelTriangulationMovedVxc.n_vertices(),
                                 false);
  std::vector<bool> gridPointTouched(
      d_parallelTriangulationMovedVxc.n_vertices(), false);
  // using a linear mapping
  dealii::MappingQGeneric<3, 3> mapping(1);

  std::vector<std::vector<unsigned int>> cellToVertexIndexMap;
  cellToVertexIndexMap.resize(
      parallelMeshUnmoved.n_locally_owned_active_cells());

  std::vector<std::vector<dealii::Point<3>>> cellToVertexParamCoordMap;
  cellToVertexParamCoordMap.resize(
      parallelMeshUnmoved.n_locally_owned_active_cells());

  std::vector<dealii::Point<3>> newVertexPosition;
  newVertexPosition.resize(d_parallelTriangulationMovedVxc.n_vertices());
  for (auto &cellIterVxc :
       d_parallelTriangulationMovedVxc.active_cell_iterators()) {
    if (cellIterVxc->is_locally_owned()) {
      for (unsigned int vertex_no = 0;
           vertex_no < GeometryInfo<3>::vertices_per_cell; ++vertex_no) {
        const unsigned global_vertex_no = cellIterVxc->vertex_index(vertex_no);

        if (gridPointTouched[global_vertex_no] ||
            !locally_owned_vertices[global_vertex_no])
          continue;

        dealii::Point<3> P_real = cellIterVxc->vertex(vertex_no);
        dealii::Point<3> P_ref;
        // can be made optimal by not going through all local cell ??
        unsigned int iElem = 0;
        for (auto &cellIter : parallelMeshUnmoved.active_cell_iterators()) {
          if (cellIter->is_locally_owned()) {
            try {
              P_ref = mapping.transform_real_to_unit_cell(cellIter, P_real);
              bool x_coord = false, y_coord = false, z_coord = false;
              if ((P_ref[0] > -1e-7) && (P_ref[0] < 1 + 1e-7)) {
                x_coord = true;
              }
              if ((P_ref[1] > -1e-7) && (P_ref[1] < 1 + 1e-7)) {
                y_coord = true;
              }
              if ((P_ref[2] > -1e-7) && (P_ref[2] < 1 + 1e-7)) {
                z_coord = true;
              }
              if (x_coord && y_coord && z_coord) {
                // store necessary data here
                cellToVertexIndexMap[iElem].push_back(global_vertex_no);
                cellToVertexParamCoordMap[iElem].push_back(P_ref);
                gridPointTouched[global_vertex_no] = true;
              }
            } catch (...) {
            }
            iElem++;
          }
        }
      }
    }
  }

  unsigned int iCell = 0;
  for (auto &cellIter : parallelMeshMoved.active_cell_iterators()) {
    if (cellIter->is_locally_owned()) {
      for (unsigned int vertexIter = 0;
           vertexIter < cellToVertexIndexMap[iCell].size(); vertexIter++) {
        dealii::Point<3> P_ref = cellToVertexParamCoordMap[iCell][vertexIter];
        dealii::Point<3> P_real =
            mapping.transform_unit_to_real_cell(cellIter, P_ref);
        newVertexPosition[cellToVertexIndexMap[iCell][vertexIter]] = P_real;
      }
      iCell++;
    }
  }

  for (auto &cellIterVxc :
       d_parallelTriangulationMovedVxc.active_cell_iterators()) {
    if (cellIterVxc->is_locally_owned()) {
      for (unsigned int vertex_no = 0;
           vertex_no < GeometryInfo<3>::vertices_per_cell; ++vertex_no) {
        const unsigned global_vertex_no = cellIterVxc->vertex_index(vertex_no);
        if (locally_owned_vertices[global_vertex_no] &&
            !vertex_moved[global_vertex_no]) {
          cellIterVxc->vertex(vertex_no) = newVertexPosition[global_vertex_no];
          vertex_moved[global_vertex_no] = true;
        }
      }
    }
  }

  d_parallelTriangulationMovedVxc.communicate_locally_moved_vertices(
      locally_owned_vertices);
}

void TriangulationManagerVxc::computeMapBetweenParentAndChildMesh(
    const parallel::distributed::Triangulation<3> &parallelParentMesh,
    const parallel::distributed::Triangulation<3> &parallelChildMesh,
    std::vector<std::vector<unsigned int>> &mapParentCellToChildCells,
    std::vector<std::map<unsigned int,
                         typename dealii::DoFHandler<3>::active_cell_iterator>>
        &mapParentCellToChildCellsIter,
    std::vector<unsigned int> &mapChildCellsToParentCell,
    unsigned int &maxChildCellRefinementWithParent) {
  maxChildCellRefinementWithParent = 0;
  mapChildCellsToParentCell.resize(
      parallelChildMesh.n_locally_owned_active_cells());
  mapParentCellToChildCells.resize(
      parallelParentMesh.n_locally_owned_active_cells());
  mapParentCellToChildCellsIter.resize(
      parallelParentMesh.n_locally_owned_active_cells());

  std::vector<unsigned int> numChildCellsInParentCell(
      parallelParentMesh.n_locally_owned_active_cells());
  std::fill(numChildCellsInParentCell.begin(), numChildCellsInParentCell.end(),
            0);
  // using a linear mapping
  dealii::MappingQGeneric<3, 3> mapping(1);

  unsigned int childCellIndex = 0;
  for (auto &cellChildIter : parallelChildMesh.active_cell_iterators()) {
    if (cellChildIter->is_locally_owned()) {
      unsigned int parentCellIndex = 0;
      dealii::Point<3> childCellCenter = cellChildIter->center();
      for (auto &parentChildIter : parallelParentMesh.active_cell_iterators()) {
        if (parentChildIter->is_locally_owned()) {
          try {
            dealii::Point<3> P_ref = mapping.transform_real_to_unit_cell(
                parentChildIter, childCellCenter);
            bool x_coord = false, y_coord = false, z_coord = false;
            if ((P_ref[0] > -1e-10) && (P_ref[0] < 1 + 1e-10)) {
              x_coord = true;
            }
            if ((P_ref[1] > -1e-10) && (P_ref[1] < 1 + 1e-10)) {
              y_coord = true;
            }
            if ((P_ref[2] > -1e-10) && (P_ref[2] < 1 + 1e-10)) {
              z_coord = true;
            }
            if (x_coord && y_coord && z_coord) {
              mapChildCellsToParentCell[childCellIndex] = parentCellIndex;
              mapParentCellToChildCells[parentCellIndex].push_back(
                  childCellIndex);
              //                                mapParentCellToChildCellsIter[parentCellIndex].push_back(cellChildIter);
              mapParentCellToChildCellsIter
                  [parentCellIndex]
                  [numChildCellsInParentCell[parentCellIndex]] = cellChildIter;
              unsigned int relativeRefinement =
                  std::abs(cellChildIter->level() - parentChildIter->level());
              if (relativeRefinement > maxChildCellRefinementWithParent) {
                maxChildCellRefinementWithParent = relativeRefinement;
              }
              numChildCellsInParentCell[parentCellIndex] =
                  numChildCellsInParentCell[parentCellIndex] + 1;
            }
          } catch (...) {
          }
          parentCellIndex++;
        }
      }
      childCellIndex++;
    }
  }
}

//
// get moved parallel mesh
//
parallel::distributed::Triangulation<3> &
TriangulationManagerVxc::getParallelMovedMeshVxc() {
  return d_parallelTriangulationMovedVxc;
}

//
// get unmoved parallel mesh
//
parallel::distributed::Triangulation<3> &
TriangulationManagerVxc::getParallelUnmovedMeshVxc() {
  return d_parallelTriangulationUnmovedVxc;
}
} // namespace invDFT

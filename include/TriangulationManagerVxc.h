// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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

#ifndef triangulationManagerInverseDFT_H_
#define triangulationManagerInverseDFT_H_
#include "headers.h"
#include "dftParameters.h"
#include "triangulationManager.h"
#include "inverseDFTParameters.h"


namespace invDFT
{
  using namespace dealii;

  /**
   * @brief This class generates and stores adaptive finite element meshes for the real-space dft problem.
   *
   *  The class uses an adpative mesh generation strategy to generate finite
   * element mesh for given domain based on five input parameters: BASE MESH
   * SIZE, ATOM BALL RADIUS, MESH SIZE ATOM BALL, MESH SIZE NEAR ATOM and MAX
   * REFINEMENT STEPS (Refer to utils/dftParameters.cc for their corresponding
   * internal variable names). Additionaly, this class also applies periodicity
   * to mesh. The class stores two types of meshes: moved and unmoved. They are
   * essentially the same meshes, except that we move the nodes of the moved
   * mesh (in the meshMovement class) such that the atoms lie on the nodes.
   * However, once the mesh is moved, dealii has issues using that mesh for
   * further refinement, which is why we also carry an unmoved triangulation.
   *
   *  @author Bikash Kanungo, Vishal Subramanian
   */
  class TriangulationManagerVxc
  {
  public:
    /** @brief Constructor.
     *
     * @param mpi_comm_parent parent mpi communicator
     * @param mpi_comm_domain domain decomposition mpi communicator
     * @param interpool_comm mpi interpool communicator over k points
     * @param interBandGroupComm mpi interpool communicator over band groups
     */
    TriangulationManagerVxc(const MPI_Comm &     mpi_comm_parent,
                            const MPI_Comm &     mpi_comm_domain,
                            const MPI_Comm &     interpoolcomm,
                            const MPI_Comm &     interBandGroupComm,
                            const dftfe::dftParameters &dftParams,
                            const inverseDFTParameters & inverseDFTParams,
                            const dealii::parallel::distributed::Triangulation<3, 3>::Settings repartitionFlag =
                              dealii::parallel::distributed::Triangulation<3, 3>::no_automatic_repartitioning);

    /**
     * destructor
     */
    ~TriangulationManagerVxc();

    void
    generateParallelUnmovedMeshVxc(
      const std::vector<std::vector<double>> &       atomPositions,
      dftfe::triangulationManager
        &dftTria); // TODO const triangulationManager will have been better but
                   // I dont think it is possible

    void
    generateParallelMovedMeshVxc(
      const parallel::distributed::Triangulation<3> &parallelMeshUnmoved,
      const parallel::distributed::Triangulation<3> &parallelMeshMoved);

    // this function should be called with the same arguments as
    // d_parallelTriangulationUnmoved otherwise you will get inconsistent
    // partitioning and break the code.
    bool refinementAlgorithmAWithManualRepartition(
      parallel::distributed::Triangulation<3> &parallelTriangulation,
      std::vector<unsigned int> &              locallyOwnedCellsRefineFlags,
      std::map<dealii::CellId, unsigned int> & cellIdToCellRefineFlagMapLocal,
      const bool   smoothenCellsOnPeriodicBoundary = false,
      const double smootheningFactor               = 2.0);

    bool consistentPeriodicBoundaryRefinementForVxc(
      parallel::distributed::Triangulation<3> &parallelTriangulation,
      std::vector<unsigned int> &              locallyOwnedCellsRefineFlags,
      std::map<dealii::CellId, unsigned int> & cellIdToCellRefineFlagMapLocal);

    void generateMeshWithManualRepartitioning(
      parallel::distributed::Triangulation<3> &parallelMesh);

    void
    computeMapBetweenParentAndChildMesh(
      const parallel::distributed::Triangulation<3> &parallelParentMesh,
      const parallel::distributed::Triangulation<3> &parallelChildMesh,
      std::vector<std::vector<unsigned int>> &       mapParentCellToChildCells,
      std::vector<
        std::map<unsigned int,
                 typename dealii::DoFHandler<3>::active_cell_iterator>>
                                &                        mapParentCellToChildCellsIter,
      std::vector<unsigned int> &mapChildCellsToParentCell,
      unsigned int &             maxChildCellRefinementWithParent);


    /**
     * @brief returns reference to parallel moved triangulation
     *
     */
    parallel::distributed::Triangulation<3> &
    getParallelMovedMeshVxc();

    /**
     * @brief returns constant reference to parallel unmoved triangulation
     *
     */
    parallel::distributed::Triangulation<3> &
    getParallelUnmovedMeshVxc();


    /**
     * @brief internal function which generates a coarse  vxc mesh which is required for the load function call in
     * restarts.
     *
     */
    void generateCoarseMeshForVxc(
      parallel::distributed::Triangulation<3> &parallelTriangulation);

  private:
    double                                  d_xcut;
    parallel::distributed::Triangulation<3> d_parallelTriangulationUnmovedVxc,
      d_parallelTriangulationMovedVxc;


    // These are dummy variables. They are used only for consistency purposes.
    // If you use these variables, you will get sef fault.
    parallel::distributed::Triangulation<3> d_serialTriangulationVxc,
      d_serialTriangulationElectrostaticsVxc,
      d_electrostaticsTriangulationRhoVxc, d_electrostaticsTriangulationDispVxc,
      d_electrostaticsTriangulationForceVxc;

    std::vector<std::vector<bool>> d_parallelTriaVxcCurrentRefinement;

    const dftfe::dftParameters &d_dftParams;
    const inverseDFTParameters & d_inverseDFTParams;

    dealii::ConditionalOStream pcout;

    std::vector<std::vector<bool>> d_parallelTriaCurrentRefinement;
    std::vector<std::vector<bool>> d_serialTriaCurrentRefinement;


    //
    // compute-time logger
    //
    TimerOutput     computing_timer;
    const MPI_Comm &d_mpi_comm_parent;
    const MPI_Comm &d_mpi_comm_domain;
    const MPI_Comm &d_interpoolcomm;
    const MPI_Comm &d_interbandgroup_comm;
  };
} // namespace invDFT
#endif

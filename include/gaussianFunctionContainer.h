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
// @author Bikash Kanungo
//


#ifndef DFTFE_GAUSSIANFUNCTIONCONTAINER_H
#define DFTFE_GAUSSIANFUNCTIONCONTAINER_H


#include "headers.h"
#include <dftUtils.h>
#include <gaussianFunctionManager.h>

#include <map>
#include <memory>
//
//
//

namespace invDFT
{
  class gaussianFunctionContainer
  {
  public:
    enum class gaussianDensityAttribute
    {
      PRIMARY,
      SECONDARY
    };

    gaussianFunctionContainer();
    ~gaussianFunctionContainer();

    gaussianFunctionManager &
    getGaussianFunctionManager(const gaussianDensityAttribute attribute);

    void
    addGaussianFunctionManager(
      const gaussianDensityAttribute           attribute,
      std::shared_ptr<gaussianFunctionManager> gfManager);

  private:
    std::map<gaussianDensityAttribute, std::shared_ptr<gaussianFunctionManager>>
      d_attributeToGaussianFunctionManager;

  }; // end of class gaussianFunctionContainer
} // end of namespace invDFT
#endif // DFTFE_GAUSSIANFUNCTIONCONTAINER_H

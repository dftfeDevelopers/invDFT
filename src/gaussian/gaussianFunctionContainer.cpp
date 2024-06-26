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
#include "gaussianFunctionContainer.h"

namespace invDFT
{
  gaussianFunctionManager &
  gaussianFunctionContainer::getGaussianFunctionManager(
    const gaussianFunctionContainer::gaussianDensityAttribute attribute)
  {
    auto it = d_attributeToGaussianFunctionManager.find(attribute);
    AssertThrow(it != d_attributeToGaussianFunctionManager.end(),
                dealii::ExcMessage(
                  "The gaussianFunctionManager is not built for "
                  " the given attribute"));
    return *(it->second);
  }

  void
  gaussianFunctionContainer::addGaussianFunctionManager(
    const gaussianFunctionContainer::gaussianDensityAttribute attribute,
    std::shared_ptr<gaussianFunctionManager>                  gfManager)

  {
    d_attributeToGaussianFunctionManager.insert(
      std::make_pair(attribute, gfManager));
  }

  gaussianFunctionContainer::~gaussianFunctionContainer()
  {}
} // end of namespace invDFT

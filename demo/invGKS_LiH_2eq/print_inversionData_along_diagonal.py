import vtk
import numpy as np

def printRhoData(inputFilename, outputFilename, n_points):
    data_array_name_1 = "rho Gaussian primary"
    data_array_name_2 = "rho Gaussian secondary"
    data_array_name_3 = "rho fe lda"
    # Determine diagonal points (change if needed)
    start = np.array([0.0, 0.0, -40.0])
    end   = np.array([0.0, 0.0, 40.0])

    # 1. Read the pvtu file
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(inputFilename)
    reader.Update()
    ugrid = reader.GetOutput()

    # Optional: Auto-detect bounding box for diagonal
    bounds = ugrid.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    start = np.array([0.0, 0.0, bounds[4]])
    end   = np.array([0.0, 0.0, bounds[5]])

    # 2. Generate sampling points along the diagonal
    points = np.linspace(start, end, n_points)


    vtk_points = vtk.vtkPoints()
    for p in points:
        vtk_points.InsertNextPoint(p)
    probe_polydata = vtk.vtkPolyData()
    probe_polydata.SetPoints(vtk_points)

    # 3. Probe the field values at these points
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(probe_polydata)
    probe.SetSourceData(ugrid)
    probe.Update()
    out = probe.GetOutput()

    array_1 = out.GetPointData().GetArray(data_array_name_1)
    if not array_1:
        raise ValueError(f"Cannot find array '{data_array_name_1}' in pvtu file.")
    array_2 = out.GetPointData().GetArray(data_array_name_2)
    if not array_2:
        raise ValueError(f"Cannot find array '{data_array_name_2}' in pvtu file.")

    array_3 = out.GetPointData().GetArray(data_array_name_3)
    if not array_3:
        raise ValueError(f"Cannot find array '{data_array_name_3}' in pvtu file.")
    
    # 4. Write coordinates and scalars to CSV
    with open(outputFilename, "w") as f:
        f.write(f"x,y,z,rho_target\n")
        for i in range(out.GetNumberOfPoints()):
            xyz = out.GetPoint(i)
            value1 = array_1.GetTuple1(i)
            value2 = array_2.GetTuple1(i)
            value3 = array_3.GetTuple1(i)
            valueFinal = value1 - value2 + value3
            f.write(f"{xyz[0]},{xyz[1]},{xyz[2]},{valueFinal}\n")
    
    print(f"rho data along diagonal written to {outputFilename}.")
     
def printVxcData(inputFilename, outputFilename, n_points, data_array_name_1):
    # Determine diagonal points (change if needed)
    start = np.array([0.0, 0.0, 0.0])
    end   = np.array([1.0, 1.0, 1.0])

    # 1. Read the pvtu file
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(inputFilename)
    reader.Update()
    ugrid = reader.GetOutput()

    # Optional: Auto-detect bounding box for diagonal
    bounds = ugrid.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    start = np.array([bounds[0], bounds[2], bounds[4]])
    end   = np.array([bounds[1], bounds[3], bounds[5]])

    # 2. Generate sampling points along the diagonal
    points = np.linspace(start, end, n_points)


    vtk_points = vtk.vtkPoints()
    for p in points:
        vtk_points.InsertNextPoint(p)
    probe_polydata = vtk.vtkPolyData()
    probe_polydata.SetPoints(vtk_points)

    # 3. Probe the field values at these points
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(probe_polydata)
    probe.SetSourceData(ugrid)
    probe.Update()
    out = probe.GetOutput()

    array_1 = out.GetPointData().GetArray(data_array_name_1)
    if not array_1:
        raise ValueError(f"Cannot find array '{data_array_name_1}' in pvtu file.")

    # 4. Write coordinates and scalars to CSV
    with open(outputFilename, "w") as f:
        f.write(f"x,y,z,{data_array_name_1}\n")
        for i in range(out.GetNumberOfPoints()):
            xyz = out.GetPoint(i)
            value1 = array_1.GetTuple1(i)
            f.write(f"{xyz[0]},{xyz[1]},{xyz[2]},{value1}\n")
    
    print(f"inital Vxc data along diagonal written to {outputFilename}.")


n_points = 400
inputRhoFilename = "inputRhoData_00.pvtu" 
outputRhoFilename = "inputRho_bond_axis.csv"

inputInitialVxcFilename = "initiaVxcGuess_00.pvtu"
outputInitialVxcFilename = "initalVxc_diagonal_data.csv"
data_array_name_initialVxc = "initial vxc"

inputFinalVxcFilename = "vxcIteration_1000000000.000000_120_00.pvtu"
outputFinalVxcFilename = "vxcFinal_120.csv"
data_array_name_final_Vxc = "vxc"
printRhoData(inputRhoFilename, outputRhoFilename, n_points)
#printVxcData(inputInitialVxcFilename, outputInitialVxcFilename, n_points, data_array_name_initialVxc)
#printVxcData(inputFinalVxcFilename, outputFinalVxcFilename, n_points, data_array_name_final_Vxc)

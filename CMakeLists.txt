cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(UNetBrainTumorSeg)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/UNetBrainTumorSeg")
set(EXTENSION_CONTRIBUTORS "Rina Ahmetaj (None)")
set(EXTENSION_DESCRIPTION "This module uses a 3D UNet architecture to segment brain tumor area on MRI scans.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/UNetBrainTumorSeg.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/UNetBrainTumorSeg/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(seg_module)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})

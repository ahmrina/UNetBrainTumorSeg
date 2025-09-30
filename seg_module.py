import logging
import os
from typing import Annotated, Optional

import vtk

import slicer
from PyQt5.QtWidgets import QFileDialog
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
import os
import tempfile
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from monai.transforms import  Compose, LoadImage, EnsureChannelFirst, Spacing, Orientation, ScaleIntensityRange, Resize, ToTensor, AsDiscrete, CenterSpatialCrop, Activations
from monai.inferers import SlidingWindowInferer
from monai.data import MetaTensor
from pathlib import Path



#
# seg_module
#


class seg_module(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("seg_module")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = ["UNets"]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Rina Ahmetaj"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""This module serves as a tool to segment   on MRI scans and predicts on
            Necrotic and Non-Enhancing Tumor Core (NCR/NET), Peritumoral Edema (ED) and GD-enhancing tumor (ET).""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _(""" This Module contains UNet model implemented using PyTorch and has been integrated using MONAI framework.
        
        
                                            For the implementation of the model I've referred to this repository: [Click Here] (https://github.com/bnsreenu/python_for_microscopists/tree/master/231_234_BraTa2020_Unet_segmentation)
                                            
                                            And for deploying to 3D Slicer I've referred to this repository: [Click here] (https://github.com/SenonETS/3DSlicerTutorial_ExtensionModuleDevelopment/tree/master/02__Interact_Debug_%26_Pipeline)""")

        # Additional initialization step after application startup is complete
    #   slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

#
# seg_moduleParameterNode
#


@parameterNodeWrapper
class seg_moduleParameterNode:
    """
    The parameters needed by module.
    FLAIRSelector - FLAIR MRI volume
    T1CESelector = T1CE MRI volume
    T2WeightedSelector = T2 Weighted MRI volume
    OutputSelector : Segmented Brain Tumor from MRI scan
    """

    FLAIRSelector: vtkMRMLScalarVolumeNode
    T1CESelector: vtkMRMLScalarVolumeNode
    T2WeightedSelector: vtkMRMLScalarVolumeNode
    OutputSelector: vtkMRMLScalarVolumeNode


#
# seg_moduleWidget
#


class seg_moduleWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = seg_moduleLogic()
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self._updatingGUIFromParameterNode = False
        self.saving_button_clicked = False

    # DEVELOPER AREA
    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.

        module_dir = os.path.normpath(os.path.dirname(__file__))
        ui_file_path = os.path.join(module_dir, "seg_module", "Resources", "UI", "seg_module.ui")
        uiWidget = slicer.util.loadUI(ui_file_path)
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # print(f"FLAIRSelector init: {hasattr(self.ui, 'FLAIRSelector')}")
        # print(f"T1CESelector init: {hasattr(self.ui, 'T1CESelector')}")
        # print(f"T2WeightedSelector init: {hasattr(self.ui, 'T2WeightedSelector')}")
        # print(f"OutputSelector init: {hasattr(self.ui, 'OutputSelector')}")

        self.ui.FLAIRSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.T2WeightedSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.T1CESelector.setMRMLScene(slicer.mrmlScene)
        self.ui.OutputSelector.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = seg_moduleLogic()

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # ---------------------------------------------------------------------------------------
        self.ui.FLAIRSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.T2WeightedSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.T1CESelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.OutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.runSegmentationButton.connect("clicked(bool)", self.updateParameterNodeFromGUI)
        self.ui.saveSegmentationButton.connect("clicked(bool)", self.saveButton)  # # save as .npy file button

        # Buttons
        self.ui.runSegmentationButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        if self.parent.isEntered:
            self.initializeParameterNode()

    def saveButton(self):
        """Called when save button is clicked."""
        self.saving_button_clicked = True
        self.updateParameterNodeFromGUI()
        self.onApplyButton()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    # DEVELOPER AREA
    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        print("initializeParameterNode called")
        self.setParameterNode(self.logic.getParameterNode())
        # print(f"Parameter node created: {self._parameterNode}, it's not None =")

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.FLAIRSelector:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.FLAIRSelector = firstVolumeNode

        if not self._parameterNode.T2WeightedSelector:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.T2WeightedSelector = firstVolumeNode

        if not self._parameterNode.T1CESelector:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.T1CESelector = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[seg_moduleParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        #  if inputParameterNode:
        # #  if not inputParameterNode.IsSingleton():
        #        raise ValueError(f'SL__Allert! \tinputParameterNode = \n{inputParameterNode.__str__()}')

        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode

        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update; need to do this GUI update whenever there is a change from the Singleton ParameterNode
        self.updateGUIFromParameterNode()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if (self._parameterNode and
                self._parameterNode.FLAIRSelector and
                self._parameterNode.T1CESelector and
                self._parameterNode.T2WeightedSelector and
                self._parameterNode.OutputSelector):
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    # DEVELOPER AREA
    def updateGUIFromParameterNode(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        self._updatingGUIFromParameterNode = True

        # self.ui.FLAIRSelector.setCurrentNode(self._parameterNode.GetNodeReference("FLAIRSelector"))
        # self.ui.T2WeightedSelector.setCurrentNode(self._parameterNode.GetNodeReference("T2WeightedSelector"))
        # self.ui.T1CESelector.setCurrentNode(self._parameterNode.GetNodeReference("T1CESelector"))
        # self.ui.OutputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputSelector"))

        self.ui.FLAIRSelector.setCurrentNode(self._parameterNode.FLAIRSelector)
        self.ui.T1CESelector.setCurrentNode(self._parameterNode.T1CESelector)
        self.ui.T2WeightedSelector.setCurrentNode(self._parameterNode.T2WeightedSelector)
        self.ui.OutputSelector.setCurrentNode(self._parameterNode.OutputSelector)

        # if (self._parameterNode.GetNodeReference("T1CESelector") and
        #     self._parameterNode.GetNodeReference("T2WeightedSelector") and
        #     self._parameterNode.GetNodeReference("FLAIRSelector") and
        #     self._parameterNode.GetNodeReference("OutputSelector")):

        if (self._parameterNode.T1CESelector and
                self._parameterNode.T2WeightedSelector and
                self._parameterNode.FLAIRSelector and
                self._parameterNode.OutputSelector):

            print("Input and Output selected, enabling buttons")

            # now buttons
            self.ui.runSegmentationButton.toolTip = _("runSegmentationButton")
            self.ui.runSegmentationButton.enabled = True

            self.ui.saveSegmentationButton.toolTip = _("saveSegmentationButton")
            self.ui.saveSegmentationButton.enabled = True

        else:
            print("Buttons could not be enabled")
            self.ui.runSegmentationButton.toolTip = _("Select input and output nodes")
            self.ui.runSegmentationButton.enabled = False

            self.ui.saveSegmentationButton.toolTip = _("Run segmentation first")
            self.ui.saveSegmentationButton.enabled = False

        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        print(f"updateParameterNodeFromGUI called")

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()

        self._parameterNode.FLAIRSelector = self.ui.FLAIRSelector.currentNode()
        self._parameterNode.T2WeightedSelector = self.ui.T2WeightedSelector.currentNode()
        self._parameterNode.T1CESelector = self.ui.T1CESelector.currentNode()
        self._parameterNode.OutputSelector = self.ui.OutputSelector.currentNode()

        self._parameterNode.EndModify(wasModified)

    def display(self):

        slicer.app.applicationLogic().GetSelectionNode().SetReferenceActiveVolumeID(self.OutputSelector.GetID())
        slicer.app.applicationLogic().PropagateVolumeSelection(0)








    # DEVELOPER AREA
    def onApplyButton(self) -> None:
        """Run processing when user clicks "Run Segmentation" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.process(self.ui.FLAIRSelector.currentNode(), self.ui.T1CESelector.currentNode(),
                               self.ui.T2WeightedSelector.currentNode(), self.ui.OutputSelector.currentNode())

            if self.saving_button_clicked and self._parameterNode.OutputSelector:
                segmentation_arr = slicer.util.arrayFromVolume(self._parameterNode.OutputSelector)
                self.logic.save_npyFile(segmentation_arr)




#
# model_inference
#

# model architecture (optional) but since model is saved it's not used
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, num_groups=8, dropout=0.2):
        super().__init__()
        self.double_conv = nn.Sequential(

            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(dropout),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(dropout)

        )

    def forward(self, x):
        return self.double_conv(x)


class UpSample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # self.up = nn.ConvTranspose3d(in_channels //2 , in_channels // 2, kernel_size = 2, stride = 2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, dropout=0.2)
        # 512, 256

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if (x1.size() != x2.size()):  # HERE I NEED TO PAD X1 TO MATCH X2

            depth = x2.size(2) - x1.size(2)
            height = x2.size(3) - x1.size(3)
            width = x2.size(4) - x1.size(4)

            x1 = F.pad(x1, (width // 2,
                            width - (width // 2),

                            height // 2,
                            height - (height // 2),

                            depth // 2,
                            depth - (depth // 2)))

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DownSample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, dropout=0.15))

    def forward(self, x):
        return self.down(x)


class UNet3D(nn.Module):

    def __init__(self, in_channels, out_channels, init_channels=24, num_groups=8):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.num_groups = num_groups

        self.doubleconv1 = DoubleConv(in_channels, init_channels)

        # encoder
        self.encoder1 = DownSample(init_channels, init_channels * 2)  # -> 32, 64
        self.encoder2 = DownSample(init_channels * 2, init_channels * 4)  # -> 64, 128
        self.encoder3 = DownSample(init_channels * 4, init_channels * 8)  # -> 128, 256
        self.encoder4 = DownSample(init_channels * 8, init_channels * 8)  # -> 256, 256

        # decoder
        self.decoder1 = UpSample(init_channels * 16, init_channels * 4)  # 512, 128
        self.decoder2 = UpSample(init_channels * 8, init_channels * 2)  # 256, 64
        self.decoder3 = UpSample(init_channels * 4, init_channels)  # 128, 32
        self.decoder4 = UpSample(init_channels * 2, init_channels)  # 64, 32

        self.output = nn.Conv3d(init_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.doubleconv1(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)
        # print("testing decoder 1")
        d1 = self.decoder1(x5, x4)
        # print("testing decoder 2")
        d2 = self.decoder2(d1, x3)
        # print("testing decoder 3")
        d3 = self.decoder3(d2, x2)
        # print("testing decoder 4")
        d4 = self.decoder4(d3, x1)

        out = self.output(d4)

        return out


"""
 preprocessing input image
- stack t1ce, t2, flair (t1 tumor is not very visible for the model so it's not used during training and won't be stacked to the image)
- normalize image
- crop into (128x128x128)
"""


class Inference:

    def __init__(self, model_path, in_channels=3):
        """Takes the model architecture from UNet3d class and the model_weights from the """
        self.device = 'cpu'
        self.model = UNet3D(in_channels, out_channels=4)
        """Model weights will be loaded still. The weights_only is set to True by default on Python 2.6"""
        self.model_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(self.model_dict)

        self.preprocess = Compose([
            EnsureChannelFirst(),
            CenterSpatialCrop(roi_size=(128, 128, 128)),
            # Spacing(pixdim=(1.0, 1.0, 1.0)),
            ScaleIntensityRange(a_min=-4000, a_max=4000, b_min=0.0, b_max=1.0),  # normalize
            ToTensor()
        ])

        self.inference = SlidingWindowInferer(
            roi_size=(128, 128, 128),  # batch, channels, 128x128x128
            sw_batch_size=1,
            overlap=0.25)

        self.postprocess = Compose([
            Activations(softmax=True, dim=1),
            AsDiscrete(argmax=True, dim=1)
        ])

    def create_meta_tensor(self, img):
        temp_file = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        np.save(temp_file.name, img)
        temp_file.close()

        loader = LoadImage()
        meta_tensor = loader(temp_file.name)

        os.unlink(temp_file.name)
        return meta_tensor

    def predict(self, input_volumes):
        """ the input volumes have to be either numpy or nibabel images """
        if hasattr(input_volumes, 'get_fdata'):
            img = input_volumes.get_fdata()
        else:
            img = input_volumes

        meta_tensor = self.create_meta_tensor(img)
        if meta_tensor.shape[-1] == 3:
            meta_tensor = meta_tensor.permute(3, 0, 1, 2)  # channels, 128, 128, 128

        meta_tensor = meta_tensor.unsqueeze(0)
        with torch.no_grad():
         infered_result = self.inference(meta_tensor, network=self.model)

        postprocessed_result = self.postprocess(infered_result)

        return postprocessed_result.cpu().detach().numpy()


#
# seg_moduleLogic
#


class seg_moduleLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        # THIS NEEDS TO BE FILLED (if there's member variables needed to be initialized here) IT COMES EMPTY
        ScriptedLoadableModuleLogic.__init__(self)


    def getParameterNode(self):
        return seg_moduleParameterNode(super().getParameterNode())


    def setDefaultParameters(self, parameterNode):
        """        Initialize parameter node with default settings.       """
        pass


    def save_npyFile(self, segmented_output):
      """Saves the segmented brain tumor as .npy file"""
      # file_path = slicer.util.saveNode(segmented_output, "Save Segmentation", "NumPy Files (*.npy)")
      file_path = QFileDialog.getSaveFileName(
          None,
          "Save Segmentation as NumPy File",  # dialog title
          "",  #  current directory
          "NumPy Files (*.npy)"
      )[0]

      if file_path:
          if not file_path.endswith('.npy'):
              file_path += '.npy'

          np.save(file_path, segmented_output)
          slicer.util.infoDisplay(f"Segmentation saved to:\n{file_path}")




    # !!!! DEVELOPER AREA
    def process(self,
                FLAIRSelector: vtkMRMLScalarVolumeNode,
                T1CESelector: vtkMRMLScalarVolumeNode,
                T2WeightedSelector: vtkMRMLScalarVolumeNode,
                OutputSelector: vtkMRMLScalarVolumeNode = None) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param FLAIRSelector: FLAIR MRI sequence
        :param T1CESelector: T1CE MRI sequence
        :param T2WeightedSelector: T2Weighted MRI sequence
        :param OutputSelector: Output Segmented Brain Tumor from MRI
        """
        print("called seg_moduleLogic.process()")
        if not FLAIRSelector or not T1CESelector or not T2WeightedSelector or not OutputSelector:
            raise ValueError("Either Input or Output is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        cliParams = {
            "FLAIRSelector": FLAIRSelector.GetID(),
            "T1CESelector": T1CESelector.GetID(),
            "T2WeightedSelector": T2WeightedSelector.GetID(),
            "OutputSelector": OutputSelector.GetID()}

        try:

            # these return numpy arrays
            flair_input = slicer.util.arrayFromVolume(FLAIRSelector)
            t1ce_input = slicer.util.arrayFromVolume(T1CESelector)
            t2_input = slicer.util.arrayFromVolume(T2WeightedSelector)

            flair_input = flair_input.transpose(1, 2, 0)
            t1ce_input = t1ce_input.transpose(1, 2, 0)
            t2_input = t2_input.transpose(1, 2, 0)


            print(f"flair volume shape: {flair_input.shape}")
            print(f"t1ce volume shape: {t1ce_input.shape}")
            print(f"t2 volume shape: {t2_input.shape}")

            stacked = np.stack([t1ce_input, t2_input, flair_input], axis=3)
            print(f"stacked MRI volumes: {stacked.shape}")


            module_dir = os.path.normpath(os.path.dirname(__file__))
            model_dir = os.path.join(module_dir, "seg_module", "models", "Best_Model_Weights.pth")

            print("calling inference")
            inference = Inference(model_dir)

            pred = inference.predict(stacked)  # MODEL PREDICTION SHAPE:(1, 1, 240, 240, 155) and type: <class 'numpy.ndarray'>
            print(f"prediction shape: {pred.shape}")

            pred = pred[0]
            segmented_output = pred[0]  # segmented output shape (240, 240, 155)


            # Displaying the output segmentation to Red Slice
            if OutputSelector:
                slicer.util.updateVolumeFromArray(OutputSelector, segmented_output)
                print("Output Selector updated to -> segmented output")

            slicer.app.applicationLogic().GetSelectionNode().SetReferenceActiveVolumeID(OutputSelector.GetID())
            slicer.app.applicationLogic().PropagateVolumeSelection(0)  # red slice is at 0

            display_node = OutputSelector.GetDisplayNode()
            if display_node:

                display_node.SetAndObserveColorNodeID("vtkMRMLColorTableNodeLabels")
                display_node.SetOpacity(0.6)
                display_node.SetScalarVisibility(True)


            stopTime = time.time()
            logging.info(f"Processing completed in {stopTime - startTime:.2f} seconds")


        except Exception as e:
            logging.error(e)


#
# seg_moduleTest
#


class seg_moduleTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_seg_module1()

    def test_seg_module1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        # self.delayDisplay("Starting the test")
        #
        # # Get/create input data
        #
        # import SampleData
        #
        # inputVolume = SampleData.downloadSample("seg_module1")
        # self.delayDisplay("Loaded test data set")
        #
        # inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(inputScalarRange[0], 0)
        # self.assertEqual(inputScalarRange[1], 695)
        #
        # outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        # threshold = 100
        #
        # # Test the module logic
        #
        # logic = seg_moduleLogic()
        #
        # # Test algorithm with non-inverted threshold
        # logic.process(inputVolume, outputVolume, threshold, True)
        # outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        # self.assertEqual(outputScalarRange[1], threshold)
        #
        # # Test algorithm with inverted threshold
        # logic.process(inputVolume, outputVolume, threshold, False)
        # outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        # self.assertEqual(outputScalarRange[1], inputScalarRange[1])
        #
        # self.delayDisplay("Test passed")






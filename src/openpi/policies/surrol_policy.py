import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_surrol_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "state": np.random.rand(14),
        "base_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist1_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist2_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class SurRoLInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0

        # We pad the proprioceptive input to the action dimension of the model.
        # For pi0-FAST, we don't pad the state. For Libero, we don't need to differentiate
        # since the pi0-FAST action_dim = 7, which is < state_dim = 8, so pad is skipped.
        # Keep this for your own dataset, but if your dataset stores the proprioceptive input
        # in a different key than "observation/state", you should change it below.
        state = _padToDimensionWithPositions(data["state"], self.action_dim, slice(0, 7), slice(7, 14))

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = data["base_image"]
        left_wrist_image = data["wrist1_image"]
        right_wrist_image = data["wrist2_image"]

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = _padToDimensionWithPositions(data["actions"], self.action_dim, slice(0, 5), slice(5, 10))
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SurRoLOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        leftWristActions = np.asarray(data["actions"][:, :5])
        rightWristActions = np.asarray(data["actions"][:, 5:])
        return {"actions": np.concatenate((leftWristActions, rightWristActions), axis = -1)}


def _padToDimensionWithPositions(x : np.ndarray, targetDimension : int, 
                                 leftSlice : slice, rightSlice : slice) -> np.ndarray:
    outputShape = list(x.shape)
    currentDimension = outputShape[-1]
    if currentDimension >= targetDimension:
        return x
    outputShape[-1] = targetDimension
    output = np.zeros(outputShape)
    leftTargetSlice = slice(start = 0, stop = currentDimension // 2)
    rightTargetSlice = slice(start = targetDimension // 2, stop = targetDimension // 2 + currentDimension // 2)
    output[..., leftTargetSlice] = x[leftSlice]
    output[..., rightTargetSlice] = x[rightSlice]
    return output

# -*- coding: utf-8 -*-
"""
Created on Sat May  3 20:16:56 2025

@author: zhang
"""
import os
from datetime import datetime
from scipy.io import savemat
import numpy as np
import torch

def save_to_mat_MC(save_dir,
                mode_classification,
                num_modes,
                test_dataset,
                visibility_value,
                temp_model,
                temp_E,
                propagated_fields,
                distance_layers,
                pixel_size,
                distance_propagation,
                propagation_step,
                wavelength,
                training_loss,
                distance_first_layer=0,
                field_size=50,
                focus_radius=5,
                detectsize=15,
                epochs=1000):
    """
    Save optical simulation data and training results to a .mat file.

    Includes:
    - temp_model: phase masks (mask_0, mask_1, ...)
    - temp_test_data: input field
    - propagation_process: list of propagated fields
    - propagation_step: propagation step size 
    - model_parameters: geometry, pixel size, wavelength, etc.
    - training_loss: loss curve during training
    """

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    filename = (
        f"{mode_classification}_M{num_modes}_{len(temp_model)}layers_"
        f"{test_dataset}_{visibility_value:.4f}_{timestamp}.mat"
    )
    filepath = os.path.join(save_dir, filename)

    # Convert phase masks
    model_dict = {}
    for i_layer, mask in enumerate(temp_model):
        if isinstance(mask, np.ndarray):
            model_dict[f'mask_{i_layer}'] = mask.astype(np.float32)
        else:
            print(f"[Warning] temp_model[{i_layer}] is not a numpy array. Skipped.")

    # Convert propagated fields
    prop_dict = {}
    for i_field, field in enumerate(propagated_fields):
        if isinstance(field, torch.Tensor):
            prop_dict[f'field_{i_field}'] = field.detach().cpu().numpy()
        else:
            print(f"[Warning] propagated_fields[{i_field}] is not a torch.Tensor. Skipped.")

    # Convert loss list to numpy
    loss_array = np.array(training_loss, dtype=np.float32)

    # Model/physical parameters
    param_dict = {
        'distance_first_layer': distance_first_layer,
        'distance_layers': distance_layers,
        'distance_propagation': distance_propagation,
        'pixel_size': pixel_size,
        'wavelength': wavelength,
        'field_size': field_size,
        'focus_radius': focus_radius,
        'detectsize': detectsize,
        'epochs': epochs
    }

    # Save everything
    savemat(filepath, {
        'temp_model': model_dict,
        'temp_test_data': temp_E.detach().cpu().numpy(),
        'propagation_process': prop_dict,
        'propagation_step_size': propagation_step,
        'model_parameters': param_dict,
        'training_loss': loss_array
    })

    print(f"✅ Data saved: {filepath}")
    return filepath


def save_to_mat_MD(save_dir,
                   mode_classification,
                   num_modes,
                   test_dataset,
                   all_phase_masks,
                   all_weights_pred_ODNN,
                   all_predictions_np,
                   all_amplitudes_diff,
                   all_phases_diff,
                   all_average_amplitudes_diff,
                   all_average_phases_diff,
                   all_complex_weights_pred,
                   all_image_data_pred,
                   all_cc_real,
                   all_cc_imag,
                   all_cc_recon_amp,
                   all_cc_recon_phase):
    """
    Save ODNN model prediction and evaluation data into a .mat file
    for MATLAB-based visualization or analysis.

    Inputs must be lists of NumPy arrays or PyTorch tensors.
    """

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = (
        f"{mode_classification}_M{num_modes}_{test_dataset}_{timestamp}.mat"
    )
    filepath = os.path.join(save_dir, filename)

    # Utility: convert all list elements to numpy (for MATLAB compatibility)
    def list_to_numpy_safe(lst):
        result = []
        for item in lst:
            if isinstance(item, torch.Tensor):
                result.append(item.detach().cpu().numpy())
            else:
                result.append(item)
        return result

    # Create save dictionary
    save_dict = {
        'all_phase_masks': list_to_numpy_safe(all_phase_masks),
        'all_weights_pred_ODNN': list_to_numpy_safe(all_weights_pred_ODNN),
        'all_predictions_np': list_to_numpy_safe(all_predictions_np),
        'all_amplitudes_diff': list_to_numpy_safe(all_amplitudes_diff),
        'all_phases_diff': list_to_numpy_safe(all_phases_diff),
        'all_average_amplitudes_diff': list_to_numpy_safe(all_average_amplitudes_diff),
        'all_average_phases_diff': list_to_numpy_safe(all_average_phases_diff),
        'all_complex_weights_pred': list_to_numpy_safe(all_complex_weights_pred),
        'all_image_data_pred': list_to_numpy_safe(all_image_data_pred),
        'all_cc_real': list_to_numpy_safe(all_cc_real),
        'all_cc_imag': list_to_numpy_safe(all_cc_imag),
        'all_cc_recon_amp': list_to_numpy_safe(all_cc_recon_amp),
        'all_cc_recon_phase': list_to_numpy_safe(all_cc_recon_phase),
    }

    # Save to .mat
    savemat(filepath, save_dict)

    print(f"✅ ODNN model data saved: {filepath}")
    return filepath

def save_to_mat_MD1(save_dir, *,
                       mode_classification,
                       num_modes,
                       test_dataset,
                       temp_model,          # (L,H,W) 或 list[np.ndarray]
                       temp_E,              # torch.Tensor
                       propagated_fields,   # list[torch.Tensor] 或 list[(S,H,W)]
                       propagation_step,
                       distance_layers,
                       pixel_size,
                       distance_propagation,
                       wavelength,
                       field_size,
                       focus_radius,
                       detectsize,
                       epochs,
                       training_loss=None):
    import numpy as np, torch, os
    from scipy.io import savemat
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    filepath = os.path.join(save_dir, f"{mode_classification}_M{num_modes}_{test_dataset}_VIS_{ts}.mat")

    # 统一为 numpy
    if isinstance(temp_model, list):
        temp_model_stack = np.stack([np.asarray(m, dtype=np.float32) for m in temp_model], axis=0)
    else:
        temp_model_stack = np.asarray(temp_model, dtype=np.float32)

    temp_E_np = torch.abs(temp_E).to(torch.float32).cpu().numpy()
    props = []
    for v in propagated_fields:
        if torch.is_tensor(v):
            arr = torch.abs(v).to(torch.float32).cpu().numpy()
        else:
            arr = np.asarray(v, dtype=np.float32)
        props.append(arr)
    # 存成 struct 的 cell 风格
    prop_dict = {f"step_{i:03d}": p for i, p in enumerate(props)}

    savemat(filepath, {
        "temp_model" : temp_model_stack,
        "temp_E"     : temp_E_np,
        "propagated" : prop_dict,
        "propagation_step"    : float(propagation_step),
        "distance_layers"     : float(distance_layers),
        "pixel_size"          : float(pixel_size),
        "distance_propagation": float(distance_propagation),
        "wavelength"          : float(wavelength),
        "field_size"          : int(field_size),
        "focus_radius"        : int(focus_radius),
        "detectsize"          : int(detectsize),
        "epochs"              : int(epochs),
        "training_loss"       : np.asarray(training_loss) if training_loss is not None else [],
    }, do_compression=True)
    print("Saved (v5 pro):", filepath)
    return filepath

def save_to_mat_MD2(
    filepath,
    MMF_eigenmodes_data,
    training_dataset_type,
    test_dataset,
    test_dataset_label,
    all_losses,
    all_phase_masks,
    all_weights_pred_ODNN,
    all_predictions_np,
    all_amplitudes_diff,
    all_average_amplitudes_diff,
    all_amplitudes_relative_diff,
    all_phases_diff,
    all_average_phases_diff,
    all_complex_weights_pred,
    all_image_data_pred,
    all_cc_real,
    all_cc_imag,
    all_cc_recon_amp,
    all_cc_recon_phase
):
    """
    Save ODNN model prediction and evaluation data into a .mat file.
    Now filepath must be constructed outside the function.
    """
    # import torch
    # from scipy.io import savemat

    def list_to_numpy_safe(lst):
        result = []
        for item in lst:
            if isinstance(item, torch.Tensor):
                result.append(item.detach().cpu().numpy())
            else:
                result.append(item)
        return result

    save_dict = {
        'MMF_eigenmodes_data': MMF_eigenmodes_data.detach().cpu().numpy()
                               if isinstance(MMF_eigenmodes_data, torch.Tensor)
                               else MMF_eigenmodes_data,
        'training_dataset_type': training_dataset_type,
        'test_dataset': test_dataset,
        'test_dataset_label': test_dataset_label,
        'all_losses': list_to_numpy_safe(all_losses),
        'all_phase_masks': list_to_numpy_safe(all_phase_masks),
        'all_weights_pred_ODNN': list_to_numpy_safe(all_weights_pred_ODNN),
        'all_predictions_np': list_to_numpy_safe(all_predictions_np),
        'all_amplitudes_diff': list_to_numpy_safe(all_amplitudes_diff),
        'all_average_amplitudes_diff': list_to_numpy_safe(all_average_amplitudes_diff),
        'all_amplitudes_relative_diff': list_to_numpy_safe(all_amplitudes_relative_diff),
        'all_phases_diff': list_to_numpy_safe(all_phases_diff),
        'all_average_phases_diff': list_to_numpy_safe(all_average_phases_diff),
        'all_complex_weights_pred': list_to_numpy_safe(all_complex_weights_pred),
        'all_image_data_pred': list_to_numpy_safe(all_image_data_pred),
        'all_cc_real': list_to_numpy_safe(all_cc_real),
        'all_cc_imag': list_to_numpy_safe(all_cc_imag),
        'all_cc_recon_amp': list_to_numpy_safe(all_cc_recon_amp),
        'all_cc_recon_phase': list_to_numpy_safe(all_cc_recon_phase),
    }

    savemat(filepath, save_dict)
    print(f"✅ ODNN model data saved to file:\n{filepath}")
    return filepath

def save_to_mat_MD_pro(
    filepath,
    temp_model,
    temp_E,
    propagated_fields,
    propagation_step,
    distance_layers,
    pixel_size,
    distance_propagation,
    wavelength,
    field_size=50,
    focus_radius=5,
    detectsize=15
):
    """
    Save phase masks and propagation fields into a .mat file
    for optical simulation analysis.

    Parameters
    ----------
    filepath : str
        Full path to the output .mat file.
    temp_model : list of np.ndarray
        List of phase masks (one per layer).
    temp_E : torch.Tensor
        Input field (E field).
    propagated_fields : list of torch.Tensor
        List of propagated fields (after each layer).
    propagation_step : float
        Step size between layers.
    distance_layers : list or np.ndarray
        List of distances for each layer.
    pixel_size : float
        Pixel pitch.
    distance_propagation : float
        Final propagation distance.
    wavelength : float
        Wavelength in meters.
    field_size : int
        Field width in pixels.
    focus_radius : int
        Radius of focus region (pixels).
    detectsize : int
        Detection area size (pixels).
    """

    # Convert model (phase masks)
    model_dict = {}
    if isinstance(temp_model, list):
        for i, mask in enumerate(temp_model):
            if isinstance(mask, np.ndarray):
                model_dict[f'mask_{i}'] = mask.astype(np.float32)
            else:
                print(f"[Warning] Phase mask {i} is not a numpy array. Skipped.")
    elif isinstance(temp_model, np.ndarray):
        model_dict['mask_0'] = temp_model.astype(np.float32)
    else:
        raise ValueError("temp_model must be a list or np.ndarray")

    # Convert propagated fields
    prop_dict = {}
    for i, field in enumerate(propagated_fields):
        if isinstance(field, torch.Tensor):
            prop_dict[f'field_{i}'] = field.detach().cpu().numpy()
        else:
            print(f"[Warning] Field {i} is not a torch.Tensor. Skipped.")

    # Model parameters
    param_dict = {
        'distance_layers': distance_layers,
        'distance_propagation': distance_propagation,
        'pixel_size': pixel_size,
        'wavelength': wavelength,
        'field_size': field_size,
        'focus_radius': focus_radius,
        'detectsize': detectsize,
        'propagation_step': propagation_step
    }

    # Input field
    if isinstance(temp_E, torch.Tensor):
        temp_E_np = temp_E.detach().cpu().numpy()
    else:
        raise ValueError("temp_E must be a torch.Tensor")

    # Save
    savemat(filepath, {
        'temp_model': model_dict,
        'temp_test_data': temp_E_np,
        'propagation_process': prop_dict,
        'model_parameters': param_dict
    })

    print(f"✅ Propagation data saved to: {filepath}")
    return filepath
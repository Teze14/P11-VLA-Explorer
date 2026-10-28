# What VLA Explorer demonstrates

The project makes the interface between vision, language, and action inspectable. The runnable baseline uses a pretrained vision-language model (CLIP) and a deliberately simple action rule. It demonstrates the architecture concept, not a pretrained end-to-end robot policy.

## Computation

For grid crop i, let v_i be the normalized CLIP image embedding and t the normalized text embedding. Compute s_i = v_i dot t. Relative weights are w_i = exp((s_i-max(s))/tau) / sum_j exp((s_j-max(s))/tau). Each weight refers to a user-defined input crop, not a learned attention head. The target is sum_i w_i center_i; delta = gain(target-origin).

CLIP is frozen. There is no local training and no invented attention tensor. Quantization reduces weight storage; freezing alone does not make forward inference free. LoRA is not used in this baseline and is not a quantization method.

## Scope changes from the brainstorming sheet

- **Retained:** engineering-student audience, image + text inputs, heatmap, target point, direction arrow, responsive controls, pretrained models.
- **Refined implementation:** browser-based ONNX execution replaces the proposed Gradio/PyTorch server. The supplied guideline explicitly allows ONNX Web runtime. This removes a dedicated inference server and exposes the same conceptual stages. Gradio remains a possible wrapper for the GPU extension.
- **Corrected claim:** the heatmap does not show exactly what a model thinks. It shows the similarity of independently encoded crops and text.
- **Delivered:** uploaded photos, one camera snapshot at a time, learned crop scoring, tensor/math views, comparisons, perturbation, JSON export, guided learning.
- **Not delivered as validated capability:** streaming webcam inference, ROS 2 actuation, calibrated 3D target localization, human learning gains, genuine robot-policy success metrics.

## Why this is a defensible educational contribution

Students can distinguish learned computation from designed postprocessing, explain how normalization changes weights, and test whether a display manipulation actually changes the model. Counterfactual inputs expose limitations. These outcomes should be measured with learner tasks, not claimed from an attractive interface alone.

## Genuine VLA extension

OpenVLA and SmolVLA learn robot action mappings. A meaningful comparison needs a robot dataset with aligned camera images, instruction, proprioceptive state, action definitions, preprocessing/normalization statistics, and a validated checkpoint. Arbitrary image-to-arrow comparisons would not establish equivalence. The optional notebook is an integration exercise, explicitly unvalidated until executed on compatible data. No actions are sent to hardware.

# Evaluation protocol — planned, not completed

## Technical evaluation

Use 12 consenting/self-owned or properly licensed tabletop photos, two target descriptions each, and manually record target boxes before inference. Keep an untouched evaluation split. Record whether the highest-weight crop overlaps the intended target, normalized distance from weighted center to target center, and target-absent failure behavior. These are image grounding measurements, not robotic task success.

Measure cold download separately from cold session initialization, new-image inference, cached-image new-prompt inference, full cache hit, and temperature/render updates. Report browser, hardware, backend, grid, sample count, median and range. Quantized model accuracy must be compared with a full-precision version before claiming negligible loss.

## Learner evaluation

Recruit 6-10 engineering students voluntarily. Do not collect sensitive personal information. Use anonymous participant IDs. Obtain any instructor/institutional approvals applicable to a classroom study before recruitment. No participants or results are claimed in this delivery.

Use a six-question pre/post assessment: (1) source of embeddings; (2) cache behavior; (3) meaning of softmax; (4) opacity vs temperature; (5) learned vs rule-based actions; (6) interpreting a target-absent failure. Track paired score changes descriptively, task completion, time, and short written explanations. With a small convenience sample, do not claim causal learning effectiveness or population-wide generalization.

Three tasks: change a target description and explain the new ranking; diagnose why opacity did not move the target; explain a weighted target located between two objects. Counterbalance task order if feasible. Ask participants to think aloud; record difficulties and revision decisions.

## Collection templates

Technical CSV: image_id,prompt,intended_target_box,grid,temperature,highest_region,weighted_x,weighted_y,overlap,normalized_error,total_ms,image_ms,text_ms,load_ms,cache_image,cache_text,notes

Learner CSV: anonymous_id,pre_score,post_score,task1_complete,task2_complete,task3_complete,total_seconds,difficulty,comment

Retain raw measurements. Clearly separate automated functional checks from human evaluation and from model accuracy evaluation.

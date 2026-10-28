# Roadmap to December 11, 2026

The guideline describes a five-week build window, while its rubric says six-week roadmap. We cover five development weeks (Nov 2-Dec 6) plus the final handoff week (Dec 7-11).

Assignments below are **proposed for team confirmation**, not claims of work already completed.

| Dates | Build and evidence | Proposed lead | Done when |
|---|---|---|---|
| Nov 2-8 | Review baseline, publish repository, reproduce timings on Jesus's laptop and a second machine; build licensed image test set | Jesus (UI/reproducibility), all review | Staff can clone/run; results include device metadata |
| Nov 9-15 | Compare crop grids and target rules; assess CLIP failure cases; implement compatible SmolVLA observation adapter if feasible | Robert (model/evaluation) | Grounding measurements and a documented VLA input/action contract |
| Nov 16-22 | Improve learning sequence and tensor views; pilot three tasks with 2 volunteers; revise confusing terminology | Michelle (learning design), Jesus | Pilot notes, revised tutorial, accessibility checks |
| Nov 23-29 | Conduct 6-10-person classroom usability/learning evaluation; profile inference and cache behavior | Michelle (study), Robert (profiling) | Anonymous raw data, descriptive analysis, documented limitations |
| Nov 30-Dec 6 | Consolidate evidence; optional genuine-VLA episode replay from a validated compatible run; regression and packaging | Robert (model), Jesus (integration) | No invented model traces; critical bugs fixed; reproducible package |
| Dec 7-11 | Freeze release, complete final report, record presentation, rehearse explanations and submit | All three | Final screenshots match release; repository/video links verified |

Stretch gates: continuous webcam only if measured new-frame latency supports it; ROS 2 only after a validated simulator/control interface and instructor agreement. Neither is necessary to demonstrate the core educational contribution.

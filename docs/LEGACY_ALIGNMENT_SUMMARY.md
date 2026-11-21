# Public technical alignment summary

Earlier private alignment reviews identified different state/action definitions
between thesis equations and legacy training code. The legacy `ddqfinaltraining`
environment has cloud/charging and one action per UAV; it is not the corrected
per-vehicle destination plus per-UAV movement model. The larger 3D network adds
convolution, LSTM, attention and replay machinery, so its name does not prove
lightweight deployment. Some old documentation reports expected results and
some charts use constructed numbers; those are not fresh measured outcomes.

The new experiment must use the local corrected simulator, explicitly map its
variables, and establish any controller improvement through measured matched
comparisons. Old checkpoints have not been proven compatible. Historical files
remain reference-only. Private thesis drafts, personal identifiers and internal
supervisor correspondence are deliberately not included in this public export.

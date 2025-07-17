## Improve the fits from what was presented on Monday

- [ ] The addition of the learnable exponents in energy spectrum was a good addition,
    but it's not quite enough


## Speed up the computations

- [ ] DRDMannTurb currently doesn't actually support float32, which is a problem. I believe
    that most of this is due to passing numpy outputs into torch functions without
    specifying the dtype

- [ ] Especially with the coherence fitting branching, things are not very generalizable
    beyond the STORM data we are interested in fitting for NREL. However, more importantly,
    things are passed around using a lot of python types, so it can be much faster.
    I think there is also a fair bit of duplicate code within one_point_spectra.py

## Improve the style, interface, etc

- [ ] There are still a few spots where the flow of data is not clear at all and very
    hard to test, since things are passed around via class attributes. For example,
    OnePointSpectra.exp_scales() does this, as well the eddylifetime evaluation
    self.beta = self.eddylifetime (or whatever the exact name is -- it's similar).

- [ ] We need to improve how the coherence data is passed in


## Debugging

- [ ] Float32 issue

- [ ] The coherence plots show a shifted

- [ ] self.curves issue -- previously, when self.curves in the CalibrationProblem
    was just `[0,1,2,3]` for the original use case, things were fine and we did not
    need any epsilon clipping. Now that I've switched to `[0,1,2,3,4,5]` for the
    "full 6 component fits", there is suddenly a NaN error introduced?

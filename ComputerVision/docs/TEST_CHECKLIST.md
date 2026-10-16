# Integration and Testing Checklist

## Before receiving implementations
- [ ] Confirm each owner's filename and function signatures.
- [ ] Confirm which libraries are permitted by the assignment.
- [ ] Agree where shared images and experiment outputs will be stored.
- [ ] Prepare a table for test ID, input, expected result, actual result, status, and evidence.

## CLAHE
- [ ] Test `clip_histogram`
- [ ] Test `build_tile_luts`
- [ ] Test `clahe`
- [ ] Check clip-excess redistribution and LUT interpolation as required by the assignment.
- [ ] Check output shape, dtype, and value range.

## Adaptive thresholding
- [ ] Test `integral_image`
- [ ] Test `box_sum`
- [ ] Test `local_mean_std`
- [ ] Test `adaptive_threshold`
- [ ] Check image borders, constant/flat images, and variance handling.
- [ ] Check output shape, dtype, and value range.

## Multiscale Hough
- [ ] Test `hough_accumulate`
- [ ] Test `find_peaks`
- [ ] Test `hough_multiscale`
- [ ] Check angle wrap-around near 0 and pi.
- [ ] Check synthetic lines with known parameters.
- [ ] Check output shape, dtype, and value range.

## Integration
- [ ] Confirm the shared `convolve2d` dependency is supplied by the group as required.
- [ ] Run each component's basic test.
- [ ] Run `main.ipynb` from the first cell to the last.
- [ ] Confirm all required experiment plots and comparisons are present.
- [ ] Restart the kernel and run all cells again before submission.

## When a test fails
- [ ] Record the exact input and parameters.
- [ ] Record expected and actual results.
- [ ] Save an error message or screenshot where useful.
- [ ] Send the report to the implementation owner to fix.
- [ ] Re-run the same test after the owner updates the code.

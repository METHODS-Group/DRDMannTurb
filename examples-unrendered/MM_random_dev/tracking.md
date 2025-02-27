## TRACKING:

- [ ] Check if dy is correct; we are integrating in fourier space
- [ ] Implement eq16 again
- [ ] Set up the comparison numerical integration plot
- [ ] Implement 10 realizations and plot average for F11, F22


## Values

### 2/24/2025
For N1 = 2^10, N2 = 2^7,
```
Debug info for: u1
  Shape: (1024, 128)
  Range: -0.000236 to 0.000221
  Mean: 0.000000, Std: 0.000053

Debug info for: u2
  Shape: (1024, 128)
  Range: -0.000232 to 0.000226
  Mean: 0.000000, Std: 0.000053
```

Halved; N1 = 2^9, N2 = 2^6

```
Debug info for: u1
  Shape: (512, 64)
  Range: -0.000858 to 0.000811
  Mean: -0.000000, Std: 0.000195

Debug info for: u2
  Shape: (512, 64)
  Range: -0.000775 to 0.000914
  Mean: -0.000000, Std: 0.000194
```

Halved again; N1 = 2^8, N2 = 2^5

```
Debug info for: u1
  Shape: (256, 32)
  Range: -0.002694 to 0.002360
  Mean: 0.000000, Std: 0.000662

Debug info for: u2
  Shape: (256, 32)
  Range: -0.002437 to 0.003034
  Mean: 0.000000, Std: 0.000669
```

Halved again; N1 = 2^7, N2 = 2^4

```
Debug info for: u1
  Shape: (128, 16)
  Range: -0.005370 to 0.007668
  Mean: -0.000000, Std: 0.001877

Debug info for: u2
  Shape: (128, 16)
  Range: -0.007383 to 0.006146
  Mean: -0.000000, Std: 0.001907
```


### 2/25/2025

Trying to

# Observational techniques in astrophysics: Imaging

## Charge-coupled device

In a CCD image sensor, pixels are represented by p-doped
metal-oxide-semiconductors (MOS) capacitors.  There is a photoactive region
(silicon layer), and a transmission region. When photons hit the capacitor
array, then each capacitor accumulates an electric charge proportional to the
light intensity at that location. Once the array has been exposed to the image,
a control circuit causes each capacitor to transfer its contents to its neighbor
(operating as a shift register).

## Sources of noise

CCD images are not perfect and contain a lot of noise, which needs to be dealt
with before those images can be used to do science. The follwoing are some types
of noise.

### Read noise

This is the noise introduced in each pixel when we read out the device. There
are, e.g., quantum effects, gaps between pixels, and analog-to-digital
converters that add noise.

### Dark current

Noise introduced by surrounding stuff, e.g., the telescope itself can emit some
electrons or whatever.

### Bias
Pixels have individual biases that are present even when the exposure time is zero.

### Gain & dynamic Range
The gain is the number of electrons needed to produce one step in the ADU converter.
CCDs are linear to a degree, then non-linearity kicks in.
Its like an overflow of the electron bucket.

## Basics of CCD Reduction
This reduction is made to go from images to data. Usually the following three
calibrations are made: Bias (take 0 sec exposure), Dark (take exposure with covered lens),
and Flat (something with twilight). There is a CCD equation for the SNR.

## Cosmic Rays
This is high energy junk from space. We don't like it when it hits our CCD.
They accumulate with longer exposure times. Mitigate, e.g., by combining (median)
multiple exposures. There are also statistical methods to remove these.

## Point Spread Function (PSF)
This is the response of an imaging system to a point source (e.g. stars or galaxies).
Some factors in the PSF is the atmosphere, the telescope optics, the detector, the weather, etc..

## Flexible image transport system (FITS)
This file format has a human readable ASCII header that contains keywords and values.
This header should be updated whenever the images are processed. The best program
for FITS files is DS9. 

## Dithering
This is done by small positional shift between exposures. This is done to
see the same object on different positions on the telescope to get some redundancy.
Another reason is to better samplethe PSF, if pixels are large relative to PSF.

This procedure yields multiple images that need to be recombined. We would like to
recombine them by interlacing (just add images with appropriate shift). But this is
impossible in practice because we are not that precise with telescope orientation and
the focal plane is not 100% flat. Another way would be to shift and add (Block
replicate each pixel). This is not so good because it convolves the image with the
original pixel again, leading to a blurring and corelated noise. The way recombination
is done is called Drizzle. This is some sort of combination of both techniques.

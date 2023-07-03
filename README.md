image-match-rs
==============

An implementation of the image matching algorithm described in 
[this paper](http://www.cs.cmu.edu/~hcwong/Pdfs/icip02.ps). The matching algorithm is designed to detect nearly 
identical images, not images with the same conceptual content. 


Usage
=====

By default, the library offers two primary functions: `get_buffer_signature(rgba, width)` and `cosine_similarity(a, b)`.
The former takes a pre-processed slice of `u8`s with each chunk of four representing the 8-bit red, green, blue, and 
alpha of a pixel, the latter two result vectors to compute their similarity. Per the source paper and our experiments
in [this research](https://github.com/alt-text-org/image-algo-testing) images with a similarity greater than `0.6` can
be considered likely matches.

If the `img` feature is used, also provided are `get_image_signature(image)` and `get_file_signature(path)` which use 
the [image library](https://crates.io/crates/image) to handle unpacking the image into an rgba buffer. All signature
functions also expose `tuned` versions which allow tweaking the crop percentage used during the signature computation,
as well as the size of the collection grid which controls the length of the feature vector produced and the size of the 
square around each grid point averaged to produce a value for that point.
 

Future Work
===========

- Unit testing. The library has been manually tested significantly, but it needs unit testing.
- Experiment with widening the possible values of each dimension in the produced signature. Presently per the paper they
  are all integers in `[-2, 2]`. It will likely require experimentation around a new suggested vector similarity cutoff.
use std::cmp::{max, min};
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::io;
use std::path::Path;

use image::{GenericImageView, ImageError, Pixel};
use image::io::Reader as ImageReader;
use num::ToPrimitive;

use ImageReadError::{DecodeError, IoError};

use crate::{compute_from_gray, DEFAULT_CROP, DEFAULT_GRID_SIZE, pixel_gray};

/// Produces a 544 signed byte signature for a provided image. The result is designed to be compared
/// to other vectors computed by a call to this method using [cosine-similarity(a, b)].
pub fn get_image_signature<I: GenericImageView>(img: I) -> Vec<i8> {
    let gray = grayscale_image(img);

    let average_square_width_fn = |width, height| {
        max(
            2_usize,
            (0.5 + min(width, height) as f32 / 20.0).floor() as usize,
        ) / 2
    };

    compute_from_gray(gray, DEFAULT_CROP, DEFAULT_GRID_SIZE, average_square_width_fn)
}

/// Produces a variable length signed byte signature for a provided image. The result is designed to
/// be compared to other vectors computed by a call to this method with identical tuning parameters
/// using [cosine-similarity(a, b)]. `crop` is a value in [0, 0.5] indicating what percentage of the
/// image to crop on all sides before grid placement. Note that this percentage is based not on the
/// raw width but a calculation of color density. `grid_size` indicates how many points to place on
/// the image for measurement in the resulting signature. Changing `grid_size` will alter the length
/// of the signature to `8 * (grid_size - 1)^2 - 12 * (grid_size - 3) - 20`. The
/// `average_square_width_fn` controls the size of the box around each grid point that's averaged
/// to produce that grid point's brightness value. The paper proposes
/// `max(2, floor(0.5 + min(cropped_width, cropped_height) / 20))` but provides no information about
/// how that was chosen.
pub fn get_tuned_image_signature<I: GenericImageView>(
    img: I,
    crop: f32,
    grid_size: usize,
    average_square_width_fn: fn(width: usize, height: usize) -> usize,
) -> Vec<i8> {
    let gray = grayscale_image(img);
    compute_from_gray(gray, crop, grid_size, average_square_width_fn)
}

/// Produces a 544 signed byte signature for a provided image file. The result is designed to be
/// compared to other vectors computed by a call to this method using [cosine-similarity(a, b)].
pub fn get_file_signature<P: AsRef<Path>>(path: P) -> Result<Vec<i8>> {
    let image = ImageReader::open(path)?.decode()?;
    Ok(get_image_signature(image))
}

/// Produces a variable length signed byte signature for a provided image file. The result is
/// designed to be compared to other vectors computed by a call to this method with identical tuning
/// parameters using [cosine-similarity(a, b)]. `crop` is a value in [0, 0.5] indicating what
/// percentage of the image to crop on all sides before grid placement. Note that this percentage is
/// based not on the raw width but a calculation of color density. `grid_size` indicates how many
/// points to place on the image for measurement in the resulting signature. Changing `grid_size`
/// will alter the length of the signature to `8 * (grid_size - 1)^2 - 12 * (grid_size - 3) - 20`.
/// The `average_square_width_fn` controls the size of the box around each grid point that's
/// averaged to produce that grid point's brightness value. The paper proposes
/// `max(2, floor(0.5 + min(cropped_width, cropped_height) / 20))` but provides no information about
/// how that was chosen.
pub fn get_tuned_file_signature<P: AsRef<Path>>(
    path: P,
    crop: f32,
    grid_size: usize,
    average_square_width_fn: fn(width: usize, height: usize) -> usize,
) -> Result<Vec<i8>> {
    let image = ImageReader::open(path)?.decode()?;
    Ok(get_tuned_image_signature(image, crop, grid_size, average_square_width_fn))
}

pub enum ImageReadError {
    IoError(io::Error),
    DecodeError(ImageError),
}

impl Debug for ImageReadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            IoError(e) => Debug::fmt(e, f),
            DecodeError(e) => Debug::fmt(e, f),
        }
    }
}

impl Display for ImageReadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            IoError(e) => Display::fmt(e, f),
            DecodeError(e) => Display::fmt(e, f),
        }
    }
}

impl Error for ImageReadError {
    fn cause(&self) -> Option<&dyn Error> {
        match self {
            IoError(e) => Some(e),
            DecodeError(e) => Some(e)
        }
    }
}

impl From<io::Error> for ImageReadError {
    fn from(e: io::Error) -> Self {
        IoError(e)
    }
}

impl From<ImageError> for ImageReadError {
    fn from(e: ImageError) -> Self {
        DecodeError(e)
    }
}

pub type Result<R> = std::result::Result<R, ImageReadError>;

fn grayscale_image<I: GenericImageView>(img: I) -> Vec<Vec<u8>> {
    let pixels = img.pixels()
        .map(|(_, _, p)| p.to_rgba().0);

    let mut result = Vec::with_capacity(img.width() as usize);
    let mut row = Vec::with_capacity(img.height() as usize);
    let mut col = 0;
    for pixel in pixels {
        row.push(pixel_gray(
            pixel[0].to_u8().unwrap(),
            pixel[1].to_u8().unwrap(),
            pixel[2].to_u8().unwrap(),
            pixel[3].to_u8().unwrap(),
        ));
        col += 1;
        if col >= img.width() {
            result.push(row);
            row = Vec::with_capacity(img.height() as usize);
            col = 0;
        }
    }

    result
}

use std::cmp::{max, min};
use std::collections::HashMap;
use image::GenericImageView;

const SIGNATURE_LEN: usize = 544;

pub fn get_signature(rgba_buffer: &[u8], width: usize) -> Vec<i8> {
    let gray = grayscale(rgba_buffer, width);
    let bounds = crop_boundaries(&gray);
    let points = grid_points(bounds);
    let averages = grid_averages(gray, points);

    compute_signature(averages)
}

fn grayscale(rgba_buffer: &[u8], width: usize) -> Vec<Vec<u8>> {
    let mut result = vec![];
    let mut idx: usize = 0;
    while idx < rgba_buffer.len() {
        let mut row = vec![];
        for _ in 0..width {
            let rgb_avg = (
                rgba_buffer[idx] as u16 +
                    rgba_buffer[idx + 1] as u16 +
                    rgba_buffer[idx + 2] as u16
            ) / 3;

            let with_alpha = ((rgb_avg as f32) * (rgba[idx + 3] as f32 / 255)) as u8;
            row.push(with_alpha);
            idx += 4;
        }
        result.push(row);
    }

    result
}

struct Bounds {
    lower_x: usize,
    upper_x: usize,
    lower_y: usize,
    upper_y: usize,
}
/*
Step 2, part 1
"For each column of the image, we compute the sum of absolute values of differences between
adjacent pixels in that column. We compute the total of all columns, and crop the image at
the 5% and 95% columns, that is, the columns such that 5% of the total sum of differences
lies on either side of the cropped image. We crop the rows of the image the same way
(using the sums of original uncropped rows)."
 */
fn crop_boundaries(pixels: &Vec<Vec<u8>>) -> Bounds {
    let row_diff_sums: Vec<i32> = (0..pixels.len()).map(|y|
        (1..pixels[y].len()).map(|x|
            (pixels[y][x] as i32).abs_diff(pixels[y][x - 1] as i32)).sum()
    ).collect();

    let (top, bottom) = get_bounds(row_diff_sums);

    let col_diff_sums: Vec<i32> = (0..pixels[0].len()).map(|x|
        (1..pixels.len()).map(|y|
            (pixels[y][x] as i32).abs_diff(pixels[y][x - 1] as i32)).sum()
    ).collect();

    let (left, right) = get_bounds(col_diff_sums);

    Bounds {
        lower_x: left,
        upper_x: right,
        lower_y: top,
        upper_y: bottom,
    }
}

fn get_bounds(diff_sums: Vec<i32>) -> (usize, usize) {
    let threshold = diff_sums.iter().sum() / 20;
    let mut lower = 0;
    let mut upper = diff_sums.len() - 1;
    let mut sum = 0;

    while sum < threshold {
        sum += diff_sums[lower];
        lower += 1;
    }
    sum = 0;
    while sum < threshold {
        sum += diff_sums[upper];
        upper -= 1;
    }
    (lower, upper)
}

fn grid_points(bounds: Bounds) -> HashMap<(i8, i8), (usize, usize)> {
    let x_width = (bounds.upper_x - bounds.lower_x) / 10;
    let y_width = (bounds.upper_y - bounds.lower_y) / 10;

    let mut points = HashMap::new();
    for x in 0..10 {
        for y in 0..10 {
            points.insert((x as i8, y as i8), (x * x_width, y * y_width))
        }
    }

    points
}

fn grid_averages(
    pixels: Vec<Vec<u8>>,
    points: HashMap<(i8, i8), (usize, usize)>,
) -> HashMap<(i8, i8), u8> {
    let square_edge = (max(
        2.0,
        (0.5 * min(pixels.len(), pixels[0].len()) as f32 / 20.0).floor(),
    ) / 2.0) as i32;

    let mut result = HashMap::new();
    for (grid_coord, (point_x, point_y)) in points {
        let mut sum: f32 = 0.0;
        for delta_x in -square_edge..=square_edge {
            for delta_y in -square_edge..=square_edge {
                sum += pixel_average(
                    &pixels,
                    (point_x as i32 + delta_x) as usize,
                    (point_y as i32 + delta_y) as usize,
                );
            }
        }

        result.insert(grid_coord, (sum / ((square_edge + 1) * (square_edge + 1)) as f32) as u8);
    }

    result
}

fn compute_signature(point_averages: HashMap<(i8, i8), u8>) -> Vec<i8> {
    let mut diff_matrix = Vec::new();
    for ((grid_x, grid_y), gray) in &point_averages {
        let raw_point_diffs: Vec<i16> = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0), (1, 0),
            (-1, 1), (0, 1), (1, 1)
        ].iter().filter_map(|delta_x, delta_y| {
            if let Some(other) = point_averages.get(&(*grid_x + delta_x, *grid_y + delta_y)) {
                Some(compute_diff(*gray, *other))
            } else {
                None
            }
        }).collect();

        let (mut dark, mut light): (Vec<i16>, Vec<i16>) = raw_point_diffs.iter()
            .filter(|d| **d != 0)
            .partition(|d| **d < 0);

        let dark_threshold = get_median(dark);
        let light_threshold = get_median(light);

        let collapsed: Vec<i8> = raw_point_diffs.into_iter()
            .map(|v| {
                if v > 0 {
                    collapse(v, light_threshold)
                } else if v < 0 {
                    collapse(v, dark_threshold)
                } else {
                    0
                }
            })
            .collect();

        diff_matrix.push(collapsed);
    }

    diff_matrix.flatten().collect()
}

fn collapse(val: i16, threshold: i16) -> i8 {
    if val.abs() > threshold.abs() {
        2 * val.signum() as i8
    } else {
        val.signum() as i8
    }
}

fn get_median(mut vec: Vec<i16>) -> i16 {
    vec.sort();
    if vec.len() % 2 == 0 {
        if vec.is_empty() {
            0
        } else {
            (vec[(vec.len() / 2) - 1] + vec[vec.len() / 2]) / 2
        }
    } else {
        vec[vec.len() / 2]
    }
}

fn compute_diff(me: u8, other: u8) -> i16 {
    let raw_result = me as i16 - other as i16;
    if raw_result.abs() <= 2 {
        0
    } else {
        raw_result
    }
}

fn pixel_average(pixels: &Vec<Vec<u8>>, x: usize, y: usize) -> f32 {
    let sum = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1)
    ].map(|(delta_x, delta_y)|
        pixels[(y as i32 + delta_y) as usize][(x as i32 + delta_x) as usize] as f32
    ).sum();

    sum / 9
}

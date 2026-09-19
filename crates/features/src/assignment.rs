//! Optimal Bipartite Matching (Hungarian Algorithm / Munkres)
//!
//! Solves the assignment problem: given a cost matrix, find the minimum cost assignment
//! of rows to columns such that each row is assigned to at most one column.

/// Solves the assignment problem for a square or rectangular cost matrix.
/// Returns a list of (row, col) assignments.
pub fn hungarian_matching(cost_matrix: &[Vec<f64>]) -> Vec<(usize, usize)> {
    if cost_matrix.is_empty() || cost_matrix[0].is_empty() {
        return Vec::new();
    }

    let rows = cost_matrix.len();
    let cols = cost_matrix[0].len();
    let n = rows.max(cols);

    // 1. Flatten and pad matrix to square n x n
    let mut matrix = vec![0.0; n * n];
    let mut max_val = 0.0;

    // Find max value for padding
    for row in cost_matrix.iter().take(rows) {
        for &val in row.iter().take(cols) {
            if val > max_val {
                max_val = val;
            }
        }
    }

    for r in 0..n {
        for c in 0..n {
            let val = if r < rows && c < cols {
                cost_matrix[r][c]
            } else {
                max_val + 1.0 // Pad with large value
            };
            matrix[r * n + c] = val;
        }
    }

    // --- Munkres Algorithm ---

    // Step 1: Row reduction
    for r in 0..n {
        let mut min_val = f64::INFINITY;
        for c in 0..n {
            min_val = min_val.min(matrix[r * n + c]);
        }
        for c in 0..n {
            matrix[r * n + c] -= min_val;
        }
    }

    // Step 2: Initial starring
    // mask: 0=normal, 1=starred, 2=primed
    let mut mask = vec![0u8; n * n];
    let mut row_covered = vec![false; n];
    let mut col_covered = vec![false; n];

    for r in 0..n {
        for c in 0..n {
            if !row_covered[r] && !col_covered[c] && matrix[r * n + c].abs() < 1e-9 {
                mask[r * n + c] = 1;
                row_covered[r] = true;
                col_covered[c] = true;
            }
        }
    }
    row_covered.fill(false);
    col_covered.fill(false);

    let mut step = 3;
    let mut prime_rc = (0, 0); // Location of the found prime

    loop {
        match step {
            3 => {
                // Step 3: Cover columns containing a starred zero
                let mut count = 0;
                for r in 0..n {
                    for c in 0..n {
                        if mask[r * n + c] == 1 {
                            col_covered[c] = true;
                        }
                    }
                }
                for &covered in &col_covered {
                    if covered {
                        count += 1;
                    }
                }

                if count >= n {
                    break;
                } // Done
                step = 4;
            }
            4 => {
                // Step 4: Find a non-covered zero and prime it
                if let Some((r, c)) = find_uncovered_zero(&matrix, &row_covered, &col_covered, n) {
                    mask[r * n + c] = 2; // Prime it

                    // If there is a starred zero in this row
                    if let Some(star_c) = find_star_in_row(&mask, r, n) {
                        row_covered[r] = true;
                        col_covered[star_c] = false;
                        step = 4; // Continue in Step 4
                    } else {
                        // No starred zero in row -> augment path
                        prime_rc = (r, c);
                        step = 5;
                    }
                } else {
                    step = 6;
                }
            }
            5 => {
                // Step 5: Construct series of alternating primed and starred zeros
                let mut path = Vec::with_capacity(n);
                path.push(prime_rc);

                let mut curr_c = prime_rc.1;

                while let Some(r) = find_star_in_col(&mask, curr_c, n) {
                    path.push((r, curr_c));
                    // Find primed zero in this row
                    if let Some(c) = find_prime_in_row(&mask, r, n) {
                        path.push((r, c));
                        curr_c = c;
                    } else {
                        break;
                    }
                }

                // Augment path
                for &(r, c) in &path {
                    if mask[r * n + c] == 1 {
                        mask[r * n + c] = 0;
                    } else {
                        mask[r * n + c] = 1;
                    }
                }

                // Clear covers and primes
                row_covered.fill(false);
                col_covered.fill(false);
                for m in &mut mask {
                    if *m == 2 {
                        *m = 0;
                    }
                }

                step = 3;
            }
            6 => {
                // Step 6: Add min value to covered rows, subtract from uncovered cols
                let mut min_val = f64::INFINITY;
                for r in 0..n {
                    for c in 0..n {
                        if !row_covered[r] && !col_covered[c] {
                            min_val = min_val.min(matrix[r * n + c]);
                        }
                    }
                }

                for r in 0..n {
                    for c in 0..n {
                        if row_covered[r] {
                            matrix[r * n + c] += min_val;
                        }
                        if !col_covered[c] {
                            matrix[r * n + c] -= min_val;
                        }
                    }
                }
                step = 4;
            }
            _ => break,
        }
    }

    let mut assignments = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if mask[r * n + c] == 1 {
                assignments.push((r, c));
            }
        }
    }
    assignments
}

fn find_uncovered_zero(m: &[f64], rc: &[bool], cc: &[bool], n: usize) -> Option<(usize, usize)> {
    for r in 0..n {
        if rc[r] {
            continue;
        }
        for c in 0..n {
            if !cc[c] && m[r * n + c].abs() < 1e-9 {
                return Some((r, c));
            }
        }
    }
    None
}

fn find_star_in_row(mask: &[u8], r: usize, n: usize) -> Option<usize> {
    (0..n).find(|&c| mask[r * n + c] == 1)
}

fn find_star_in_col(mask: &[u8], c: usize, n: usize) -> Option<usize> {
    (0..n).find(|&r| mask[r * n + c] == 1)
}

fn find_prime_in_row(mask: &[u8], r: usize, n: usize) -> Option<usize> {
    (0..n).find(|&c| mask[r * n + c] == 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hungarian_simple() {
        let cost = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let matches = hungarian_matching(&cost);
        assert_eq!(matches.len(), 3);

        let mut total_cost = 0.0;
        for (r, c) in matches {
            total_cost += cost[r][c];
        }
        // Min cost should be 5.0 (1.0 + 2.0 + 2.0)
        assert_eq!(total_cost, 5.0);
    }
}

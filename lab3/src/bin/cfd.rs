use std::env;
use std::f32;
use std::time::Instant;

fn boundary_psi(psi: &mut [f32], m: i32, n: i32, b: i32, h: i32, w: i32) {
    for i in (b + 1)..(b + w) {
        psi[(i * (m + 2)) as usize] = (i - b) as f32;
    }

    for i in (b + w)..(m + 1) {
        psi[(i * (m + 2)) as usize] = w as f32;
    }

    for i in 1..(h + 1) {
        psi[((m + 1) * (m + 2) + i) as usize] = w as f32;
    }

    for i in (h + 1)..(h + w) {
        psi[((m + 1) * (m + 2) + i) as usize] = (w - i + h) as f32;
    }
}

fn jacobi_step(psi: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut psi_tmp = psi.to_vec();

    for i in 1..=m {
        for j in 1..=n {
            psi_tmp[i * (m + 2) + j] = 0.25
                * (psi[i * (m + 2) + j - 1]
                    + psi[i * (m + 2) + j + 1]
                    + psi[(i - 1) * (m + 2) + j]
                    + psi[(i + 1) * (m + 2) + j]);
        }
    }

    return psi_tmp;
}

fn deltasq(psi_tmp: &[f32], psi: &[f32], m: usize, n: usize) -> f32 {
    let mut sum = 0.0;

    for i in 1..=m {
        for j in 1..=n {
            let diff = psi_tmp[i * (m + 2) + j] - psi[i * (m + 2) + j];
            sum += diff * diff;
        }
    }

    return sum;
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let scale_factor: usize = args[1].parse().unwrap();
    let num_iter: usize = args[2].parse().unwrap();
    let print_freq: usize = 10;
    let tol: f32 = 0.0;
    let bbase: usize = 10;
    let hbase: usize = 15;
    let wbase: usize = 5;
    let mbase: usize = 32;
    let nbase: usize = 32;
    let irrotational: usize = 1;
    let mut checkerr: usize = 0;

    if tol > 0.0 {
        checkerr = 1;
    }

    if checkerr == 0 {
        println!("scale factor = {}, iterations = {}", scale_factor, num_iter);
    } else {
        println!(
            "scale factor = {}, iterations = {}, tol = {}",
            scale_factor, num_iter, tol
        );
    }

    let b = bbase * scale_factor;
    let h = hbase * scale_factor;
    let w = wbase * scale_factor;
    let m = mbase * scale_factor;
    let n = nbase * scale_factor;

    println!("Running CFD on {} x {} grid", m, n);

    let mut psi = vec![0.0; (m + 2) * (n + 2)];
    boundary_psi(&mut psi, m as i32, n as i32, b as i32, h as i32, w as i32);

    let bnorm = psi.iter().map(|&x| x * x).sum::<f32>().sqrt();

    println!("\nStarting main loop...\n");
    let t_start = Instant::now();

    let mut error = 0.0;
    let mut start = Instant::now();

    for i in 1..=num_iter {
        let psi_tmp = jacobi_step(&psi, m, n);

        if checkerr != 0 || i == num_iter {
            error = deltasq(&psi_tmp, &psi, m, n).sqrt() / bnorm;
        }

        if checkerr != 0 && error < tol {
            println!("Converged on iter {}", i);
            break;
        }

        for i in 1..=m {
            for j in 1..=n {
                psi[i * (m + 2) + j] = psi_tmp[i * (m + 2) + j];
            }
        }

        if i % print_freq == 0 {
            if checkerr == 0 {
                println!("Completed iter {}, rate = {:?}", i, start.elapsed() / 10);
                start = Instant::now();
            } else {
                println!("Completed iter {}, error = {}", i, error);
            }
        }
    }

    let t_stop = Instant::now();
    let t_tot = t_stop.duration_since(t_start);
    let t_iter = t_tot / num_iter as u32;

    println!("... finished");
    println!("After {} iterations, the error is {}", num_iter, error);
    println!("Time for {} iterations was {:?} seconds", num_iter, t_tot);
    println!("Each iteration took {:?} seconds", t_iter);
}

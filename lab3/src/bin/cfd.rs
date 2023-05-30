extern crate ocl;

use ocl::ProQue;
use std::env;
use std::f64;
use std::time::Instant;

fn cfd_parallel(scale_factor: i32, num_iterations: i32) {
    let src: &str = r#"
        __kernel void jacobi(__global double *psi_tmp, __global const double *psi, int m, int n) {
            int i = get_global_id(0) + 1;
            int j = get_global_id(1) + 1;

            if (i <= m && j <= n) {
                psi_tmp[i * (m + 2) + j] = 0.25 * (
                    psi[(i - 1) * (m + 2) + j] + 
                    psi[(i + 1) * (m + 2) + j] + 
                    psi[i * (m + 2) + j - 1] + 
                    psi[i * (m + 2) + j + 1]
                );
            }
        }

        __kernel void copy_arrays(__global const double *psi_tmp, __global double *psi, int m, int n)
        {
            int i = get_global_id(0) + 1;
            int j = get_global_id(1) + 1;

            psi[i * (m + 2) + j] = psi_tmp[i * (m + 2) + j];
        }
    "#;

    let print_frequency: i32 = 10;

    let b_base: i32 = 10;
    let h_base: i32 = 15;
    let w_base: i32 = 5;
    let m_base: i32 = 32;
    let n_base: i32 = 32;

    let b: i32 = b_base * scale_factor;
    let h: i32 = h_base * scale_factor;
    let w: i32 = w_base * scale_factor;
    let m: i32 = m_base * scale_factor;
    let n: i32 = n_base * scale_factor;

    println!("Running CFD on {} x {} grid", m, n);

    let mut psi: Vec<f64> = vec![0.0; ((m + 2) * (n + 2)) as usize];
    let mut psi_tmp: Vec<f64> = psi.clone();

    boundary_psi(&mut psi, m as i32, b as i32, h as i32, w as i32);

    let b_norm: f64 = psi.iter().map(|&x| x * x).sum::<f64>().sqrt();

    println!("Starting the main loop...");

    let mut error: f64 = 0.0;
    let start: Instant = Instant::now();

    let ocl_pq: ProQue = ProQue::builder()
        .src(src)
        .dims((m + 2, n + 2))
        .build()
        .unwrap();

    let psi_buffer: ocl::Buffer<f64> = ocl_pq
        .buffer_builder::<f64>()
        .copy_host_slice(&psi)
        .build()
        .unwrap();

    let psi_tmp_buffer: ocl::Buffer<f64> = ocl_pq
        .buffer_builder::<f64>()
        .copy_host_slice(&psi_tmp)
        .build()
        .unwrap();

    for i in 1..=num_iterations {
        let kernel: ocl::Kernel = ocl_pq
            .kernel_builder("jacobi")
            .global_work_size((m, n))
            .arg_named("psi_tmp", &psi_tmp_buffer)
            .arg_named("psi", &psi_buffer)
            .arg_named("m", &(m as i32))
            .arg_named("n", &(n as i32))
            .build()
            .unwrap();

        unsafe {
            kernel.enq().unwrap();
        }

        if i == num_iterations {
            psi_tmp_buffer.read(&mut psi_tmp).enq().unwrap();
            psi_buffer.read(&mut psi).enq().unwrap();

            error = deltasq(&psi_tmp, &psi, m, n).sqrt() / b_norm;
        }

        let kernel: ocl::Kernel = ocl_pq
            .kernel_builder("copy_arrays")
            .global_work_size((m, n))
            .arg_named("psi_tmp", &psi_tmp_buffer)
            .arg_named("psi", &psi_buffer)
            .arg_named("m", &m)
            .arg_named("n", &n)
            .build()
            .unwrap();

        unsafe {
            kernel.enq().unwrap();
        }

        if i % print_frequency == 0 {
            println!(
                "Iteration number {i}, {:?} per iteration",
                start.elapsed() / i as u32
            )
        }

        ocl_pq.finish().unwrap();
    }

    println!("Finished main loop...");

    println!(
        "After {} iterations, the error is {}",
        num_iterations, error
    );
    println!(
        "Time for {} iterations was {:?}",
        num_iterations,
        start.elapsed()
    );
    println!(
        "Each iteration took {:?}",
        start.elapsed() / num_iterations as u32
    );
}

fn deltasq(psi_tmp: &[f64], psi: &[f64], m: i32, n: i32) -> f64 {
    let mut sum: f64 = 0.0;

    for i in 1..=m {
        for j in 1..=n {
            let diff: f64 = psi_tmp[(i * (m + 2) + j) as usize] - psi[(i * (m + 2) + j) as usize];
            sum += diff * diff;
        }
    }

    return sum;
}

fn copy_array(m: i32, n: i32, psi: &mut [f64], psi_tmp: &mut [f64]) {
    for i in 1..=m {
        for j in 1..=n {
            psi[(i * (m + 2) + j) as usize] = psi_tmp[(i * (m + 2) + j) as usize];
        }
    }
}

fn jacobi(psi: &[f64], m: i32, n: i32) -> Vec<f64> {
    let mut psi_tmp: Vec<f64> = psi.to_vec();

    for i in 1..=m {
        for j in 1..=n {
            psi_tmp[(i * (m + 2) + j) as usize] = 0.25
                * (psi[(i * (m + 2) + j - 1) as usize]
                    + psi[(i * (m + 2) + j + 1) as usize]
                    + psi[((i - 1) * (m + 2) + j) as usize]
                    + psi[((i + 1) * (m + 2) + j) as usize]);
        }
    }

    return psi_tmp;
}

fn boundary_psi(psi: &mut [f64], m: i32, b: i32, h: i32, w: i32) {
    for i in (b + 1)..(b + w) {
        psi[(i * (m + 2)) as usize] = (i - b) as f64;
    }

    for i in (b + w)..(m + 1) {
        psi[(i * (m + 2)) as usize] = w as f64;
    }

    for i in 1..(h + 1) {
        psi[((m + 1) * (m + 2) + i) as usize] = w as f64;
    }

    for i in (h + 1)..(h + w) {
        psi[((m + 1) * (m + 2) + i) as usize] = (w - i + h) as f64;
    }
}

fn cfd_sequential(scale_factor: i32, num_iterations: i32) {
    let print_frequency: i32 = 10;

    let b_base: i32 = 10;
    let h_base: i32 = 15;
    let w_base: i32 = 5;
    let m_base: i32 = 32;
    let n_base: i32 = 32;

    let b: i32 = b_base * scale_factor;
    let h: i32 = h_base * scale_factor;
    let w: i32 = w_base * scale_factor;
    let m: i32 = m_base * scale_factor;
    let n: i32 = n_base * scale_factor;

    println!("Running CFD on {} x {} grid", m, n);

    let mut psi: Vec<f64> = vec![0.0; ((m + 2) * (n + 2)) as usize];

    boundary_psi(&mut psi, m as i32, b as i32, h as i32, w as i32);

    let b_norm: f64 = psi.iter().map(|&x| x * x).sum::<f64>().sqrt();

    println!("Starting the main loop...");

    let mut error: f64 = 0.0;
    let start: Instant = Instant::now();

    for i in 1..=num_iterations {
        let mut psi_tmp: Vec<f64> = jacobi(&psi, m, n);

        if i == num_iterations {
            error = deltasq(&psi_tmp, &psi, m, n).sqrt() / b_norm;
        }

        copy_array(m, n, &mut psi, &mut psi_tmp);

        if i % print_frequency == 0 {
            println!(
                "Iteration number {i}, {:?} per iteration",
                start.elapsed() / i as u32
            )
        }
    }

    println!("Finished main loop...");

    println!(
        "After {} iterations, the error is {}",
        num_iterations, error
    );
    println!(
        "Time for {} iterations was {:?}",
        num_iterations,
        start.elapsed()
    );
    println!(
        "Each iteration took {:?}",
        start.elapsed() / num_iterations as u32
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let scale_factor: i32 = args[1].parse().unwrap();
    let num_iterations: i32 = args[2].parse().unwrap();
    let mode: String = args[3].parse().unwrap();

    match mode.as_str() {
        "p" => cfd_parallel(scale_factor, num_iterations),
        "s" => cfd_sequential(scale_factor, num_iterations),
        _ => panic!("invalid mode {}", mode),
    };
}

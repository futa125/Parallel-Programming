use std::env;
use std::f64;
use std::time::Instant;

use ocl::ProQue;

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

fn deltasq(psi_tmp: &[f64], psi: &[f64], m: usize, n: usize) -> f64 {
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
    let src: &str = r#"
    __kernel void jacobistep(__global double *psinew, __global const double *psi, int m, int n) {
        int i = get_global_id(0) + 1;
        int j = get_global_id(1) + 1;
        if (i <= m && j <= n) {
            psinew[i*(m+2)+j] = 0.25 * (psi[(i-1)*(m+2)+j] + psi[(i+1)*(m+2)+j] + psi[i*(m+2)+j-1] + psi[i*(m+2)+j+1]);
        }
    }
    __kernel void copy_arrays(__global const double *psinew, __global double *psi, int m, int n)
    {
        int i = get_global_id(0) + 1;
        int j = get_global_id(1) + 1;
        psi[i*(m+2)+j]=psinew[i*(m+2)+j];

    }
    __kernel void boundarypsi(__global double *psi, int m, int n, int b, int h, int w)
    {    
        int i,j;
        for (i=b+1;i<=b+w-1;i++)
        {
            psi[i*(m+2)+0] = (double)(i-b);
        }
        for (i=b+w;i<=m;i++)
        {
            psi[i*(m+2)+0] = (double)(w);
        }
        for (j=1; j <= h; j++)
        {
            psi[(m+1)*(m+2)+j] = (double) w;
        }
        for (j=h+1;j<=h+w-1; j++)
        {
            psi[(m+1)*(m+2)+j]=(double)(w-j+h);
        }
    }
    "#;
    let args: Vec<String> = env::args().collect();
    let scale_factor: usize = args[1].parse().unwrap();
    let num_iter: usize = args[2].parse().unwrap();
    let print_freq: usize = 10;
    let tol: f64 = 0.0;
    let bbase: usize = 10;
    let hbase: usize = 15;
    let wbase: usize = 5;
    let mbase: usize = 32;
    let nbase: usize = 32;
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

    let mut psi: Vec<f64> = vec![0.0; (m + 2) * (n + 2)];
    let mut psi_tmp: Vec<f64> = psi.clone();
    boundary_psi(&mut psi, m as i32, b as i32, h as i32, w as i32);

    let bnorm = psi.iter().map(|&x| x * x).sum::<f64>().sqrt();

    println!("\nStarting main loop...\n");
    let t_start = Instant::now();

    let mut error = 0.0;
    let mut start = Instant::now();

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
    let psinew_buffer: ocl::Buffer<f64> = ocl_pq
        .buffer_builder::<f64>()
        .copy_host_slice(&psi)
        .build()
        .unwrap();

    for i in 1..=num_iter {
        let kernel: ocl::Kernel = ocl_pq
            .kernel_builder("jacobistep")
            .global_work_size((m, n))
            .arg_named("psinew", &psinew_buffer)
            .arg_named("psi", &psi_buffer)
            .arg_named("m", &(m as i32))
            .arg_named("n", &(n as i32))
            .build()
            .unwrap();

        unsafe {
            kernel.enq().unwrap();
        }

        if checkerr != 0 || i == num_iter {
            psinew_buffer.read(&mut psi_tmp).enq().unwrap();
            psi_buffer.read(&mut psi).enq().unwrap();
            error = deltasq(&psi_tmp, &psi, m, n).sqrt() / bnorm;
        }

        if checkerr != 0 && error < tol {
            println!("Converged on iter {}", i);
            break;
        }

        let kernel: ocl::Kernel = ocl_pq
            .kernel_builder("copy_arrays")
            .global_work_size((m, n))
            .arg_named("psinew", &psinew_buffer)
            .arg_named("psi", &psi_buffer)
            .arg_named("m", &(m as i32))
            .arg_named("n", &(n as i32))
            .build()
            .unwrap();

        unsafe {
            kernel.enq().unwrap();
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

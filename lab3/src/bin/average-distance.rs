extern crate ocl;
extern crate rand;

use ocl::{
    flags::{MEM_COPY_HOST_PTR, MEM_READ_ONLY, MEM_WRITE_ONLY},
    Error, ProQue,
};
use rand::{rngs, Rng};
use std::time::Instant;

fn avg_distance(n: usize, m: usize) -> Result<f32, Error> {
    let src: &str = r#"
        __kernel void avg_distance(const int n, const __global float* x, const __global float* y, __global float* distances) {
            int id = get_global_id(0);
            int size = get_global_size(0);

            for (int i = id; i < n; i += size) {
                float sum = 0.0f;

                for (int j = i + 1; j < n; j++) {
                    float dx = x[i] - x[j];
                    float dy = y[i] - y[j];

                    sum += sqrt(dx * dx + dy * dy);
                }

                distances[i] = sum / (n - i);
            }
        }
    "#;

    let mut rng: rngs::ThreadRng = rand::thread_rng();

    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0..=1.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0..=1.0)).collect();
    let mut distances: Vec<f32> = vec![0.0; n];

    let ocl_pq: ProQue = ProQue::builder().src(src).dims(n).build()?;

    let x_buffer: ocl::Buffer<f32> = ocl_pq
        .buffer_builder::<f32>()
        .copy_host_slice(&x)
        .flags(MEM_READ_ONLY | MEM_COPY_HOST_PTR)
        .build()?;
    let y_buffer: ocl::Buffer<f32> = ocl_pq
        .buffer_builder::<f32>()
        .copy_host_slice(&y)
        .flags(MEM_READ_ONLY | MEM_COPY_HOST_PTR)
        .build()?;
    let distances_buffer: ocl::Buffer<f32> = ocl_pq
        .buffer_builder::<f32>()
        .flags(MEM_WRITE_ONLY)
        .build()?;

    let kernel: ocl::Kernel = ocl_pq
        .kernel_builder("avg_distance")
        .global_work_size(n / m)
        .arg_named("n", &n)
        .arg_named("x", &x_buffer)
        .arg_named("y", &y_buffer)
        .arg_named("distances", &distances_buffer)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    distances_buffer.read(&mut distances).enq()?;

    return Ok(distances.iter().sum::<f32>() / n as f32);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        println!("Usage: cargo run N M");
        std::process::exit(1);
    }

    let n = args[1].parse::<usize>().unwrap();
    let m = args[2].parse::<usize>().unwrap();

    let start = Instant::now();

    let avg_dist: Result<f32, Error> = avg_distance(n, m);

    println!("Elapsed Time: {:?}", start.elapsed());
    println!("Average distance: {}", avg_dist.unwrap());
}

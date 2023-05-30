extern crate ocl;
extern crate rand;

use ocl::{flags::MEM_WRITE_ONLY, Error, ProQue};
use std::time::Instant;

fn calculate_pi(n: i32, m: i32, l: i32) -> Result<f64, Error> {
    let src: &str = r#"
    __kernel void calculate_pi(__global float* sums, const float h, const int n, const int m) {
        int id = get_global_id(0);
        int start = id * m + 1;
        int end = min(start + m, n + 1);
        float x, sum = 0.0;
    
        for(int i = start; i < end; i++) {
            x = h * ((float)i - 0.5);
            sum += 4.0 / (1.0 + x*x);
        }
        
        sums[id] = sum;
    }
    "#;

    let h: f32 = 1.0 / n as f32;
    let work_groups: i32 = (n as f32 / m as f32).round() as i32;

    let mut sums: Vec<f32> = vec![0.0; work_groups as usize];

    let ocl_pq: ProQue = ProQue::builder().src(src).dims(n).build()?;

    let sums_buffer: ocl::Buffer<f32> = ocl_pq
        .buffer_builder::<f32>()
        .flags(MEM_WRITE_ONLY)
        .build()?;

    let kernel: ocl::Kernel = ocl_pq
        .kernel_builder("calculate_pi")
        .global_work_size(work_groups)
        .local_work_size(l)
        .arg_named("sums", &sums_buffer)
        .arg_named("h", &h)
        .arg_named("n", &n)
        .arg_named("m", &m)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    sums_buffer.read(&mut sums).enq()?;

    return Ok(h as f64 * sums.iter().map(|&x| x as f64).sum::<f64>());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        println!("Usage: cargo run N M L");
        std::process::exit(1);
    }

    let n: i32 = args[1].parse::<i32>().unwrap();
    let m: i32 = args[2].parse::<i32>().unwrap();
    let l: i32 = args[3].parse::<i32>().unwrap();

    let start = Instant::now();

    let pi: Result<f64, Error> = calculate_pi(n, m, l);

    println!("Elapsed Time: {:?}", start.elapsed());
    println!("Pi: {:}", pi.unwrap());
}

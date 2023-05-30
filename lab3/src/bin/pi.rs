extern crate ocl;

use ocl::{flags::MEM_WRITE_ONLY, Error, ProQue};
use std::time::Instant;

fn calculate_pi_sequential(n: i32) -> f64 {
    let h: f64 = 1.0 / n as f64;
    let mut sum: f64 = 0.0;

    for i in 1..=n {
        let x = h * (i as f64 - 0.5);
        sum += h * (4.0 / (1.0 + x * x));
    }

    return sum;
}

fn calculate_pi_parallel(n: i32, m: i32, l: i32) -> Result<f64, Error> {
    let src: &str = r#"
        __kernel void calculate_pi(__global double* sums, const int n, const int m) {
            int id = get_global_id(0);
            int start = id * m + 1;
            int end = min(start + m, n + 1);
            double x, sum = 0.0;
            double h = 1.0 / (double)n;
        
            for(int i = start; i < end; i++) {
                x = h * ((double)i - 0.5);
                sum += 4.0 / (1.0 + x*x);
            }
            
            sums[id] = h * sum;
        }
    "#;

    let work_groups: i32 = n / m;

    let mut sums: Vec<f64> = vec![0.0; work_groups as usize];

    let ocl_pq: ProQue = ProQue::builder().src(src).dims(n).build()?;

    let sums_buffer: ocl::Buffer<f64> = ocl_pq
        .buffer_builder::<f64>()
        .flags(MEM_WRITE_ONLY)
        .build()?;

    let kernel: ocl::Kernel = ocl_pq
        .kernel_builder("calculate_pi")
        .global_work_size(work_groups)
        .local_work_size(l)
        .arg_named("sums", &sums_buffer)
        .arg_named("n", &n)
        .arg_named("m", &m)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    sums_buffer.read(&mut sums).enq()?;

    return Ok(sums.iter().sum::<f64>());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        println!("Usage: cargo run N M L mode");
        std::process::exit(1);
    }

    let n: i32 = args[1].parse::<i32>().unwrap();
    let m: i32 = args[2].parse::<i32>().unwrap();
    let l: i32 = args[3].parse::<i32>().unwrap();
    let mode: String = args[4].parse::<String>().unwrap();

    let start = Instant::now();

    let pi = match mode.as_str() {
        "p" => calculate_pi_parallel(n, m, l).unwrap(),
        "s" => calculate_pi_sequential(n),
        _ => panic!("invalid mode {}", mode),
    };

    println!("Elapsed Time: {:?}", start.elapsed());
    println!("Pi: {pi}");
}

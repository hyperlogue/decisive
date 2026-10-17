use std::{
    env, fs,
    io::{Read, Write},
    net::TcpStream,
    sync::Barrier,
    thread,
    time::Instant,
};
fn request(port: u16, body: &str) -> f64 {
    let begin = Instant::now();
    let mut socket = TcpStream::connect(("127.0.0.1", port)).unwrap();
    socket
        .set_read_timeout(Some(std::time::Duration::from_secs(120)))
        .unwrap();
    write!(socket, "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
    let mut response = Vec::new();
    socket.read_to_end(&mut response).unwrap();
    assert!(
        response.starts_with(b"HTTP/1.1 200") || response.starts_with(b"HTTP/1.0 200"),
        "HTTP request failed: {}",
        String::from_utf8_lossy(&response)
    );
    begin.elapsed().as_secs_f64() * 1000.0
}
fn main() {
    let args: Vec<_> = env::args().filter(|arg| arg != "--bench").collect();
    let port = args[1].parse().unwrap();
    let corpus = fs::read_to_string(&args[2]).unwrap();
    let bodies: Vec<_> = corpus.lines().filter(|s| !s.is_empty()).collect();
    let workers: usize = args[3].parse().unwrap();
    let rounds: usize = args[4].parse().unwrap();
    assert!(workers > 0 && rounds > 0 && !bodies.is_empty());
    for _ in 0..3 {
        for body in &bodies {
            request(port, body);
        }
    }
    let barrier = Barrier::new(workers + 1);
    let (elapsed, samples) = thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|worker| {
                let bodies = &bodies;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    (0..rounds)
                        .map(|i| request(port, bodies[(i * workers + worker) % bodies.len()]))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let begin = Instant::now();
        barrier.wait();
        let samples: Vec<_> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        (begin.elapsed().as_secs_f64(), samples)
    });
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let percentile = |p: f64| sorted[(p * sorted.len() as f64).ceil() as usize - 1];
    println!(
        "{{\"workers\":{workers},\"requests\":{},\"elapsed_seconds\":{elapsed},\"requests_per_second\":{},\"p50_ms\":{},\"p95_ms\":{},\"latencies_ms\":{:?}}}",
        samples.len(),
        samples.len() as f64 / elapsed,
        percentile(0.5),
        percentile(0.95),
        samples
    );
}
